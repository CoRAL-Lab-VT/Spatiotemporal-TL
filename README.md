# A Transferable Deep Learning Framework to Propagate Extreme Water Levels from Sparse Tide-Gauges across Spatial Domains

**Spatiotemporal Deep Learning for Extreme Water Level Mapping**

## 📌 Overview

This repository implements an end-to-end pipeline to predict high-resolution water level maps over a geographic region using:

- Gridded geophysical drivers (atmospheric pressure, rainfall, wind, river discharge, DEM)
- Sparse in-situ water level station measurements
- Historical water level rasters (the prediction target)

Three modeling approaches are provided:

### 1. UNet-style ConvLSTM Segmentation Model (UNET)
A U-Net with temporal encoding and attention gating to regress water level at every pixel.

### 2. ASTGCN / Graph-Temporal Transformer Hybrid (GGCN)
A graph convolutional network on a dynamic water surface graph, fused with temporal attention and spatial features. 

### 3. Attention-Boosted GCN + LSTM + CBAM/SAM (AbGGCN)
An improved graph model with channel/spatial attention (CBAM), temporal attention, and sequence modeling. 
All three models share:
- A consistent preprocessing / normalization pipeline  
- Sequence-to-one temporal supervision (past `seq_length` frames → next-frame water level map)  
- Masked loss (only evaluate valid water pixels, ignore land / nodata)  
- Multi-GPU training via `tf.distribute.MirroredStrategy`

---

## 📥 Data Inputs

### 1. Gridded Forcing Variables (`atm_pressure`, `wind_speed`, `precipitation`, `river_discharge`)

Each directory contains a time series of georeferenced `.tif` rasters. Each file corresponds to one timestep. Files are loaded in natural-sorted order (`0, 1, 2, ... 10, 11` instead of lexicographic `0,1,10,11,2...`).

Each raster is read with `rasterio`:

```python
with rasterio.open(filepath) as src:
    img = src.read(1)          # 2D array [H, W]
    crs = src.crs              # coordinate reference system
    transform = src.transform  # affine georeferencing
```

These channels are later stacked into a 5-channel tensor along with DEM.

---

### 2. DEM (`DEM/dem_idw.tif`, `DEM/dem_idw2.tif`)

- Elevation is static in space for an event, but two or more DEM variants can be used.  
- The code tiles `dem_idw.tif` for the first `n1=217` timesteps, and `dem_idw2.tif` for the remaining timesteps, then concatenates along time to align with the meteorological data length.

Result: a `(T, H, W)` DEM “timeseries” that matches other inputs.

---

### 3. In-situ Gauge Data (`training_water_level/`)

Each CSV represents one gauging station:

```text
x, y, water_level
...
```

- `(x,y)` are station coordinates in the same CRS as the rasters.  
- `water_level` is a time series column.

All station CSVs are combined into:
- `station_df`: shape `[T, num_stations]`
- `station_coords`: `{ station_name: (x, y) }`

These station values are spatially embedded back into the grid:
- Either as one-hot “pins” per station (for UNet)
- Or merged into a single extra channel (for the GCN variants)

See **Station Embedding** below.

---

### 4. Water Level Maps (`training_water_level_map/`)

These `.tif` rasters are the **supervision target** — gridded water level at each timestep.

They are also used to derive:
- A binary **valid-water mask** (pixels that are water vs nodata/land/invalid)
- Min/max scaling stats for denormalization later

---

## 🧼 Preprocessing & Normalization

All three models follow roughly the same data prep steps.

### 1. Natural sort & stack

Each data source (`atm_pressure`, etc.) is loaded into a `(T, H, W)` array, then we stack features into one tensor:

```python
spatial_feats = np.stack([
    atm_pressure, wind_speed, precipitation, river_discharge, dem
], axis=-1)  # -> shape (T, H, W, C=5)
```

---

### 2. Padding to be divisible by network stride

- For UNet we pad to multiples of 8.  
- For GCN models we pad to multiples of 4.  

Padding is symmetric reflect-padding around the borders:

```python
pad_h = (8 - H % 8) % 8
top_pad = pad_h // 2
spatial_padded = np.pad(
    spatial_feats,
    ((0,0),(top_pad, pad_h-top_pad),(left_pad, pad_w-left_pad),(0,0)),
    mode='reflect'
)
```

This gives cleaner downsampling/upsampling and also aligns tensors for graph reshaping.

We record `top_pad`, `left_pad`, etc. for later unpadding/inference.

---

### 3. Masked min–max normalization → `[0.1, 1.0]`

Each predictor channel is normalized per-channel using only **valid water pixels** (land/nodata is ignored by masking them as NaN during the min/max calculation). Then any remaining NaNs get filled with 0.0:

```python
normed = 0.1 + 0.9 * (feat - mn) / (mx - mn)
feat_norm = np.nan_to_num(normed, nan=0.0)
```

The same `[0.1,1.0]` scaling is also used for station levels and target water levels.

We persist these stats to `norm_stats.npz` in each model’s output dir so you can:
- Denormalize predictions back to meters  
- Reuse the same scaling at inference

Saved stats include:
- Per-channel mins/maxs for spatial predictors  
- Station min/max  
- Water level min/max  
- Padding info (`top_pad`, `left_pad`, etc.)  
- Sequence length, grid shape

---

### 4. Temporal windowing

All models use an **auto-regressive forecasting pattern**:

**Given:**
- Past `seq_length` timesteps of forcing + station inputs

**Predict:**
- Water level map at time `t = seq_length` (the “next” frame)

We build samples like:

```python
for i in range(len(spatial_data) - seq_length):
    X_spatial[i] = spatial_data[i : i+seq_length]        # shape (seq, H, W, C)
    X_station[i] = station_series[i : i+seq_length]      # shape (seq, num_stations)
    y[i]         = wl_map[i+seq_length]                  # shape (H, W)
    y_mask[i]    = valid_mask[i+seq_length]              # shape (H, W), True where invalid
```

Then we split chronologically:  
- first 80% timesteps for train  
- last 20% for validation  

(no shuffle, to avoid leakage).

We also keep a per-sample mask to weight loss only on valid pixels.

---

## 📍 Station Embedding

Sparse gauge readings are critical for water level propagation. We inject the station information spatially so models learn how gauges influence the spatial domain.

There are two strategies in this repo:

### A. Multi-channel “pin map” embedding (UNet)

1. Convert each station point `(x,y)` to grid row/col using `rasterio.transform.rowcol`.  
2. Create one binary mask channel per station (1.0 at that pixel, else 0).  
3. For each timestep and each station, multiply station value by that mask to retain only the reading into the raster at the correct location.  
4. Concatenate these station maps with the meteorological channels, per timestep.

This produces an input tensor like:

```text
(seq_length, H, W, C + num_stations)
```

### B. Single fused station channel (GCN / ASTGCN / AbGGCN)

Instead of one channel per station, we build a single extra channel per timestep where each station’s normalized level is written into its grid cell and zeros elsewhere. That fused channel is then appended to the spatial predictors.

This keeps the feature dimension smaller and fits nicely into graph-style processing.

---

## 🕸 Graph Construction (for GCN-style models)

For graph-based models (`GGCN.py`, `AbGGCN.py`), every valid pixel in the padded grid is treated as a node in a graph.

### Nodes
Each node corresponds to a spatial cell `(i,j)` that is not masked out (i.e. where water levels are meaningful).

### Edges
- 8-neighborhood connectivity (N,S,E,W + diagonals).  
- Only connect neighbors that are also valid water cells.  
- Store this as a sparse adjacency matrix `A` (TensorFlow `SparseTensor`).

### Normalization
We compute the common GCN normalization so message passing is numerically stable.

This adjacency and normalization are reused across all samples because topology is spatially fixed.

---

## 🎯 Loss Function (Masked MSE)

All models predict a **continuous water level raster**.

But:
- Some pixels are invalid water/no-water/nodata.
- We don’t want those to contribute to loss.

So labels are packed as:

```python
y_true[..., 0] = normalized_water_level
y_true[..., 1] = valid_mask   # 1 where valid, 0 where invalid
```

Then `masked_mse` is:

```python
se   = (y_pred - y_val)**2
num  = sum(se * mask)
den  = sum(mask) + 1e-7
loss = num / den
```

This is computed per-sample, averaged over batch.

---

## 🧠 Model 1: ConvLSTM U-Net with Attention Gates (`UNET.py`)

This model treats the problem like image-to-image translation with temporal context:

- **Encoder:** per-timestep 2D convolutions wrapped in `TimeDistributed` blocks to learn local hydrodynamic structure.  
- **Bottleneck:** `ConvLSTM2D` to model spatiotemporal evolution across the `seq_length` history.  
- **Decoder:** classic U-Net upsampling with skip connections.  
- **Attention gates** on the skip connections to suppress irrelevant spatial regions before concatenation.

### Key components

- **`td_conv_block`:**  
  A `TimeDistributed` Conv → LayerNorm → ReLU → Dropout → Conv → LayerNorm → ReLU block.  
  Learns spatial features for each timestep independently but shares weights across time.

- **ConvLSTM bottleneck:**  
  Integrates the temporal axis.

- **Attention Gate:**
  ```python
  def attention_gate(x, g, inter_channels):
      theta_x = Conv2D(inter_channels, 1)(x)
      phi_g   = Conv2D(inter_channels, 1)(g)
      alpha   = sigmoid( Conv2D(1,1)( relu(theta_x + phi_g) ) )
      return x * alpha
  ```
  This focuses skip information on areas relevant to the decoder stage.

- **LayerNormalization everywhere** instead of BatchNorm (safer for small batch / multi-GPU sync).

- **L2 weight decay + configurable dropout,** exposed to KerasTuner hyperparameters (`base_filters`, `dropout_rate`, `l2_weight`).

### Inputs

```text
data_input: (seq_length, H_pad, W_pad, channels_with_station)
mask_input: (H_pad, W_pad)  # to help model learn valid vs invalid
```

### Outputs

```text
predicted_water_level: (H_pad, W_pad, 1)
```

### Training

- Strategy: `tf.distribute.MirroredStrategy()` for multi-GPU.  
- Loss: custom `masked_mse`.  
- Callbacks:
  - `ModelCheckpoint`
  - `ReduceLROnPlateau`
  - (optionally) Keras Tuner for hyperparam search.

---

## 🧠 Model 2: ASTGCN / Temporal Attention / Graph Fusion (`GGCN.py`)

> Saved to `ASTGCN_OUTPUTS/`

This model reframes the raster as a graph and mixes three ideas:

1. **Spatial graph convolution** over valid water cells.  
2. **Temporal modeling** across the `seq_length` window.  
3. **Transformer-style / Multi-Head Attention** over time and space for dynamic weighting.

### Pipeline sketch

**Input assembly**  
Combine:
- Normalized spatial predictors `[atm_pressure, wind_speed, precip, discharge, DEM]`
- Embedded station channel

into shape `(B, seq_length, H_pad, W_pad, F_total)`.

**Grid → Graph projection**  
Flatten `H_pad x W_pad` into `N` nodes and gather only valid nodes using the precomputed mask.  
Keep both:
- Static features (elevation, etc.)
- Dynamic water-related features.

**`GatedGraphConvolution` layer**  
Custom layer that:
- Computes an adaptive gate per node from static features.
- Multiplies that gate with dynamic water features.
- Propagates with normalized sparse adjacency.
- Applies Dense + LayerNorm + Dropout.

This is conceptually similar to a GCN but with learnable per-node gating to emphasize hydrologically important cells (`gate_dense_1`, `gate_dense_2`, `feature_dense`, etc.).

**Temporal attention / transformer reshape**  
The code defines helper layers like:
- `ReshapeForTransformer`: `(B, T, N, F)` → `(B*N, T, F)`
- `MultiHeadAttention` across the T dimension
- `ReshapePooled` and `ReshapeBackToSpatial` to project features back to spatial layout.

These steps let the model learn which timesteps in the history matter most for each spatial node.

**Prediction head**  
After fusing graph output + static context, we reshape back to `(H_pad, W_pad)` and predict continuous water level for the next frame.  
A final sigmoid-then-rescale to `[0.1,1.0]` is defined via `rescale_sigmoid`.

### Extras

- **`GradientNormCallback`:** debug callback that measures gradient norm at epoch end, to monitor stability / exploding gradients.  
- **`sample_weights`:** per-sample scalar weight ~ variance of the target, to upweight dynamic flood periods vs boring flat water.

---

## 🧠 Model 3: Attention-Boosted GCN + LSTM + CBAM (`AbGGCN.py`)

> Saved to `GCN_L-SAM_CBAM/`

This variant keeps the same general data pipeline as `GGCN.py`, but adds several architectural upgrades aimed at improving spatial focus and temporal reasoning.

### Key upgrades

#### 1. CBAM-like spatial + channel attention (`StandardCBAM`)

`StandardCBAM` implements a masked Convolutional Block Attention Module:

**Channel attention:**
- Compute masked average-pooled and masked max-pooled descriptors across H,W.  
- Pass them through a shared MLP.  
- Broadcast back to each pixel to reweight channels.

**Spatial attention:**
- A conv over channel-compressed features to get a spatial importance map.  
- Multiply by the valid-water mask so land/invalid areas are not amplified.

This focuses the model on hydrologically relevant regions (active channels, floodplains).

#### 2. Temporal attention (`CustomAttentionLayer`)

Learns a weighted sum over timesteps, but explicitly **over-emphasizes the most informative 10% of timesteps**.

Implementation:
- Compute softmax attention over T.  
- Identify top-k timesteps (`k = max(1, 10% of T)`).  
- Boost those weights by an `emphasis_factor` (default 1.5).  
- Collapse time with this boosted weighting.

This biases the model to care more about flood peaks / rapid changes instead of quiet periods.

#### 3. Sequence modeling (BiLSTM / Bidirectional LSTM)

Where `GGCN.py` leans on MultiHeadAttention for temporal fusion, `AbGGCN.py` includes recurrent sequence modeling layers (Bidirectional LSTM) to capture directional dynamics and hysteresis (rising vs receding limb of hydrograph).

#### 4. Mixed precision for speed

`AbGGCN.py` explicitly switches global policy to `mixed_float16` for faster multi-GPU training and better memory usage.

---

## ⚙️ Training Details

### Hardware / distribution

All scripts:
- Detect available GPUs  
- Enable memory growth  
- Wrap model build and compile in `tf.distribute.MirroredStrategy()` for multi-GPU data parallelism.

### Reproducibility

All scripts set the same seed for:

```python
import random, numpy as np, tensorflow as tf
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
```

This gives run-to-run consistency (as much as possible under multi-GPU).

### Optimization

- Loss: masked MSE (and optionally masked MAE for reporting).  
- Learning rate scheduling: `ReduceLROnPlateau`.  
- Checkpointing: best val loss via `ModelCheckpoint`.  
- Optional hyperparameter tuning with `keras_tuner`.

---

## 🔁 Inference / Deployment Workflow

1. **Prepare inputs for a new forecast time `t*`**

   You need the last `seq_length` (e.g. 6) timesteps of:
   - `atm_pressure`, `wind_speed`, `precipitation`, `river_discharge` rasters  
   - DEM (use `dem_idw` / `dem_idw2` tiling logic)  
   - station CSV values at those timesteps  

   Apply the **exact same padding and normalization** using the saved `norm_stats.npz` from training.

2. **Form the model input batch**

   - Shape `(1, seq_length, H_pad, W_pad, F_total)` for UNet / GCN inputs.  
   - Also build `mask_input` (UNet) or the graph adjacency / valid mask (GCN variants).

3. **Run `model.predict(...)`**

   Output is a normalized raster in `[0.1,1.0]`.

4. **Denormalize back to physical water level**

   Use `wl_min`, `wl_max` from `norm_stats.npz`:

   ```python
   water_level_meters = wl_min + ( (pred - 0.1) / 0.9 ) * (wl_max - wl_min)
   ```

   Clip to valid region using the validity mask.

5. **(Optional) Unpad**

   Remove the reflect padding using `top_pad`, `left_pad`, etc., and write a GeoTIFF with the original `transform` and `crs`.

---

## 📊 Visualization & Diagnostics

- `GGCN.py` writes visualizations under `ASTGCN_OUTPUTS/visualization/`.

You can plot:
- Ground truth vs prediction maps  
- Error maps masked to water pixels  
- Time series at gauge locations by sampling predicted rasters at station coordinates.

- `GradientNormCallback` in the GCN models prints gradient norms each epoch.  
  Spikes can indicate instability or exploding gradients in backprop through the sparse graph or attention blocks.

---

## 🧪 Tips / Gotchas

### CRS Consistency
All rasters and station x,y coordinates must share the same CRS.  
The code warns if mismatches are found but still assumes the `atm_pressure` CRS/transform is the reference.  
You should make sure your data are truly aligned.

### Station coverage is sparse
If there are very few gauges, the model can overfit those cells.  
The gated GCN and CBAM attention aim to generalize away from just “copying” the station values at their exact pixels.

### Padding matters
You must reuse the stored `top_pad`, `left_pad`, etc. for inference.  
If you change padding rules at inference, spatial alignment between predicted map and georeferencing will break.

### Masked loss = no penalty over land
This is great for floodplain focus, but also means the model might hallucinate nonsense over permanently dry terrain.  
Always multiply predictions by the valid-water mask (or threshold DEM / channel network) when interpreting.

### Seq length
All scripts assume `seq_length = 6`. Changing this requires:
- Regenerating sequences  
- Rebuilding models (input shapes)  
- Re-training

### Mixed precision
The AbGGCN model uses `mixed_float16`.  
If you run on older GPUs without good half-precision support, disable that, or you may see NaNs.

---

## 📝 Repro Checklist

**Before training:**
- [ ] All `.tif` stacks present and aligned in time.  
- [ ] `training_water_level/` CSVs exist and have `x,y,water_level`.  
- [ ] `training_water_level_map/` rasters exist for the same timeline.  
- [ ] `seq_length` in code matches what you expect (default 6).  
- [ ] You can run on ≥1 GPU (for reasonable speed).  
- [ ] You’re using the same `norm_stats.npz` for inference as for training that model.

**After training:**
- [ ] Best checkpoints saved under the model’s output directory.  
- [ ] `norm_stats.npz` saved.  
- [ ] Validation loss/plots look reasonable (no NaN loss).

---

## 📚 Reference
Daramola, S., Muñoz Pauta, D.F., Sakib S.M., Thurman H., Allen G. 2025. A Transferable Deep Learning Framework to Propagate Extreme Water Levels from Sparse Tide-Gauges across Spatial Domains. Expert Systems with Applications. https://doi.org/10.1016/j.eswa.2025.130222 

---
