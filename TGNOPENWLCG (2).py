import os
import re
import random
import logging
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Reshape, Conv2D, LayerNormalization,
    Add, Activation, Lambda, Dropout, MultiHeadAttention, Multiply
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, Callback
)
import keras_tuner as kt
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split

# Create main output directory
os.makedirs('ASTGCN_OUTPUTS', exist_ok=True)
os.makedirs(os.path.join('ASTGCN_OUTPUTS', 'Models'), exist_ok=True)
os.makedirs(os.path.join('ASTGCN_OUTPUTS', 'visualization'), exist_ok=True)

# Configure GPU, Logging & Mixed Precision
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
tf.keras.backend.clear_session()
from tensorflow.keras import mixed_precision
tf.keras.mixed_precision.set_global_policy('float32')

# Set up multi-GPU strategy
strategy = tf.distribute.MirroredStrategy()

# Seeds for Reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Data Directories
base_dir = os.getcwd()
dirs = {
    'atm_pressure':    os.path.join(base_dir, 'atm_pressure'),
    'wind_speed':      os.path.join(base_dir, 'wind_speed'),
    'precipitation':   os.path.join(base_dir, 'precipitation'),
    'river_discharge': os.path.join(base_dir, 'river_discharge'),
    'dem':             os.path.join(base_dir, 'DEM'),
    'stations':        os.path.join(base_dir, 'training_water_level'),
    'water_levels':    os.path.join(base_dir, 'training_water_level_map'),
}

# Hyperparameters
seq_length = 6

# I/O & Preprocessing Utilities
def natural_sort(file_list):
    def alphanum_key(key):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('(\d+)', key)]
    return sorted(file_list, key=alphanum_key)

def load_tiff_images(data_dir, feature_name=""):
    images, filenames = [], []
    crs, transform = None, None
    file_list = natural_sort([f for f in os.listdir(data_dir) if f.endswith(".tif")])
    if not file_list:
        raise ValueError(f"No TIFF files found in {data_dir}")
    for fn in file_list:
        fp = os.path.join(data_dir, fn)
        try:
            with rasterio.open(fp) as src:
                img = src.read(1)
                if img.size == 0:
                    raise ValueError(f"Empty image in {fp}")
                crs = crs or src.crs
                transform = transform or src.transform
                images.append(img)
                filenames.append(fn)
        except Exception as e:
            logging.error(f"Failed to load {fp}: {e}")
    arr = np.array(images)
    if np.all(np.isnan(arr)):
        raise ValueError(f"All {feature_name} data is NaN!")
    print(f"Loaded {len(arr)} {feature_name} TIFFs: min={np.nanmin(arr)}, max={np.nanmax(arr)}, shape={arr.shape}")
    return arr, filenames, crs, transform

def load_single_tiff_image(filepath, feature_name=""):
    with rasterio.open(filepath) as src:
        img = src.read(1)
        if img.size == 0:
            raise ValueError(f"Empty image in {filepath}")
        crs, transform = src.crs, src.transform
    print(f"Loaded {feature_name}: min={np.nanmin(img)}, max={np.nanmax(img)}, shape={img.shape}")
    return img, crs, transform

def load_station_data(station_dir):
    station_files = natural_sort([f for f in os.listdir(station_dir) if f.endswith('.csv')])
    if not station_files:
        raise ValueError(f"No CSV files found in {station_dir}")
    station_data, station_coords = {}, {}
    for fn in station_files:
        df = pd.read_csv(os.path.join(station_dir, fn))
        if {'x','y','water_level'}.issubset(df.columns):
            name = os.path.splitext(fn)[0]
            station_data[name] = df['water_level'].values
            station_coords[name] = (df['x'].iloc[0], df['y'].iloc[0])
    df = pd.DataFrame(station_data)
    print(f"Station data: shape={df.shape}, min={df.min().min()}, max={df.max().max()}")
    return df, station_coords

def load_water_level_maps(wl_dir):
    files = natural_sort([f for f in os.listdir(wl_dir) if f.endswith('.tif')])
    if not files:
        raise ValueError(f"No TIFFs in {wl_dir}")
    maps = [rasterio.open(os.path.join(wl_dir, fn)).read(1) for fn in files]
    arr = np.array(maps)
    print(f"Water level maps: shape={arr.shape}, min={np.nanmin(arr)}, max={np.nanmax(arr)}")
    return arr, files

def create_sequences(spatial_data, station_data, target_data, mask_data, seq_length):
    Xs, Xt, Ys, Ym = [], [], [], []
    for i in range(len(spatial_data) - seq_length):
        Xs.append(spatial_data[i:i+seq_length])
        Xt.append(station_data[i:i+seq_length])
        Ys.append(target_data[i+seq_length])
        Ym.append(mask_data[i+seq_length])
    return (np.array(Xs), np.array(Xt), np.array(Ys), np.array(Ym))

def map_stations_to_grid(station_coords, transform, grid_shape):
    indices = {}
    H, W = grid_shape
    for name, (x,y) in station_coords.items():
        i, j = rowcol(transform, x, y)
        indices[name] = (np.clip(int(round(i)),0,H-1),
                         np.clip(int(round(j)),0,W-1))
    return indices

def embed_all_stations_into_single_grid(station_data, station_indices, grid_shape, top_pad, left_pad):
    S, T, K = station_data.shape
    H, W = grid_shape
    emb = np.zeros((S, T, H, W), dtype=np.float32)
    locs = list(station_indices.values())
    for s in range(S):
        for t in range(T):
            for k,(i,j) in enumerate(locs):
                pi, pj = i+top_pad, j+left_pad
                if 0 <= pi < H and 0 <= pj < W:
                    emb[s,t,pi,pj] = station_data[s,t,k]
    return emb

def create_masked_adjacency_matrix(height, width, mask):
    N = height * width
    idxs, vals = [], []
    mask_flat = (~mask[0]).flatten()
    for i in range(height):
        for j in range(width):
            idx = i * width + j
            if mask_flat[idx]:
                neigh = [(i-1,j), (i+1,j), (i,j-1), (i,j+1),
                         (i-1,j-1), (i-1,j+1), (i+1,j-1), (i+1,j+1)]
                for ni, nj in neigh:
                    if 0 <= ni < height and 0 <= nj < width:
                        jdx = ni * width + nj
                        if mask_flat[jdx]:
                            idxs += [[idx, jdx], [jdx, idx]]
                            vals += [1.0, 1.0]
    return tf.sparse.SparseTensor(
        indices=np.array(idxs, dtype=np.int64),
        values=np.array(vals, dtype=np.float32),
        dense_shape=[N, N]
    )

def normalize_sparse_adj(adj: tf.sparse.SparseTensor) -> tf.sparse.SparseTensor:
    N = adj.dense_shape[0]
    I = tf.sparse.eye(N, dtype=adj.dtype)
    A_hat = tf.sparse.add(adj, I)
    deg = tf.sparse.reduce_sum(A_hat, axis=1)
    deg_inv_sqrt = tf.pow(deg + 1e-12, -0.5)
    i, j = tf.unstack(A_hat.indices, axis=1)
    vals = A_hat.values * tf.gather(deg_inv_sqrt, i) * tf.gather(deg_inv_sqrt, j)
    return tf.sparse.SparseTensor(indices=A_hat.indices,
                                  values=vals,
                                  dense_shape=A_hat.dense_shape)

# Helper Functions
@register_keras_serializable(package='Custom', name='slice_static')
def slice_static(x):
    return x[..., :5]

@register_keras_serializable(package='Custom', name='slice_water')
def slice_water(x):
    return x[..., 5:]

@register_keras_serializable(package='Custom', name='last_timestep')
def last_timestep(x):
    return x[:, -1, ...]

@register_keras_serializable(package='Custom', name='rescale_sigmoid')
def rescale_sigmoid(x):
    return 0.1 + 0.9 * x

@register_keras_serializable(package='Custom', name='squeeze_last_dim')
def squeeze_last_dim(x):
    return tf.squeeze(x, -1)

# Custom Layer for Printing Shape
@register_keras_serializable(package='Custom', name='PrintShapeLayer')
class PrintShapeLayer(Layer):
    def call(self, inputs):
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape
        
@register_keras_serializable(package='Custom', name='ReshapeForTransformer')
class ReshapeForTransformer(Layer):
    def call(self, inputs):
        B = tf.shape(inputs)[0]
        T = tf.shape(inputs)[1]
        N_nodes = tf.shape(inputs)[2]
        F = tf.shape(inputs)[3]
        return tf.reshape(inputs, [B * N_nodes, T, F])

@register_keras_serializable(package='Custom', name='ReshapePooled')
class ReshapePooled(Layer):
    def call(self, inputs):
        pooled, original_gcn_sum = inputs
        B = tf.shape(original_gcn_sum)[0]
        N_nodes = tf.shape(original_gcn_sum)[2]
        F = tf.shape(pooled)[1]
        return tf.reshape(pooled, [B, N_nodes, F])

@register_keras_serializable(package='Custom', name='ReshapeBackToSpatial')
class ReshapeBackToSpatial(Layer):
    def call(self, inputs):
        normed, static_seq = inputs
        B = tf.shape(normed)[0]
        height = tf.shape(static_seq)[2]
        width = tf.shape(static_seq)[3]
        F = tf.shape(normed)[2]
        return tf.reshape(normed, [B, height, width, F])

# Load & Normalize Spatial Features
spatial_data = {}
for feat in ['atm_pressure','wind_speed','precipitation','river_discharge']:
    arr, _, crs, tfm = load_tiff_images(dirs[feat], feat)
    spatial_data[feat] = (arr, crs, tfm)
    
T = spatial_data['atm_pressure'][0].shape[0]

dem1_fp = os.path.join(dirs['dem'], 'dem_idw.tif')
dem2_fp = os.path.join(dirs['dem'], 'dem_idw2.tif')
dem1, crs1, tfm1 = load_single_tiff_image(dem1_fp, 'DEM1')
dem2, crs2, tfm2 = load_single_tiff_image(dem2_fp, 'DEM2')
n1 = 217
n2 = T - n1
dem_stack = np.concatenate([np.tile(dem1,(n1,1,1)), np.tile(dem2,(n2,1,1))], axis=0)
spatial_data['dem'] = (dem_stack, crs1, tfm1)

ref_crs, ref_tfm = spatial_data['atm_pressure'][1:]
for name,(arr, crs, tfm) in spatial_data.items():
    if crs!=ref_crs or tfm!=ref_tfm:
        logging.warning(f"CRS/TFM mismatch in {name}; using atm_pressure ref")

spatial_feats = np.stack([spatial_data[f][0] for f in ['atm_pressure','wind_speed','precipitation','river_discharge','dem']], axis=-1)
T, H, W, C = spatial_feats.shape
pad_h, pad_w = (4 - H%4)%4, (4 - W%4)%4
top_pad, left_pad = pad_h//2, pad_w//2
spatial_padded = np.pad(spatial_feats, ((0,0),(top_pad,pad_h-top_pad),(left_pad,pad_w-left_pad),(0,0)), mode='reflect')

wl_maps, wl_files = load_water_level_maps(dirs['water_levels'])
wl_mn, wl_mx = np.nanmin(wl_maps), np.nanmax(wl_maps)
wl_norm = np.nan_to_num(0.1 + 0.9*(wl_maps - wl_mn)/(wl_mx - wl_mn), nan=0.0)
wl_padded = np.pad(wl_norm, ((0,0),(top_pad,pad_h-top_pad),(left_pad,pad_w-left_pad)), mode='reflect')
mask_nan = np.pad(np.isnan(wl_maps), ((0,0),(top_pad,pad_h-top_pad),(left_pad,pad_w-left_pad)), constant_values=True)

# Masked Spatial Normalization (from Version 3)
mask_padded = mask_nan
spatial_norm = np.zeros_like(spatial_padded, dtype=np.float32)
for ch in range(C):
    feat = spatial_padded[..., ch].copy()
    feat[mask_padded] = np.nan
    valid = ~np.isnan(feat)
    mn, mx = feat[valid].min(), feat[valid].max()
    normed = 0.1 + 0.9 * (feat - mn) / (mx - mn)
    spatial_norm[..., ch] = np.nan_to_num(normed, nan=0.0)
print("Spatial normalized (masked):", spatial_norm.shape, np.nanmin(spatial_norm), np.nanmax(spatial_norm))

station_df, station_coords = load_station_data(dirs['stations'])
valid = station_df.dropna().values.flatten()
mn, mx = valid.min(), valid.max()
station_norm_df = 0.1 + 0.9*(station_df - mn)/(mx - mn) if mx>mn else pd.DataFrame(0.1, index=station_df.index, columns=station_df.columns)
station_norm = station_norm_df.fillna(0.0).values

spatial_mins = []
spatial_maxs = []
for ch in range(C):
    feat = spatial_padded[...,ch].copy()
    feat[mask_padded] = np.nan
    valid = ~np.isnan(feat)
    mn, mx = feat[valid].min(), feat[valid].max()
    spatial_mins.append(mn)
    spatial_maxs.append(mx)
spatial_mins = np.array(spatial_mins)
spatial_maxs = np.array(spatial_maxs)

valid = station_df.dropna().values.flatten()
station_min, station_max = valid.min(), valid.max()

wl_min, wl_max = wl_mn, wl_mx

np.savez(
    os.path.join('ASTGCN_OUTPUTS', 'norm_stats.npz'),
    spatial_mins=spatial_mins,
    spatial_maxs=spatial_maxs,
    station_min=station_min,
    station_max=station_max,
    wl_min=wl_min,
    wl_max=wl_max,
    pad_h=pad_h,
    pad_w=pad_w,
    top_pad=top_pad,
    left_pad=left_pad,
    seq_length=seq_length
)
print("Normalization statistics saved to ASTGCN_OUTPUTS/norm_stats.npz")

Xs, Xt, y_all, m_all = create_sequences(spatial_norm, station_norm, wl_padded, mask_nan, seq_length)

Xs_tr, Xs_val, Xt_tr, Xt_val, y_tr, y_val, m_tr, m_val = train_test_split(
    Xs, Xt, y_all, m_all,
    test_size=0.2, shuffle=False, random_state=seed_value
)
print("Train shapes:", Xs_tr.shape, Xt_tr.shape, y_tr.shape)
print("Validation shapes:", Xs_val.shape, Xt_val.shape, y_val.shape)

station_indices = map_stations_to_grid(station_coords, ref_tfm, (H, W))
print(f"Number of stations: {len(station_indices)}")
suggested_layers = max(20, int(np.sqrt(H**2 + W**2) / len(station_indices)))
print(f"Suggested GCN layers: {suggested_layers}")

embedded_tr = embed_all_stations_into_single_grid(Xt_tr, station_indices, (H + pad_h, W + pad_w), top_pad, left_pad)[..., None]
embedded_val = embed_all_stations_into_single_grid(Xt_val, station_indices, (H + pad_h, W + pad_w), top_pad, left_pad)[..., None]
X_tr = np.concatenate([Xs_tr, embedded_tr], axis=-1)
X_val = np.concatenate([Xs_val, embedded_val], axis=-1)
print("Combined input shape:", X_tr.shape)

y_tr_masked = np.stack([y_tr, (~m_tr).astype(np.float32)], axis=-1)
y_val_masked = np.stack([y_val, (~m_val).astype(np.float32)], axis=-1)

sample_weights = np.clip(
    np.var(y_tr.reshape(len(Xs_tr), -1), axis=1) / np.mean(np.var(y_tr.reshape(len(Xs_tr), -1), axis=1)),
    0.1, 10.0
).astype(np.float32)

height_padded, width_padded = spatial_norm.shape[1:3]
raw_adj = create_masked_adjacency_matrix(height_padded, width_padded, mask_nan)
adj_matrix = normalize_sparse_adj(raw_adj)
print("Masked adjacency matrix shape:", adj_matrix.dense_shape)

N = int(adj_matrix.dense_shape[0].numpy())

# Valid mask for graph nodes (from Version 3)
valid_mask = (~mask_nan[0]).astype(np.float32)
valid_mask_tr = np.tile(valid_mask[None, :, :], (len(Xs_tr), 1, 1))
valid_mask_val = np.tile(valid_mask[None, :, :], (len(Xs_val), 1, 1))

def masked_mse(y_true, y_pred):
    y_pred = tf.cast(y_pred, tf.float32)
    y_val = y_true[..., 0]
    mask = y_true[..., 1]
    se = tf.square(y_pred - y_val)
    num = tf.reduce_sum(se * mask, axis=[1, 2])
    den = tf.reduce_sum(mask, axis=[1, 2]) + 1e-7
    return num / den

@register_keras_serializable(package='Custom', name='GatedGraphConvolution')
class GatedGraphConvolution(tf.keras.layers.Layer):
    def __init__(self, filters, static_dim, dropout_rate=0.3, l2_strength=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.static_dim = static_dim
        self.dropout_rate = dropout_rate
        self.l2_strength = l2_strength
        self.min_gate = tf.Variable(0.1, trainable=True, dtype=self.dtype, name='min_gate')
        self.feature_dense = Dense(filters, kernel_regularizer=regularizers.l2(self.l2_strength))
        self.gate_dense_1 = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(self.l2_strength))
        self.gate_dense_2 = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(self.l2_strength))
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout_rate)

    def build(self, input_shape):
        water_shape, static_shape, _ = input_shape
        F_in = water_shape[-1]
        static_dim = static_shape[-1]
        self.feature_dense.build((None, F_in))
        self.gate_dense_1.build((None, 3))
        self.gate_dense_2.build((None, static_dim - 3))
        self.layer_norm.build((None, None, None, self.filters))
        super().build(input_shape)

    def call(self, inputs):
        water, static, adj = inputs
        water = tf.cast(water, self.compute_dtype)
        static = tf.cast(static, self.compute_dtype)
        sg1 = static[..., :3]
        sg2 = static[..., 3:]
        gate = self.gate_dense_1(sg1) * self.gate_dense_2(sg2)
        gate = tf.maximum(gate, tf.cast(self.min_gate, self.compute_dtype))
        B = tf.shape(water)[0]
        T = tf.shape(water)[1]
        N = tf.shape(water)[2]
        F_in = tf.shape(water)[3]
        x = tf.reshape(water * gate, [B * T, N, F_in])
        x_t = tf.transpose(x, perm=[2, 0, 1])

        def sp_mul(f):
            return tf.transpose(tf.sparse.sparse_dense_matmul(adj, tf.transpose(f)))

        xt = tf.map_fn(sp_mul,
                       x_t,
                       fn_output_signature=tf.TensorSpec(shape=(None, None), dtype=self.compute_dtype))
        xt = tf.reshape(tf.transpose(xt, perm=[1, 2, 0]), [B * T * N, F_in])
        out = self.feature_dense(xt)
        out = tf.reshape(out, [B, T, N, self.filters])
        out = self.layer_norm(out)
        out = self.dropout(out)
        return out

    def compute_output_shape(self, input_shape):
        water_shape, _, _ = input_shape
        batch_size, time, N, _ = water_shape
        return (batch_size, time, N, self.filters)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'static_dim': self.static_dim,
            'dropout_rate': self.dropout_rate,
            'l2_strength': self.l2_strength,
        })
        return config

class GradientNormCallback(Callback):
    def __init__(self, X_train, y_train, sample_weights, batch_size=4):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.sample_weights = sample_weights
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        X_batch = self.X_train[:self.batch_size]
        y_batch = self.y_train[:self.batch_size]
        sw_batch = self.sample_weights[:self.batch_size]
        with tf.GradientTape() as tape:
            predictions = self.model(X_batch, training=True)
            loss = self.model.compiled_loss(y_batch, predictions, sample_weight=sw_batch)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        avg_grad_norm = tf.reduce_mean([tf.norm(g) for g in gradients if g is not None])
        print(f"Epoch {epoch + 1} - Avg Gradient Norm: {avg_grad_norm.numpy()}")

@register_keras_serializable(package='Custom', name='SparseTensorLayer')
class SparseTensorLayer(Layer):
    def __init__(self, N, **kwargs):
        super(SparseTensorLayer, self).__init__(**kwargs)
        self.N = N

    def call(self, inputs):
        indices, values, dense_shape = inputs
        batch_size = tf.shape(indices)[0]
        
        def create_sparse_tensor():
            indices_first = indices[0]
            values_first = values[0]
            dense_shape_first = dense_shape[0]
            return tf.sparse.SparseTensor(indices_first, values_first, dense_shape_first)
        
        def empty_sparse_tensor():
            return tf.sparse.SparseTensor(
                indices=tf.zeros([0, 2], dtype=tf.int64),
                values=tf.zeros([0], dtype=tf.float32),
                dense_shape=[tf.constant(self.N, dtype=tf.int64), tf.constant(self.N, dtype=tf.int64)]
            )
        
        adj = tf.cond(batch_size > 0, create_sparse_tensor, empty_sparse_tensor)
        return adj

    def compute_output_shape(self, input_shape):
        return (None, None)

    def get_config(self):
        config = super(SparseTensorLayer, self).get_config()
        config.update({'N': self.N})
        return config

@register_keras_serializable(package='Custom', name='DynamicReshapeGCN')
class DynamicReshapeGCN(Layer):
    def call(self, inputs):
        x, ref_tensor = inputs
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        filters_out = tf.shape(x)[3]
        height_out = tf.shape(ref_tensor)[2]
        width_out = tf.shape(ref_tensor)[3]
        return tf.reshape(x, [B, T, height_out, width_out, filters_out])
        
@register_keras_serializable(package='Custom', name='SoftmaxLayer')
class SoftmaxLayer(Layer):
    def __init__(self, axis, **kwargs):
        super(SoftmaxLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.nn.softmax(inputs, axis=self.axis)

    def get_config(self):
        config = super(SoftmaxLayer, self).get_config()
        config.update({'axis': self.axis})
        return config
        
@register_keras_serializable(package='Custom', name='ReduceSumLayer')
class ReduceSumLayer(Layer):
    def __init__(self, axis, **kwargs):
        super(ReduceSumLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.axis)

    def get_config(self):
        config = super(ReduceSumLayer, self).get_config()
        config.update({'axis': self.axis})
        return config
        
@register_keras_serializable(package='Custom', name='flatten_mask')
def flatten_mask(m):
    # (batch, N, 1) → (batch*N,)
    return tf.reshape(m, [-1])

@register_keras_serializable(package='Custom', name='tile_time_dim')
def tile_time_dim(m):
    # (batch*N,) → (batch*N, seq_length)
    return tf.tile(m[:, None], [1, seq_length])

@register_keras_serializable(package='Custom', name='tile_attn_mask')
def tile_attn_mask(m):
    # (batch*N, seq_length) → (batch*N, seq_length, seq_length)
    return tf.tile(m[:, None, :], [1, seq_length, 1])
    
@register_keras_serializable(package='Custom', name='expand_mask_dim')
def expand_mask_dim(m):
    # (batch, H, W) → (batch, H, W, 1)
    return tf.expand_dims(m, -1)
    
@register_keras_serializable(package='Custom', name='cast_to_bool')
def cast_to_bool(x):
    return tf.cast(x, tf.bool)

def build_model(hp):
    with strategy.scope():
        data_input = Input(shape=(seq_length, None, None, 6), name='data_input')
        adj_indices = Input(shape=(None, 2), dtype=tf.int64, name='adj_indices')
        adj_values = Input(shape=(None,), dtype=tf.float32, name='adj_values')
        adj_shape = Input(shape=(2,), dtype=tf.int64, name='adj_shape')
        mask_input = Input(shape=(None, None), name='mask_input')  # Added from Version 3
        
        adj = SparseTensorLayer(N=N)([adj_indices, adj_values, adj_shape])
        node_mask = Reshape((-1, 1))(mask_input)  # Added from Version 3

        static_seq = Lambda(slice_static)(data_input)
        water_seq  = Lambda(slice_water)(data_input)
        static_nodes = Reshape((seq_length, -1, 5))(static_seq)
        water_nodes  = Reshape((seq_length, -1, 1))(water_seq)
        static_last = Lambda(last_timestep)(static_seq)

        filters       = hp.Int('filters', 32, 64, step=16)
        dropout_rate  = hp.Float('dropout_rate', 0.0, 0.3, step=0.1)
        num_gcn_layers= 3
        l2_strength   = hp.Choice('l2_strength', [1e-8, 1e-7, 1e-6])

        x = water_nodes
        outputs = []
        for i in range(num_gcn_layers):
            ggc = GatedGraphConvolution(filters, static_dim=5, dropout_rate=dropout_rate, l2_strength=l2_strength, name=f'ggc_{i}')
            x_new = ggc([x, static_nodes, adj])
            if i > 0:
                if x.shape[-1] != x_new.shape[-1]:
                    x = Dense(filters, use_bias=False, kernel_regularizer=regularizers.l2(l2_strength))(x)
                x_new = Add()([x_new, x])
            x = x_new
            outputs.append(x)
        gcn_sum = Add(name='sum_of_gcn_outputs')(outputs)
        gcn_last = gcn_sum[:, -1, :, :]

        gcn_for_transformer = ReshapeForTransformer()(gcn_sum)

        def transformer_encoder(inputs, mask, head_size=32, num_heads=4, ff_dim=64, dropout=0):
            # inputs: (batch_nodes, T, F)
            # mask:   boolean tensor of shape (batch_nodes, T) where True means “keep”
            x = MultiHeadAttention(
                key_dim=head_size,
                num_heads=num_heads,
                dropout=dropout
            )(
                query=inputs,
                value=inputs,
                attention_mask=mask
            )
            x = Dropout(dropout)(x)
            res = x + inputs
            x = LayerNormalization()(res)
            x = Dense(ff_dim, activation="relu")(x)
            x = Dropout(dropout)(x)
            x = Dense(inputs.shape[-1])(x)
            x = LayerNormalization()(x + res)
            return x

        # mask out invalid nodes before the temporal transformer:
        # mask out invalid nodes before the temporal transformer:
        flat_mask        = Lambda(flatten_mask,    name='flat_node_mask')(node_mask)
        key_mask         = Lambda(tile_time_dim,   name='key_mask')(flat_mask)
        attn_mask_float  = Lambda(tile_attn_mask,  name='attn_mask')(key_mask)
        # NEW: cast to bool so MultiHeadAttention treats it as keep/drop
        attn_mask_bool   = Lambda(cast_to_bool, name='attn_mask_bool')(attn_mask_float)
        
        x = gcn_for_transformer
        for i in range(2):
            x = transformer_encoder(x, attn_mask_bool,
                                    head_size=32,
                                    num_heads=4,
                                    ff_dim=64,
                                    dropout=dropout_rate)

        att_scores = Dense(1, use_bias=False)(x)
        att_weights = SoftmaxLayer(axis=1)(att_scores)
        pooled = ReduceSumLayer(axis=1)(x * att_weights)  # Fixed line
        
        pooled = ReshapePooled()([pooled, gcn_sum])
        pooled = Multiply(name='mask_pooled')([pooled, node_mask])  # Added from Version 3
        gcn_last = Multiply(name='mask_gcn_last')([gcn_last, node_mask])  # Added from Version 3

        fusion = Add(name='gcn_transformer_add')([gcn_last, pooled])
        normed = LayerNormalization(name='norm_after_fusion')(fusion)

        prop_map = ReshapeBackToSpatial()([normed, static_seq])

        # — final 3×3 convs —
        prop_map = Conv2D(1, 3, padding='same', activation='linear', kernel_regularizer=regularizers.l2(l2_strength), name='prop_map_conv')(prop_map)
        static_feat = Conv2D(16, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(l2_strength), name='static_map_conv1')(static_last)
        static_map = Conv2D(1, 3, padding='same', activation='linear', kernel_regularizer=regularizers.l2(l2_strength), name='static_map_conv2')(static_feat)
        
        # — expand mask to (batch, H, W, 1) and zero out invalid pixels —
        mask_exp  = Lambda(expand_mask_dim,  name='mask_expand')(mask_input)
        prop_map   = Multiply(name='mask_prop_map')([prop_map, mask_exp])
        static_map = Multiply(name='mask_static_map')([static_map, mask_exp])
        
        # — fuse safely —
        fused        = Add(name='final_fusion_add')([prop_map, static_map])
        sig          = Activation('sigmoid', name='sigmoid_activation')(fused)
        rescaled     = Lambda(rescale_sigmoid, name='rescale_sigmoid')(sig)
        
        # zero out any invalid pixels in the final map
        masked_final = Multiply(name='mask_final')([rescaled, mask_exp])
        out          = Lambda(squeeze_last_dim, name='squeeze_last_dim')(masked_final)
        
        model = Model(inputs=[data_input, adj_indices, adj_values, adj_shape, mask_input], outputs=out)

        opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=opt, loss=masked_mse, metrics=[masked_mse], weighted_metrics=[masked_mse])
        return model

# Dictionary of everything needed to reconstruct your model
custom_objects = {
    # Serializable functions used in Lambda layers
    'slice_static': slice_static,
    'slice_water': slice_water,
    'last_timestep': last_timestep,
    'rescale_sigmoid': rescale_sigmoid,
    'squeeze_last_dim': squeeze_last_dim,

    # Custom layers
    'PrintShapeLayer': PrintShapeLayer,
    'ReshapeForTransformer': ReshapeForTransformer,
    'ReshapePooled': ReshapePooled,
    'ReshapeBackToSpatial': ReshapeBackToSpatial,
    'GatedGraphConvolution': GatedGraphConvolution,
    'SparseTensorLayer': SparseTensorLayer,
    'DynamicReshapeGCN': DynamicReshapeGCN,
    'SoftmaxLayer': SoftmaxLayer,
    'ReduceSumLayer': ReduceSumLayer,
    'flatten_mask':   flatten_mask,
    'tile_time_dim':  tile_time_dim,
    'tile_attn_mask': tile_attn_mask,
    'expand_mask_dim': expand_mask_dim,
    'cast_to_bool': cast_to_bool,

    # Custom loss
    'masked_mse': masked_mse
}

def visualize_trial(trial_id):
    model_fp = os.path.join('ASTGCN_OUTPUTS', 'Models', f'model_{trial_id}.keras')
    if not os.path.exists(model_fp):
        return

    # Load the trained model for this trial
    model = load_model(model_fp, custom_objects=custom_objects)

    # Create visualization folder for this trial
    vis_dir = os.path.join('ASTGCN_OUTPUTS', 'visualization', f'model_{trial_id}')
    os.makedirs(vis_dir, exist_ok=True)

    # Run inference on validation data
    y_pred = model.predict(
        [X_val, adj_val_indices_stacked, adj_val_values_stacked, adj_val_shape_stacked, valid_mask_val],
        batch_size=4
    )

    # Denormalize, mask, and crop back to original grid
    y_pred_orig = (y_pred - 0.1) / 0.9 * (wl_mx - wl_mn) + wl_mn
    y_val_orig  = (y_val   - 0.1) / 0.9 * (wl_mx - wl_mn) + wl_mn
    y_pred_mask = np.where(m_val, np.nan, y_pred_orig)
    y_val_mask  = np.where(m_val, np.nan, y_val_orig)
    y_pred_crop = y_pred_mask[:, top_pad:top_pad+H, left_pad:left_pad+W]
    y_val_crop  = y_val_mask[:,  top_pad:top_pad+H, left_pad:left_pad+W]

    # Save sample actual vs. predicted vs. difference maps
    for i in range(min(2, y_pred_crop.shape[0])):
        fig, axs = plt.subplots(1, 3, figsize=(15,5))
        axs[0].imshow(y_val_crop[i],   cmap='viridis'); axs[0].set_title('Actual')
        axs[1].imshow(y_pred_crop[i],  cmap='viridis'); axs[1].set_title('Predicted')
        diff = y_pred_crop[i] - y_val_crop[i]
        vmax = np.nanmax(np.abs(diff))
        axs[2].imshow(diff,            cmap='bwr', vmin=-vmax, vmax=vmax); axs[2].set_title('Difference')
        for ax in axs:
            fig.colorbar(ax.images[0], ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(vis_dir, f'prediction_sample_{i}.png'))
        plt.close(fig)

    # Station‐wise time series comparison: single figure with 3 columns
    val_start = int(0.8 * len(Xs))
    times = np.arange(
        val_start + seq_length,
        val_start + seq_length + y_pred_crop.shape[0]
    )
    
    num_stations = len(station_indices)
    cols = 3
    rows = int(np.ceil(num_stations / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), sharex=True)
    axes = axes.flatten()
    
    for idx, (st, (i, j)) in enumerate(station_indices.items()):
        pi, pj = i + top_pad, j + left_pad
        ts_csv  = station_df.iloc[times, idx].values
        ts_rast  = wl_maps[times, i, j]
        ts_pred = y_pred_orig[:, pi, pj]
    
        ax = axes[idx]
        ax.plot(times,    ts_csv,  label='CSV')
        ax.plot(times,    ts_rast, label='Raster')
        ax.plot(times,    ts_pred, label='Predicted')
        ax.set_title(f'Station {st}')
        ax.set_ylabel('Water Level')
        if idx == 0:
            ax.legend(loc='upper right')
    
    # Remove any extra empty subplots
    for ax in axes[num_stations:]:
        fig.delaxes(ax)
    
    fig.tight_layout()
    fig.savefig(os.path.join(vis_dir, 'all_stations_timeseries.png'))
    plt.close(fig)

    # Initial embedding visualization
    plt.figure()
    plt.imshow(embedded_val[0,0,:,:,0], cmap='viridis')
    plt.title('Initial Water Level Embedding (First Timestep)')
    plt.colorbar()
    plt.savefig(os.path.join(vis_dir, 'initial_embedding.png'))
    plt.close()

    # Save validation-loss history and plot loss curve
    trial = tuner.oracle.get_trial(str(trial_id))

    # try val_masked_mse first, fallback to val_loss
    for metric_name in ('val_masked_mse', 'val_loss'):
        try:
            history = trial.metrics.get_history(metric_name)
            hist_df = pd.DataFrame(history)
            print(f"Plotting {metric_name} for trial {trial_id}")
            break
        except ValueError:
            continue
    else:
        print(f"No validation history for trial {trial_id}; skipping plot.")
        return

    # now hist_df has columns 'epoch' and 'value'
    hist_df.to_csv(os.path.join(vis_dir, 'loss_history.csv'), index=False)

    # plot it
    plt.figure()
    plt.plot(hist_df['epoch'], hist_df['value'])
    plt.title(f'{metric_name} per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.savefig(os.path.join(vis_dir, 'loss_curve.png'))
    plt.close()

class CustomModelCheckpoint(ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        if tf.distribute.get_replica_context().replica_id_in_sync_group == 0:
            super().on_epoch_end(epoch, logs)

class MyTuner(kt.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        # ––– Create brand-new callbacks for THIS trial –––
        early_stp = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            mode='min',
            factor=0.2,
            patience=10,
            threshold=1e-6,
            threshold_mode='rel',
            cooldown=0,
            min_lr=1e-8,
            verbose=1
        )
        ckpt = CustomModelCheckpoint(
            filepath=os.path.join(
                'ASTGCN_OUTPUTS', 'Models',
                f'model_{trial.trial_id}.keras'
            ),
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        )

        # ––– Overwrite any existing callbacks with exactly these three –––
        kwargs['callbacks'] = [early_stp, reduce_lr, ckpt]

        # ––– Run the trial with its own callbacks –––
        results = super().run_trial(trial, *args, **kwargs)

        # ––– Visualize & save artifacts for this trial only –––
        visualize_trial(trial.trial_id)

        return results

tuner = MyTuner(
     hypermodel=build_model,
    objective='val_loss',
    max_trials=50,
    directory=os.path.join('ASTGCN_OUTPUTS', 'astgcn_tuning_optimized'),
    project_name='water_level_optimized',
    overwrite=False
)

adj_train_indices = adj_matrix.indices.numpy()
adj_train_values = adj_matrix.values.numpy()
adj_train_shape = adj_matrix.dense_shape.numpy()

np.save(os.path.join('ASTGCN_OUTPUTS', 'adj_indices.npy'), adj_train_indices)
np.save(os.path.join('ASTGCN_OUTPUTS', 'adj_values.npy'), adj_train_values)
np.save(os.path.join('ASTGCN_OUTPUTS', 'adj_shape.npy'), adj_train_shape)
print("Adjacency matrix components saved to ASTGCN_OUTPUTS/")

num_train_samples = len(Xs_tr)
num_val_samples = len(Xs_val)
adj_train_indices_stacked = np.tile(adj_train_indices[None, :, :], (num_train_samples, 1, 1))
adj_train_values_stacked = np.tile(adj_train_values[None, :], (num_train_samples, 1))
adj_train_shape_stacked = np.tile(adj_train_shape[None, :], (num_train_samples, 1))
adj_val_indices_stacked = np.tile(adj_train_indices[None, :, :], (num_val_samples, 1, 1))
adj_val_values_stacked = np.tile(adj_train_values[None, :], (num_val_samples, 1))
adj_val_shape_stacked = np.tile(adj_train_shape[None, :], (num_val_samples, 1))

# --- build train / val datasets with shuffle, repeat, and prefetch ---
batch_size = 4

train_ds = (
    tf.data.Dataset
      .from_tensor_slices((
          (X_tr,
           adj_train_indices_stacked,
           adj_train_values_stacked,
           adj_train_shape_stacked,
           valid_mask_tr),
          y_tr_masked,
          sample_weights
      ))
      .shuffle(buffer_size=len(X_tr), seed=seed_value)
      .repeat()                              # infinite repeat
      .batch(batch_size, drop_remainder=True)
      .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset
      .from_tensor_slices((
          (X_val,
           adj_val_indices_stacked,
           adj_val_values_stacked,
           adj_val_shape_stacked,
           valid_mask_val),
          y_val_masked
      ))
      .repeat()                              # infinite repeat
      .batch(batch_size, drop_remainder=True)
      .prefetch(tf.data.AUTOTUNE)
)

# compute how many batches fit in one “epoch”
steps_per_epoch  = len(X_tr) // batch_size
validation_steps = len(X_val) // batch_size

tuner.search(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=300,
    callbacks=[],
    verbose=1
)