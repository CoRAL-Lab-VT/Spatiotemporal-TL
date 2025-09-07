import os
import re
import random
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout, ConvLSTM2D, LayerNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import regularizers
import keras_tuner as kt
import logging
import json
import shutil

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
tf.keras.backend.clear_session()

# Configure GPU (if available)
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available:", len(gpus))
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except RuntimeError as e:
        print(e)

# Set up multi-GPU strategy
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

# Create main output directory
os.makedirs('UNET_OUTPUTS', exist_ok=True)

# Set Random Seeds for Reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Data Loading and Preprocessing Functions
def natural_sort(file_list):
    def alphanum_key(key):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', key)]
    return sorted(file_list, key=alphanum_key)

def load_tiff_images(data_dir, feature_name=""):
    images, filenames = [], []
    crs, transform = None, None
    file_list = natural_sort([f for f in os.listdir(data_dir) if f.endswith(".tif")])
    if not file_list:
        raise ValueError(f"No TIFF files found in {data_dir}")
    for filename in file_list:
        filepath = os.path.join(data_dir, filename)
        try:
            with rasterio.open(filepath) as src:
                img = src.read(1)
                if img.size == 0:
                    raise ValueError(f"Empty image in {filepath}")
                if crs is None: crs = src.crs
                if transform is None: transform = src.transform
                images.append(img)
                filenames.append(filename)
        except Exception as e:
            logging.error(f"Failed to load {filepath}: {e}")
    images = np.array(images)
    if np.all(np.isnan(images)):
        raise ValueError(f"All {feature_name} data is NaN!")
    print(f"Loaded {len(images)} {feature_name} TIFFs from {data_dir}")
    print(f"{feature_name} min: {np.nanmin(images)}, max: {np.nanmax(images)}, shape: {images.shape}")
    return images, filenames, crs, transform

def load_single_tiff_image(filepath, feature_name=""):
    try:
        with rasterio.open(filepath) as src:
            img = src.read(1)
            if img.size == 0:
                raise ValueError(f"Empty image in {filepath}")
            crs = src.crs
            transform = src.transform
        print(f"Loaded {feature_name} TIFF from {filepath}")
        print(f"{feature_name} min: {np.nanmin(img)}, max: {np.nanmax(img)}, shape: {img.shape}")
        return img, crs, transform
    except Exception as e:
        raise ValueError(f"Failed to load {filepath}: {e}")

def load_station_data(station_dir):
    station_files = natural_sort([f for f in os.listdir(station_dir) if f.endswith('.csv')])
    if not station_files:
        raise ValueError(f"No CSV files found in {station_dir}")
    station_data, station_coords = {}, {}
    for file in station_files:
        filepath = os.path.join(station_dir, file)
        df = pd.read_csv(filepath)
        if {'x', 'y', 'water_level'}.issubset(df.columns):
            station_name = os.path.splitext(file)[0]
            station_data[station_name] = df['water_level'].values
            station_coords[station_name] = (df['x'].iloc[0], df['y'].iloc[0])
    df = pd.DataFrame(station_data)
    print(f"Station data shape: {df.shape}, min: {df.min().min()}, max: {df.max().max()}")
    return df, station_coords

def load_water_level_maps(water_level_dir):
    water_levels, filenames = [], natural_sort([f for f in os.listdir(water_level_dir) if f.endswith('.tif')])
    if not filenames:
        raise ValueError(f"No TIFF files found in {water_level_dir}")
    for file in filenames:
        with rasterio.open(os.path.join(water_level_dir, file)) as src:
            wl = src.read(1)
            water_levels.append(wl)
    water_levels = np.array(water_levels)
    print(f"Water level maps shape: {water_levels.shape}, min: {np.nanmin(water_levels)}, max: {np.nanmax(water_levels)}")
    return water_levels, filenames

def create_sequences(spatial_data, station_data, target_data, mask_data, seq_length):
    X_spatial, X_station, y, y_mask = [], [], [], []
    for i in range(len(spatial_data) - seq_length):
        X_spatial.append(spatial_data[i:i+seq_length])
        X_station.append(station_data[i:i+seq_length])
        y.append(target_data[i+seq_length])
        y_mask.append(mask_data[i+seq_length])
    return (np.array(X_spatial), np.array(X_station), np.array(y), np.array(y_mask))

def map_stations_to_grid(station_coords, transform, grid_shape):
    station_grid_indices = {}
    for station, (x, y) in station_coords.items():
        row, col = rowcol(transform, x, y)
        row = max(0, min(int(round(row)), grid_shape[0] - 1))
        col = max(0, min(int(round(col)), grid_shape[1] - 1))
        station_grid_indices[station] = (row, col)
    return station_grid_indices

def create_station_masks(station_grid_indices, grid_shape):
    masks = np.zeros((len(station_grid_indices), *grid_shape), dtype=np.float32)
    for idx, (station, (row, col)) in enumerate(station_grid_indices.items()):
        masks[idx, row, col] = 1.0
    return masks

def embed_stations_into_grid(station_data, station_masks):
    samples, seq_len, num_stations = station_data.shape
    height, width = station_masks.shape[1:]
    embedded = np.zeros((samples, seq_len, height, width, num_stations), dtype=np.float32)
    for s in range(num_stations):
        embedded[..., s] = station_data[:, :, s, np.newaxis, np.newaxis] * station_masks[s]
    return embedded

# Define Data Directories
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

dem_files = [os.path.join(dirs['dem'], f) for f in ['dem_idw.tif', 'dem_idw2.tif']]

# Load and Preprocess Data
print("Loading spatial features...")
spatial_data = {}
for name, path in dirs.items():
    if name in ['atm_pressure', 'wind_speed', 'precipitation', 'river_discharge']:
        images, filenames, crs, transform = load_tiff_images(path, name)
        spatial_data[name] = (images, filenames, crs, transform)
    elif name == 'dem':
        dem1_data = load_single_tiff_image(dem_files[0], "DEM_1")
        dem2_data = load_single_tiff_image(dem_files[1], "DEM_2")
        num_timesteps_total = spatial_data['atm_pressure'][0].shape[0]
        num_timesteps1 = 217
        num_timesteps2 = max(0, num_timesteps_total - num_timesteps1)
        dem_array = np.concatenate([
            np.tile(dem1_data[0], (num_timesteps1, 1, 1)),
            np.tile(dem2_data[0], (num_timesteps2, 1, 1))
        ], axis=0)
        spatial_data[name] = (dem_array, None, dem1_data[1], dem1_data[2])

# Verify CRS and align data
ref_crs, ref_transform = spatial_data['atm_pressure'][2], spatial_data['atm_pressure'][3]
for name, (data, _, crs, transform) in spatial_data.items():
    if crs != ref_crs or transform != ref_transform:
        logging.warning(f"CRS/Transform mismatch in {name}. Using reference from atm_pressure.")

spatial_features = np.stack([spatial_data[n][0] for n in ['atm_pressure', 'wind_speed', 'precipitation', 'river_discharge', 'dem']], axis=-1)
print(f"Spatial features shape: {spatial_features.shape}")

# Load water level maps early to get mask
water_level_maps, wl_filenames = load_water_level_maps(dirs['water_levels'])
wl_min, wl_max = np.nanmin(water_level_maps), np.nanmax(water_level_maps)
if wl_max - wl_min == 0:
    raise ValueError("Water level maps have no variance!")
water_level_norm = 0.1 + 0.9 * (water_level_maps - wl_min) / (wl_max - wl_min)
water_level_norm = np.nan_to_num(water_level_norm, nan=0.0)
mask_nan = np.isnan(water_level_maps)

# Pad spatial features to be divisible by 8 for U-Net
timesteps, height, width, channels = spatial_features.shape
pad_height = (8 - height % 8) if height % 8 else 0
pad_width = (8 - width % 8) if width % 8 else 0
top_pad, left_pad = pad_height // 2, pad_width // 2
spatial_padded = np.pad(spatial_features, 
                        ((0, 0), (top_pad, pad_height - top_pad), 
                         (left_pad, pad_width - left_pad), (0, 0)), 
                        mode='reflect')
print(f"Padded spatial features shape: {spatial_padded.shape}")

padded_height = spatial_padded.shape[1]  
padded_width = spatial_padded.shape[2]

mask_nan_padded = np.pad(mask_nan, 
                         ((0, 0), (top_pad, pad_height - top_pad), 
                          (left_pad, pad_width - left_pad)), 
                         constant_values=True)

# Masked spatial normalization
spatial_norm = np.zeros_like(spatial_padded, dtype=np.float32)
spatial_mins = []
spatial_maxs = []
for ch in range(channels):
    feat = spatial_padded[..., ch].copy()
    feat[mask_nan_padded] = np.nan
    valid = ~np.isnan(feat)
    mn, mx = feat[valid].min(), feat[valid].max()
    normed = 0.1 + 0.9 * (feat - mn) / (mx - mn) if mx > mn else np.full_like(feat, 0.1)
    spatial_norm[..., ch] = np.nan_to_num(normed, nan=0.0)
    spatial_mins.append(mn)
    spatial_maxs.append(mx)
spatial_mins = np.array(spatial_mins)
spatial_maxs = np.array(spatial_maxs)
print(f"Spatial normalized (masked): shape={spatial_norm.shape}, min={np.nanmin(spatial_norm)}, max={np.nanmax(spatial_norm)}")

# Load station data
station_df, station_coords = load_station_data(dirs['stations'])
if station_df.isnull().all().any():
    raise ValueError("One or more stations have all NaN values!")
valid = station_df.dropna().values.flatten()
station_min, station_max = valid.min(), valid.max()
station_norm_df = 0.1 + 0.9 * (station_df - station_min) / (station_max - station_min) if station_max > station_min else pd.DataFrame(0.1, index=station_df.index, columns=station_df.columns)
station_norm = station_norm_df.fillna(0.0).values
num_stations = station_df.shape[1]

seq_length = 6
num_spatial_features = spatial_norm.shape[-1]
ch_per_timestep = num_spatial_features + num_stations

# Padded water levels
water_level_padded = np.pad(water_level_norm, 
                            ((0, 0), (top_pad, pad_height - top_pad), 
                             (left_pad, pad_width - left_pad)), 
                            mode='reflect')
print(f"Padded water level shape: {water_level_padded.shape}")

# Save normalization statistics and model parameters
np.savez(
    os.path.join('UNET_OUTPUTS', 'norm_stats.npz'),
    spatial_mins=spatial_mins,
    spatial_maxs=spatial_maxs,
    station_min=station_min,
    station_max=station_max,
    wl_min=wl_min,
    wl_max=wl_max,
    pad_height=pad_height,
    pad_width=pad_width,
    top_pad=top_pad,
    left_pad=left_pad,
    patch_height=padded_height,
    patch_width=padded_width,
    seq_length=seq_length,
    num_stations=num_stations,
    ch_per_timestep=ch_per_timestep
)
print("Normalization statistics and model parameters saved to UNET_OUTPUTS/norm_stats.npz")

# Map stations to grid and adjust for padding
station_grid_indices = map_stations_to_grid(station_coords, ref_transform, water_level_maps.shape[1:])
adjusted_station_grid_indices = {station: (row + top_pad, col + left_pad) 
                                for station, (row, col) in station_grid_indices.items()}
station_masks = create_station_masks(adjusted_station_grid_indices, spatial_norm.shape[1:3])

# Create sequences
X_spatial, X_station, y_target, y_mask = create_sequences(spatial_norm, station_norm, water_level_padded, mask_nan_padded, seq_length)

# Split into 80% train and 20% validation
total_samples = len(X_spatial)
train_end = int(0.8 * total_samples)
X_spatial_train = X_spatial[:train_end]
X_spatial_val = X_spatial[train_end:]
X_station_train = X_station[:train_end]
X_station_val = X_station[train_end:]
y_train = y_target[:train_end]
y_val = y_target[train_end:]
y_mask_train = y_mask[:train_end]
y_mask_val = y_mask[train_end:]
print(f"Train shapes: {X_spatial_train.shape}, {X_station_train.shape}, {y_train.shape}")
print(f"Validation shapes: {X_spatial_val.shape}, {X_station_val.shape}, {y_val.shape}")

# Embed stations
embedded_stations_train = embed_stations_into_grid(X_station_train, station_masks)
X_spatial_train_combined = np.concatenate([X_spatial_train, embedded_stations_train], axis=-1)
embedded_stations_val = embed_stations_into_grid(X_station_val, station_masks)
X_spatial_val_combined = np.concatenate([X_spatial_val, embedded_stations_val], axis=-1)
print(f"Combined train input shape: {X_spatial_train_combined.shape}")
print(f"Combined val input shape: {X_spatial_val_combined.shape}")

# Prepare target data with mask
y_train_with_mask = np.stack([y_train, (~y_mask_train).astype(np.float32)], axis=-1)
y_val_with_mask = np.stack([y_val, (~y_mask_val).astype(np.float32)], axis=-1)
print(f"Training mask valid ratio: {(~y_mask_train).mean()}")

# Define Masked Loss and Metric Functions
def masked_mse(y_true, y_pred):
    y_val = y_true[..., 0:1]
    mask = y_true[..., 1:2]
    se = tf.square(y_pred - y_val)
    num = tf.reduce_sum(se * mask, axis=[1, 2, 3])
    den = tf.reduce_sum(mask, axis=[1, 2, 3]) + 1e-7
    return num / den

def masked_mae(y_true, y_pred):
    y_val = y_true[..., 0:1]
    mask = y_true[..., 1:2]
    ae = tf.abs(y_pred - y_val)
    num = tf.reduce_sum(ae * mask, axis=[1, 2, 3])
    den = tf.reduce_sum(mask, axis=[1, 2, 3]) + 1e-7
    return num / den

# Attention Gate 
def attention_gate(x, g, inter_channels):
    theta_x = Conv2D(inter_channels, (1, 1), padding='same')(x)
    phi_g = Conv2D(inter_channels, (1, 1), padding='same')(g)
    concat_xg = tf.keras.layers.add([theta_x, phi_g])
    act_xg = tf.keras.layers.Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = tf.keras.layers.Activation('sigmoid')(psi)
    return tf.keras.layers.multiply([x, sigmoid_xg])
    
@register_keras_serializable(package='Custom', name='ExpandDimsLast')
class ExpandDimsLast(Layer):
    def call(self, inputs):
        return tf.expand_dims(inputs, -1)
    def compute_output_shape(self, input_shape):
        return (*input_shape, 1)
    
# Attention U-Net Model 
def build_attention_unet_model(hp, input_tensor):
    filters = hp.Int('base_filters', min_value=64, max_value=128, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.3, step=0.1)
    l2_weight = hp.Choice('l2_weight', [1e-8, 1e-7, 1e-6, 1e-5, 1e-4])
    
    c1 = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_weight), activation=None)(input_tensor)
    c1 = LayerNormalization()(c1)
    c1 = Activation('relu')(c1)
    c1 = Dropout(dropout_rate)(c1)
    c1 = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_weight), activation=None)(c1)
    c1 = LayerNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(filters * 2, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_weight), activation=None)(p1)
    c2 = LayerNormalization()(c2)
    c2 = Activation('relu')(c2)
    c2 = Dropout(dropout_rate)(c2)
    c2 = Conv2D(filters * 2, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_weight), activation=None)(c2)
    c2 = LayerNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(filters * 4, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_weight), activation=None)(p2)
    c3 = LayerNormalization()(c3)
    c3 = Activation('relu')(c3)
    c3 = Dropout(dropout_rate)(c3)
    c3 = Conv2D(filters * 4, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_weight), activation=None)(c3)
    c3 = LayerNormalization()(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(filters * 8, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_weight), activation=None)(p3)
    c4 = LayerNormalization()(c4)
    c4 = Activation('relu')(c4)
    c4 = Dropout(dropout_rate)(c4)
    c4 = Conv2D(filters * 8, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_weight), activation=None)(c4)
    c4 = LayerNormalization()(c4)
    c4 = Activation('relu')(c4)
    
    u5 = UpSampling2D((2, 2))(c4)
    att_c3 = attention_gate(c3, u5, inter_channels=filters * 4 // 2)
    u5 = Concatenate()([u5, att_c3])
    c5 = Conv2D(filters * 4, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_weight), activation=None)(u5)
    c5 = LayerNormalization()(c5)
    c5 = Activation('relu')(c5)
    c5 = Dropout(dropout_rate)(c5)
    c5 = Conv2D(filters * 4, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_weight), activation=None)(c5)
    c5 = LayerNormalization()(c5)
    c5 = Activation('relu')(c5)
    
    u6 = UpSampling2D((2, 2))(c5)
    att_c2 = attention_gate(c2, u6, inter_channels=filters * 2 // 2)
    u6 = Concatenate()([u6, att_c2])
    c6 = Conv2D(filters * 2, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_weight), activation=None)(u6)
    c6 = LayerNormalization()(c6)
    c6 = Activation('relu')(c6)
    c6 = Dropout(dropout_rate)(c6)
    c6 = Conv2D(filters * 2, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_weight), activation=None)(c6)
    c6 = LayerNormalization()(c6)
    c6 = Activation('relu')(c6)
    
    u7 = UpSampling2D((2, 2))(c6)
    att_c1 = attention_gate(c1, u7, inter_channels=filters // 2)
    u7 = Concatenate()([u7, att_c1])
    c7 = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_weight), activation=None)(u7)
    c7 = LayerNormalization()(c7)
    c7 = Activation('relu')(c7)
    c7 = Dropout(dropout_rate)(c7)
    c7 = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_weight), activation=None)(c7)
    c7 = LayerNormalization()(c7)
    c7 = Activation('relu')(c7)
    
    outputs = Conv2D(1, (1, 1), activation='linear')(c7)
    return outputs

# ConvLSTM + Attention U-Net Model 
def build_convlstm_unet_model(hp):
    with strategy.scope():
        # 1) data input
        data_in  = Input(shape=(seq_length, padded_height, padded_width, ch_per_timestep), name='data_input')
        # 2) mask input (1=valid, 0=invalid)
        mask_in  = Input(shape=(padded_height, padded_width), name='mask_input')
        # expand mask to H×W×1 so we can multiply
        mask_exp = ExpandDimsLast(name='mask_expand')(mask_in)

        # … your ConvLSTM core as before …
        x = ConvLSTM2D(filters=hp.Int('convlstm_filters',32,128,32),
                       kernel_size=(3,3), padding='same', return_sequences=True)(data_in)
        x = LayerNormalization()(x)
        x = ConvLSTM2D(filters=x.shape[-1], kernel_size=(3,3), padding='same', return_sequences=False)(x)
        x = LayerNormalization()(x)

        # attention U-Net head
        unet_out = build_attention_unet_model(hp, x)

        # zero out invalid pixels before loss & metrics
        masked_out = tf.keras.layers.Multiply(name='mask_final')([unet_out, mask_exp])

        model = Model(inputs=[data_in, mask_in], outputs=masked_out, name='ConvLSTM_UNet_with_Mask')
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss=masked_mse,
            metrics=[masked_mse, masked_mae]
        )
        return model

# Custom Tuner
class MyBayesianTuner(kt.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        early_stp = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-8,
            verbose=1
        )
        
        # save full model (.keras)
        ckpt_full = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join('UNET_OUTPUTS','Models',f'model_{trial.trial_id}.keras'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        # save weights only (.weights.h5)
        ckpt_weights = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join('UNET_OUTPUTS','Models',f'model_{trial.trial_id}.weights.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
        kwargs['callbacks'] = [early_stp, reduce_lr, ckpt_full, ckpt_weights]
        
        results = super(MyBayesianTuner, self).run_trial(trial, *args, **kwargs)
        
        # Also write out the chosen hyperparameters to JSON —
        hp_values = trial.hyperparameters.values  # dict of hp_name: value
        json_path = os.path.join('UNET_OUTPUTS','Models', f'model_{trial.trial_id}.json')
        with open(json_path, 'w') as fp:
            json.dump(hp_values, fp, indent=2)
            
        visualize_trial(trial.trial_id)
        return results

# Visualization Function
def visualize_trial(trial_id):
    model_fp = os.path.join('UNET_OUTPUTS', 'Models', f'model_{trial_id}.keras')
    if not os.path.exists(model_fp):
        return

    custom_objects = {
        'masked_mse': masked_mse,
        'masked_mae': masked_mae,
        'ExpandDimsLast': ExpandDimsLast
    }
    model = tf.keras.models.load_model(model_fp, custom_objects=custom_objects)

    y_pred = model.predict([X_spatial_val_combined, valid_mask_val], batch_size=4)

    y_pred_orig = (y_pred[..., 0] - 0.1) / 0.9 * (wl_max - wl_min) + wl_min
    y_val_orig  = (y_val - 0.1) / 0.9 * (wl_max - wl_min) + wl_min
    y_pred_mask = np.where(y_mask_val, np.nan, y_pred_orig)

    vis_dir = os.path.join('UNET_OUTPUTS', 'visualization', f'model_{trial_id}')
    os.makedirs(vis_dir, exist_ok=True)

    for i in range(min(2, y_pred.shape[0])):
        fig, axs = plt.subplots(1, 3, figsize=(15,5))
        axs[0].imshow(y_val_orig[i], cmap='viridis')
        axs[0].set_title('Actual')
        axs[1].imshow(y_pred_mask[i], cmap='viridis')
        axs[1].set_title('Predicted')
        diff = y_pred_mask[i] - y_val_orig[i]
        vmax = np.nanmax(np.abs(diff))
        axs[2].imshow(diff, cmap='bwr', vmin=-vmax, vmax=vmax)
        axs[2].set_title('Difference')
        for ax in axs:
            fig.colorbar(ax.images[0], ax=ax, shrink=0.6)
        fig.tight_layout()
        fig.savefig(os.path.join(vis_dir, f'prediction_sample_{i}.png'))
        plt.close(fig)

    val_start = train_end
    times = np.arange(val_start + seq_length, val_start + seq_length + len(y_val))

    num_stations = len(adjusted_station_grid_indices)
    cols = 3
    rows = int(np.ceil(num_stations / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), sharex=True)
    axes = axes.flatten()
    for idx, (st, (row, col)) in enumerate(adjusted_station_grid_indices.items()):
        ts_csv = station_df.iloc[times, idx].values
        ts_rast = y_val_orig[:, row, col]
        ts_pred = y_pred_mask[:, row, col]
        ax = axes[idx]
        ax.plot(times, ts_csv, label='CSV')
        ax.plot(times, ts_rast, label='Raster')
        ax.plot(times, ts_pred, label='Predicted')
        ax.set_title(f'Station {st}')
        if idx == 0:
            ax.legend(loc='upper right')
    for ax in axes[num_stations:]:
        fig.delaxes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(vis_dir, 'all_stations_timeseries.png'))
    plt.close(fig)

    plt.figure()
    plt.imshow(np.sum(embedded_stations_val[0,0,:,:,:], axis=-1), cmap='viridis')
    plt.title('Initial Station Embeddings Sum (First Timestep)')
    plt.colorbar()
    plt.savefig(os.path.join(vis_dir, 'initial_embedding.png'))
    plt.close()

    trial = tuner.oracle.get_trial(str(trial_id))
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

    hist_df.to_csv(os.path.join(vis_dir, 'loss_history.csv'), index=False)
    plt.figure()
    plt.plot(hist_df['epoch'], hist_df['value'])
    plt.title(f'{metric_name} per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.savefig(os.path.join(vis_dir, 'loss_curve.png'))
    plt.close()

# Prepare datasets
batch_size = 4

# Build per-sample valid masks (1=valid, 0=invalid) from y_mask arrays
valid_mask_train = (~y_mask_train).astype(np.float32)  # shape (num_train, H_pad, W_pad)
valid_mask_val   = (~y_mask_val).astype(np.float32)    # shape (num_val,   H_pad, W_pad)

# Now include those as the mask_input to the model
train_ds = (
    tf.data.Dataset.from_tensor_slices(
        ((X_spatial_train_combined, valid_mask_train), y_train_with_mask)
    )
    .shuffle(buffer_size=len(X_spatial_train_combined), seed=seed_value)
    .repeat()
    .batch(batch_size, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices(
        ((X_spatial_val_combined, valid_mask_val), y_val_with_mask)
    )
    .repeat()
    .batch(batch_size, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

steps_per_epoch  = len(X_spatial_train_combined) // batch_size
validation_steps = len(X_spatial_val_combined)   // batch_size

# Hyperparameter Tuning
tuner_dir = os.path.join('UNET_OUTPUTS', 'tuner_dir')
os.makedirs(os.path.join('UNET_OUTPUTS', 'Models'), exist_ok=True)
os.makedirs(os.path.join('UNET_OUTPUTS', 'visualization'), exist_ok=True)

try:
    tuner = MyBayesianTuner(
        hypermodel=build_convlstm_unet_model,
        objective='val_loss',
        max_trials=50,
        executions_per_trial=1,
        directory=tuner_dir,
        project_name='convlstm_unet_tuning',
        seed=seed_value,
        overwrite=False
    )
except Exception as e:
    logging.warning(f"Failed to load existing tuner state: {e}. Resetting tuner directory.")
    if os.path.exists(tuner_dir):
        shutil.rmtree(tuner_dir)
    tuner = MyBayesianTuner(
        hypermodel=build_convlstm_unet_model,
        objective='val_loss',
        max_trials=50,
        executions_per_trial=1,
        directory=tuner_dir,
        project_name='convlstm_unet_tuning',
        seed=seed_value,
        overwrite=True
    )

try:
    tuner.search(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=300,
        callbacks=[],
        verbose=1
    )
except Exception as e:
    logging.error(f"Error during tuner search: {e}")
    if tuner.oracle.get_best_trials(1):
        logging.info("Using best trial found so far.")
    else:
        raise RuntimeError("No valid trials completed. Cannot proceed with training.")

