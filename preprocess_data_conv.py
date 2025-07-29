"""Preprocess ASL sign language data for Conv1D model training."""

import gc
import os
import math
import zipfile
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

# ================================
# Configuration
# ================================
@dataclass
class Config:
    """Central configuration for data preprocessing."""
    # Data dimensions
    N_ROWS: int = 543  # landmarks per frame
    N_DIMS: int = 3
    DIM_NAMES: List[str] = None
    
    # Model parameters
    NUM_CLASSES: int = 250
    INPUT_SIZE: int = 32
    MASK_VAL: int = 4237
    SEED: int = 42
    
    # Hand dominance threshold
    HAND_THRESHOLD: float = 0.60  # Keep frames with >= 60% non-NaN hand points
    
    # Data processing
    CLIP_RANGE: Tuple[float, float] = (-10.0, 10.0)
    MIN_STD: float = 0.01
    
    def __post_init__(self):
        if self.DIM_NAMES is None:
            self.DIM_NAMES = ['x', 'y', 'z']

# ================================
# Landmark Indices
# ================================
class LandmarkIndices:
    """Manage landmark indices for face, hands, and pose."""
    
    # Lips landmarks (40 points)
    LIPS = np.array([
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    ])
    
    # Hand landmarks (21 points each)
    LEFT_HAND = np.arange(468, 489)
    RIGHT_HAND = np.arange(522, 543)
    
    # Pose landmarks (5 points each side)
    LEFT_POSE = np.array([502, 504, 506, 508, 510])
    RIGHT_POSE = np.array([503, 505, 507, 509, 511])
    
    def __init__(self):
        """Initialize combined landmark arrays."""
        # Combined indices for left/right dominant configurations
        self.left_dominant = np.concatenate([self.LIPS, self.LEFT_HAND, self.LEFT_POSE])
        self.right_dominant = np.concatenate([self.LIPS, self.RIGHT_HAND, self.RIGHT_POSE])
        self.all_hands = np.concatenate([self.LEFT_HAND, self.RIGHT_HAND])
        
        # Calculate relative positions in processed data
        self.n_cols = len(self.left_dominant)
        self._calculate_relative_indices()
    
    def _calculate_relative_indices(self):
        """Calculate relative indices for processed data."""
        # Indices in processed array
        self.lips_idx = np.where(np.isin(self.left_dominant, self.LIPS))[0]
        self.left_hand_idx = np.where(np.isin(self.left_dominant, self.LEFT_HAND))[0]
        self.right_hand_idx = np.where(np.isin(self.left_dominant, self.RIGHT_HAND))[0]
        self.pose_idx = np.where(np.isin(self.left_dominant, self.LEFT_POSE))[0]
        
        # Start positions
        self.lips_start = 0
        self.left_hand_start = len(self.lips_idx)
        self.right_hand_start = self.left_hand_start + len(self.left_hand_idx)
        self.pose_start = self.right_hand_start + len(self.right_hand_idx)

# ================================
# Preprocessing Layer
# ================================
class PreprocessLayer(tf.keras.layers.Layer):
    """TensorFlow layer for data preprocessing in TFLite."""
    
    def __init__(self):
        super().__init__()
        self.landmarks = LandmarkIndices()
        self._init_constants()
    
    def _init_constants(self):
        """Initialize TensorFlow constants."""
        # Normalization correction for right-dominant hand
        correction = np.zeros((self.landmarks.n_cols, 3), dtype=np.float32)
        correction[self.landmarks.left_hand_idx, 0] = 0.50
        correction[self.landmarks.pose_idx, 0] = 0.50
        self.norm_correction = tf.constant(correction)
        
        # Landmark indices
        self.left_dominant_idx = tf.constant(self.landmarks.left_dominant, dtype=tf.int32)
        self.right_dominant_idx = tf.constant(self.landmarks.right_dominant, dtype=tf.int32)
        self.left_hand_idx = tf.constant(self.landmarks.LEFT_HAND, dtype=tf.int32)
        self.right_hand_idx = tf.constant(self.landmarks.RIGHT_HAND, dtype=tf.int32)
        
        # Constants
        self.n_hand_points = tf.constant(len(self.landmarks.LEFT_HAND), dtype=tf.int32)
        self.hand_threshold = tf.constant(Config.HAND_THRESHOLD, dtype=tf.float32)
        self.flip_x = tf.constant([-1, 1, 1], dtype=tf.float32)
    
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, Config.N_ROWS, 3], dtype=tf.float32),))
    def call(self, data):
        """Process input data: detect dominant hand, filter frames, normalize."""
        # Detect dominant hand
        left_dominant = self._detect_dominant_hand(data)
        
        # Filter frames with sufficient hand data
        data, frame_indices = self._filter_valid_frames(data, left_dominant)
        
        # Extract relevant landmarks
        data = self._extract_landmarks(data, left_dominant)
        
        # Adjust coordinates for right-dominant hand
        if not left_dominant:
            data = self._adjust_right_dominant(data)
        
        # Resize to target length
        data, frame_indices = self._resize_sequence(data, frame_indices)
        
        # Handle NaN frames
        data, frame_indices = self._handle_nan_frames(data, frame_indices)
        
        # Normalize
        data = self._normalize_data(data)
        
        return data, frame_indices
    
    def _detect_dominant_hand(self, data):
        """Determine dominant hand based on non-NaN values."""
        left_sum = tf.reduce_sum(tf.where(tf.math.is_nan(
            tf.gather(data, self.left_hand_idx, axis=1)), 0, 1))
        right_sum = tf.reduce_sum(tf.where(tf.math.is_nan(
            tf.gather(data, self.right_hand_idx, axis=1)), 0, 1))
        return left_sum >= right_sum
    
    def _filter_valid_frames(self, data, left_dominant):
        """Keep frames with sufficient hand landmarks."""
        hand_idx = self.left_hand_idx if left_dominant else self.right_hand_idx
        hand_data = tf.gather(data, hand_idx, axis=1)
        
        # Count non-NaN values per frame
        non_nan_count = tf.reduce_sum(
            tf.where(tf.math.is_nan(hand_data), 0, 1), axis=[1, 2])
        
        # Keep frames above threshold
        min_points = tf.cast(self.n_hand_points, tf.float32) * self.hand_threshold
        valid_frames = tf.where(tf.cast(non_nan_count, tf.float32) >= min_points)
        valid_frames = tf.squeeze(valid_frames, axis=1)
        
        # Filter data
        data = tf.gather(data, valid_frames, axis=0)
        frame_indices = tf.cast(valid_frames, tf.float32)
        frame_indices = frame_indices - tf.reduce_min(frame_indices)
        
        return data, frame_indices
    
    def _extract_landmarks(self, data, left_dominant):
        """Extract relevant landmarks based on dominant hand."""
        indices = self.left_dominant_idx if left_dominant else self.right_dominant_idx
        return tf.gather(data, indices, axis=1)
    
    def _adjust_right_dominant(self, data):
        """Mirror and adjust coordinates for right-dominant hand."""
        data_clean = tf.where(tf.math.is_nan(data), 0.0, data)
        return self.norm_correction + (data_clean * self.flip_x)
    
    def _resize_sequence(self, data, frame_indices):
        """Resize sequence to target length."""
        n_frames = tf.shape(data)[0]
        
        if n_frames < Config.INPUT_SIZE:
            # Pad shorter sequences
            pad_size = Config.INPUT_SIZE - n_frames
            data = tf.pad(data, [[0, pad_size], [0, 0], [0, 0]])
            frame_indices = tf.pad(frame_indices, [[0, pad_size]], constant_values=-1)
        else:
            # Downsample longer sequences
            indices = self._downsample_indices(n_frames)
            data = tf.gather(data, indices, axis=0)
            frame_indices = tf.gather(frame_indices, indices, axis=0)
        
        return data, frame_indices
    
    def _downsample_indices(self, n_frames):
        """Generate downsampled indices with reduced edge probability."""
        # Lower probability for first/last frames
        probs = tf.concat([
            [0.05],
            tf.fill([n_frames - 2], 0.95),
            [0.05]
        ], axis=0)
        
        # Sample without replacement
        seed = (tf.reduce_sum(tf.cast(n_frames, tf.int32)), 
                tf.reduce_max(tf.cast(n_frames, tf.int32)))
        
        return tf.reshape(
            tf.random.stateless_categorical(
                tf.math.log([probs]), Config.INPUT_SIZE, seed),
            [Config.INPUT_SIZE])
    
    def _handle_nan_frames(self, data, frame_indices):
        """Replace frames containing NaN with zeros."""
        has_nan = tf.reduce_any(tf.math.is_nan(data), axis=[1, 2])
        data = tf.where(has_nan[..., tf.newaxis, tf.newaxis], 0.0, data)
        frame_indices = tf.where(has_nan, -1.0, frame_indices)
        return data, frame_indices
    
    def _normalize_data(self, data):
        """Normalize data with robust statistics."""
        # Calculate statistics
        mean = tf.reduce_mean(data, axis=[0, 1], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), 0.0, mean)
        
        std = tf.math.reduce_std(data, axis=[0, 1], keepdims=True)
        std = tf.where(tf.math.is_nan(std), 1.0, std)
        std = tf.where(std < Config.MIN_STD, 1.0, std)
        
        # Normalize and clip
        data = (data - mean) / std
        data = tf.where(tf.math.is_nan(data), 0.0, data)
        data = tf.clip_by_value(data, *Config.CLIP_RANGE)
        
        return data

# ================================
# Data I/O Functions
# ================================
def load_parquet_landmarks(path: str) -> np.ndarray:
    """Load landmark data from parquet file."""
    df = pd.read_parquet(path, columns=['x', 'y', 'z'])
    n_frames = len(df) // Config.N_ROWS
    return df.values.reshape(n_frames, Config.N_ROWS, Config.N_DIMS).astype(np.float32)

def save_compressed(data: np.ndarray, filename: str):
    """Save numpy array as compressed zip."""
    npy_name = filename.replace('.zip', '.npy')
    with zipfile.ZipFile(filename, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        np.save(npy_name, data)
        zf.write(npy_name)
        os.remove(npy_name)

def load_compressed(filename: str) -> np.ndarray:
    """Load numpy array from compressed zip."""
    with zipfile.ZipFile(filename, 'r') as zf:
        zf.extractall()
    npy_name = filename.replace('.zip', '.npy')
    data = np.load(npy_name)
    os.remove(npy_name)
    return data

# ================================
# Data Processing Functions
# ================================
def create_nan_samples() -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic samples with masked values for robustness."""
    n_samples = 10
    landmarks = LandmarkIndices()
    
    # Initialize arrays
    samples = np.zeros((n_samples, Config.INPUT_SIZE, landmarks.n_cols, Config.N_DIMS), dtype=np.float32)
    frame_idxs = np.full((n_samples, Config.INPUT_SIZE), -1, dtype=np.float32)
    
    # Define masking patterns
    patterns = [
        (0, slice(0, 7)),          # First 7 frames
        (1, slice(0, 30)),         # First 30 frames
        (2, slice(None)),          # All frames
        (3, slice(0, 1)),          # First frame only
        (4, slice(0, -1)),         # All but last
        (5, slice(16, 18)),        # Middle 2 frames
        (6, slice(0, None, 2)),    # Even frames
        (7, slice(1, None, 2)),    # Odd frames
        (8, slice(0, None, 3)),    # Every 3rd frame
        (9, slice(1, None, 5)),    # Every 5th frame starting at 1
    ]
    
    # Apply patterns
    for idx, pattern in patterns:
        samples[idx, pattern] = Config.MASK_VAL
        valid_frames = np.arange(Config.INPUT_SIZE)[pattern]
        frame_idxs[idx, :len(valid_frames)] = valid_frames
    
    return samples, frame_idxs

def process_dataset(train_df: pd.DataFrame, 
                   preprocess_layer: PreprocessLayer,
                   show_progress: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process entire dataset through preprocessing pipeline."""
    iterator = tqdm(train_df[['file_path', 'sign_ord']].values) if show_progress else train_df[['file_path', 'sign_ord']].values
    
    X, frame_indices, y = [], [], []
    
    for file_path, label in iterator:
        # Load and preprocess
        data = load_parquet_landmarks(file_path)
        processed, frames = preprocess_layer(data)
        
        X.append(processed.numpy())
        frame_indices.append(frames.numpy())
        y.append(label)
    
    return (np.array(X, dtype=np.float32),
            np.array(frame_indices, dtype=np.float32),
            np.array(y, dtype=np.int32))

def split_train_val(X: np.ndarray, y: np.ndarray, 
                   frame_indices: np.ndarray,
                   participant_ids: np.ndarray,
                   val_size: float = 0.10) -> Dict:
    """Split data by participant for robust validation."""
    # Create synthetic samples
    nan_samples, nan_frames = create_nan_samples()
    
    # Split by participant
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=Config.SEED)
    train_idx, val_idx = next(splitter.split(X, y, groups=participant_ids))
    
    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    frames_train, frames_val = frame_indices[train_idx], frame_indices[val_idx]
    
    # Add synthetic samples to training
    X_train = np.concatenate([X_train, nan_samples])
    y_train = np.concatenate([y_train, np.zeros(len(nan_samples), dtype=np.int32)])
    frames_train = np.concatenate([frames_train, nan_frames])
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'frames_train': frames_train,
        'X_val': X_val,
        'y_val': y_val,
        'frames_val': frames_val,
        'validation_data': ({'frames': X_val, 'non_empty_frame_idxs': frames_val}, y_val)
    }

# ================================
# Analysis Functions
# ================================
def analyze_frame_statistics(train_df: pd.DataFrame, 
                           sample_size: int = 1000,
                           show_plots: bool = False):
    """Analyze frame statistics in dataset."""
    # Sample files
    sample_files = train_df['file_path'].sample(min(sample_size, len(train_df)), 
                                               random_state=Config.SEED)
    
    stats = {
        'unique_frames': [],
        'missing_frames': [],
        'max_frame': []
    }
    
    for file_path in tqdm(sample_files, desc="Analyzing frames"):
        df = pd.read_parquet(file_path, columns=['frame'])
        
        unique = df['frame'].nunique()
        max_frame = df['frame'].max()
        min_frame = df['frame'].min()
        missing = (max_frame - min_frame + 1) - unique
        
        stats['unique_frames'].append(unique)
        stats['missing_frames'].append(missing)
        stats['max_frame'].append(max_frame)
    
    # Print statistics
    percentiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    for name, values in stats.items():
        print(f"\n{name.upper()}:")
        print(pd.Series(values).describe(percentiles=percentiles))
    
    # Plot if requested
    if show_plots:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (name, values) in zip(axes, stats.items()):
            pd.Series(values).hist(bins=50, ax=ax)
            ax.set_title(name.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# ================================
# Main Pipeline
# ================================
def prepare_data(config: Dict[str, bool] = None) -> Dict:
    """Main data preparation pipeline."""
    if config is None:
        config = {
            'preprocess': True,
            'use_validation': False,
            'show_plots': False,
            'analyze_stats': True
        }
    
    # Load metadata
    print("Loading metadata...")
    train_df = pd.read_csv('train.csv')
    train_df['file_path'] = train_df['path'].apply(lambda x: f'./{x}')
    train_df['sign_ord'] = train_df['sign'].astype('category').cat.codes
    
    # Create translation dictionaries
    sign2ord = train_df[['sign', 'sign_ord']].set_index('sign')['sign_ord'].to_dict()
    ord2sign = {v: k for k, v in sign2ord.items()}
    
    print(f"Dataset: {len(train_df)} samples, {train_df['sign'].nunique()} unique signs")
    
    # Analyze statistics
    if config['analyze_stats']:
        analyze_frame_statistics(train_df, show_plots=config['show_plots'])
    
    # Process or load data
    if config['preprocess']:
        print("\nProcessing data...")
        preprocess_layer = PreprocessLayer()
        X, frame_indices, y = process_dataset(train_df, preprocess_layer)
        
        # Save processed data
        print("Saving processed data...")
        save_compressed(X, 'X.zip')
        save_compressed(frame_indices, 'NON_EMPTY_FRAME_IDXS.zip')
        save_compressed(y, 'y.zip')
    else:
        print("\nLoading preprocessed data...")
        X = load_compressed('X.zip')
        frame_indices = load_compressed('NON_EMPTY_FRAME_IDXS.zip')
        y = load_compressed('y.zip')
    
    print(f"Data shape: {X.shape}, Memory: {(X.nbytes + frame_indices.nbytes + y.nbytes) / 1e9:.2f} GB")
    
    # Split data
    if config['use_validation']:
        print("\nSplitting data with validation set...")
        split_data = split_train_val(X, y, frame_indices, train_df['participant_id'].values)
        result = {
            'X_train': split_data['X_train'],
            'y_train': split_data['y_train'],
            'NON_EMPTY_FRAME_IDXS_TRAIN': split_data['frames_train'],
            'validation_data': split_data['validation_data'],
            'y_val': split_data['y_val']
        }
    else:
        print("\nUsing all data for training...")
        nan_samples, nan_frames = create_nan_samples()
        result = {
            'X_train': np.concatenate([X, nan_samples]),
            'y_train': np.concatenate([y, np.zeros(len(nan_samples), dtype=np.int32)]),
            'NON_EMPTY_FRAME_IDXS_TRAIN': np.concatenate([frame_indices, nan_frames]),
            'validation_data': None,
            'y_val': None
        }
    
    # Add metadata
    result.update({
        'SIGN2ORD': sign2ord,
        'ORD2SIGN': ord2sign,
        'train': train_df
    })
    
    # Cleanup
    del X, y, frame_indices
    gc.collect()
    
    print("\nData preparation complete!")
    return result

if __name__ == "__main__":
    # Run with default configuration
    data = prepare_data({
        'preprocess': True,
        'use_validation': False,
        'show_plots': False,
        'analyze_stats': True
    })