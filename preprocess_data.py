"""
Data preprocessing module for Sign Language Recognition Transformer model.

This module handles data loading, preprocessing, and preparation for the 
transformer-based sign language recognition model.
"""

import os
import gc
import zipfile
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
import matplotlib as mpl

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
    
    # Training settings
    BATCH_ALL_SIGNS_N: int = 2
    BATCH_SIZE: int = 128
    
    # Visualization
    SHOW_PLOTS: bool = False
    IS_INTERACTIVE: bool = True
    
    def __post_init__(self):
        if self.DIM_NAMES is None:
            self.DIM_NAMES = ['x', 'y', 'z']
    
    @property
    def verbose(self) -> int:
        return 1 if self.IS_INTERACTIVE else 2


# Configure matplotlib globally
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 24
})


# ================================
# Landmark Indices
# ================================
class LandmarkIndices:
    """Manage landmark indices for face, hands, and pose."""
    
    # Lips landmarks (40 points)
    LIPS_IDXS0 = np.array([
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    ])
    
    # Hand landmarks (21 points each)
    LEFT_HAND_IDXS0 = np.arange(468, 489)
    RIGHT_HAND_IDXS0 = np.arange(522, 543)
    
    # Pose landmarks (5 points each side)
    LEFT_POSE_IDXS0 = np.array([502, 504, 506, 508, 510])
    RIGHT_POSE_IDXS0 = np.array([503, 505, 507, 509, 511])
    
    def __init__(self):
        """Initialize combined landmark arrays."""
        # Combined indices for left/right dominant configurations
        self.LANDMARK_IDXS_LEFT_DOMINANT0 = np.concatenate([
            self.LIPS_IDXS0, self.LEFT_HAND_IDXS0, self.LEFT_POSE_IDXS0
        ])
        self.LANDMARK_IDXS_RIGHT_DOMINANT0 = np.concatenate([
            self.LIPS_IDXS0, self.RIGHT_HAND_IDXS0, self.RIGHT_POSE_IDXS0
        ])
        self.HAND_IDXS0 = np.concatenate([self.LEFT_HAND_IDXS0, self.RIGHT_HAND_IDXS0])
        
        # Number of columns after preprocessing
        self.N_COLS = len(self.LANDMARK_IDXS_LEFT_DOMINANT0)
        
        # Calculate relative indices in processed data
        self._calculate_relative_indices()
    
    def _calculate_relative_indices(self):
        """Calculate relative indices for processed data."""
        # Indices in processed array
        self.LIPS_IDXS = np.argwhere(np.isin(
            self.LANDMARK_IDXS_LEFT_DOMINANT0, self.LIPS_IDXS0
        )).squeeze()
        self.LEFT_HAND_IDXS = np.argwhere(np.isin(
            self.LANDMARK_IDXS_LEFT_DOMINANT0, self.LEFT_HAND_IDXS0
        )).squeeze()
        self.RIGHT_HAND_IDXS = np.argwhere(np.isin(
            self.LANDMARK_IDXS_LEFT_DOMINANT0, self.RIGHT_HAND_IDXS0
        )).squeeze()
        self.HAND_IDXS = np.argwhere(np.isin(
            self.LANDMARK_IDXS_LEFT_DOMINANT0, self.HAND_IDXS0
        )).squeeze()
        self.POSE_IDXS = np.argwhere(np.isin(
            self.LANDMARK_IDXS_LEFT_DOMINANT0, self.LEFT_POSE_IDXS0
        )).squeeze()
        
        # Start positions
        self.LIPS_START = 0
        self.LEFT_HAND_START = len(self.LIPS_IDXS)
        self.RIGHT_HAND_START = self.LEFT_HAND_START + len(self.LEFT_HAND_IDXS)
        self.POSE_START = self.RIGHT_HAND_START + len(self.RIGHT_HAND_IDXS)


# ================================
# Preprocessing Layer
# ================================
class PreprocessLayer(tf.keras.layers.Layer):
    """
    TensorFlow layer to process data in TFLite.
    Data needs to be processed in the model itself, so we cannot use Python.
    """
    
    def __init__(self):
        super(PreprocessLayer, self).__init__()
        self.landmarks = LandmarkIndices()
        self.config = Config()
        self._init_constants()
    
    def _init_constants(self):
        """Initialize TensorFlow constants."""
        normalisation_correction = tf.constant([
            # Add 0.50 to left hand (original right hand) and subtract 0.50 from right hand (original left hand)
            [0] * len(self.landmarks.LIPS_IDXS) + 
            [0.50] * len(self.landmarks.LEFT_HAND_IDXS) + 
            [0.50] * len(self.landmarks.POSE_IDXS),
            # Y coordinates stay intact
            [0] * len(self.landmarks.LANDMARK_IDXS_LEFT_DOMINANT0),
            # Z coordinates stay intact
            [0] * len(self.landmarks.LANDMARK_IDXS_LEFT_DOMINANT0),
        ], dtype=tf.float32)
        self.normalisation_correction = tf.transpose(normalisation_correction, [1, 0])
    
    def pad_edge(self, t, repeats, side):
        """Pad tensor by repeating edge values."""
        if side == 'LEFT':
            return tf.concat((tf.repeat(t[:1], repeats=repeats, axis=0), t), axis=0)
        elif side == 'RIGHT':
            return tf.concat((t, tf.repeat(t[-1:], repeats=repeats, axis=0)), axis=0)
    
    @tf.function(
        input_signature=(tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32),),
    )
    def call(self, data0):
        """Process input data: detect dominant hand, filter frames, normalize."""
        # Number of Frames in Video
        N_FRAMES0 = tf.shape(data0)[0]
        
        # Find dominant hand by comparing summed absolute coordinates
        left_hand_sum = tf.math.reduce_sum(tf.where(
            tf.math.is_nan(tf.gather(data0, self.landmarks.LEFT_HAND_IDXS0, axis=1)), 0, 1
        ))
        right_hand_sum = tf.math.reduce_sum(tf.where(
            tf.math.is_nan(tf.gather(data0, self.landmarks.RIGHT_HAND_IDXS0, axis=1)), 0, 1
        ))
        left_dominant = left_hand_sum >= right_hand_sum
        
        # Count non NaN Hand values in each frame for the dominant hand
        if left_dominant:
            frames_hands_non_nan_sum = tf.math.reduce_sum(
                tf.where(
                    tf.math.is_nan(tf.gather(data0, self.landmarks.LEFT_HAND_IDXS0, axis=1)), 
                    0, 1
                ),
                axis=[1, 2],
            )
        else:
            frames_hands_non_nan_sum = tf.math.reduce_sum(
                tf.where(
                    tf.math.is_nan(tf.gather(data0, self.landmarks.RIGHT_HAND_IDXS0, axis=1)), 
                    0, 1
                ),
                axis=[1, 2],
            )
        
        # Find frames indices with coordinates of dominant hand
        non_empty_frames_idxs = tf.where(frames_hands_non_nan_sum > 0)
        non_empty_frames_idxs = tf.squeeze(non_empty_frames_idxs, axis=1)
        # Filter frames
        data = tf.gather(data0, non_empty_frames_idxs, axis=0)
        
        # Cast Indices in float32 to be compatible with Tensorflow Lite
        non_empty_frames_idxs = tf.cast(non_empty_frames_idxs, tf.float32)
        # Normalize to start with 0
        non_empty_frames_idxs -= tf.reduce_min(non_empty_frames_idxs)
        
        # Number of Frames in Filtered Video
        N_FRAMES = tf.shape(data)[0]
        
        # Gather Relevant Landmark Columns
        if left_dominant:
            data = tf.gather(data, self.landmarks.LANDMARK_IDXS_LEFT_DOMINANT0, axis=1)
        else:
            data = tf.gather(data, self.landmarks.LANDMARK_IDXS_RIGHT_DOMINANT0, axis=1)
            data = (
                self.normalisation_correction + (
                    (data - self.normalisation_correction) * 
                    tf.where(self.normalisation_correction != 0, -1.0, 1.0)
                )
            )
        
        # Video fits in INPUT_SIZE
        if N_FRAMES < self.config.INPUT_SIZE:
            # Pad With -1 to indicate padding
            non_empty_frames_idxs = tf.pad(
                non_empty_frames_idxs, 
                [[0, self.config.INPUT_SIZE - N_FRAMES]], 
                constant_values=-1
            )
            # Pad Data With Zeros
            data = tf.pad(
                data, 
                [[0, self.config.INPUT_SIZE - N_FRAMES], [0, 0], [0, 0]], 
                constant_values=0
            )
            # Fill NaN Values With 0
            data = tf.where(tf.math.is_nan(data), 0.0, data)
            return data, non_empty_frames_idxs
        # Video needs to be downsampled to INPUT_SIZE
        else:
            # Repeat
            if N_FRAMES < self.config.INPUT_SIZE ** 2:
                repeats = tf.math.floordiv(
                    self.config.INPUT_SIZE * self.config.INPUT_SIZE, N_FRAMES0
                )
                data = tf.repeat(data, repeats=repeats, axis=0)
                non_empty_frames_idxs = tf.repeat(non_empty_frames_idxs, repeats=repeats, axis=0)

            # Pad To Multiple Of Input Size
            pool_size = tf.math.floordiv(len(data), self.config.INPUT_SIZE)
            if tf.math.mod(len(data), self.config.INPUT_SIZE) > 0:
                pool_size += 1

            if pool_size == 1:
                pad_size = (pool_size * self.config.INPUT_SIZE) - len(data)
            else:
                pad_size = (pool_size * self.config.INPUT_SIZE) % len(data)

            # Pad Start/End with Start/End value
            pad_left = tf.math.floordiv(pad_size, 2) + tf.math.floordiv(self.config.INPUT_SIZE, 2)
            pad_right = tf.math.floordiv(pad_size, 2) + tf.math.floordiv(self.config.INPUT_SIZE, 2)
            if tf.math.mod(pad_size, 2) > 0:
                pad_right += 1

            # Pad By Concatenating Left/Right Edge Values
            data = self.pad_edge(data, pad_left, 'LEFT')
            data = self.pad_edge(data, pad_right, 'RIGHT')

            # Pad Non Empty Frame Indices
            non_empty_frames_idxs = self.pad_edge(non_empty_frames_idxs, pad_left, 'LEFT')
            non_empty_frames_idxs = self.pad_edge(non_empty_frames_idxs, pad_right, 'RIGHT')

            # Reshape to Mean Pool
            data = tf.reshape(data, [self.config.INPUT_SIZE, -1, self.landmarks.N_COLS, self.config.N_DIMS])
            non_empty_frames_idxs = tf.reshape(non_empty_frames_idxs, [self.config.INPUT_SIZE, -1])

            # Mean Pool
            data = tf.experimental.numpy.nanmean(data, axis=1)
            non_empty_frames_idxs = tf.experimental.numpy.nanmean(non_empty_frames_idxs, axis=1)

            # Fill NaN Values With 0
            data = tf.where(tf.math.is_nan(data), 0.0, data)
            
            return data, non_empty_frames_idxs


# ================================
# Data I/O Functions
# ================================
def print_shape_dtype(arrays: List[np.ndarray], names: List[str]):
    """Print shape and dtype for list of arrays."""
    for arr, name in zip(arrays, names):
        print(f'{name} shape: {arr.shape}, dtype: {arr.dtype}')


def load_relevant_data_subset(pq_path: str) -> np.ndarray:
    """Load landmark data from parquet file."""
    config = Config()
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / config.N_ROWS)
    data = data.values.reshape(n_frames, config.N_ROWS, len(data_columns))
    return data.astype(np.float32)


def save_compressed(data: np.ndarray, filename: str):
    """Save numpy array as compressed zip."""
    npy_name = filename.replace('.zip', '.npy')
    with zipfile.ZipFile(filename, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        # Save to temporary npy file first
        np.save(npy_name, data)
        # Add to zip and remove temporary file
        zf.write(npy_name, arcname=os.path.basename(npy_name))
        os.remove(npy_name)


def load_compressed(filename: str) -> np.ndarray:
    """Load numpy array from compressed zip."""
    with zipfile.ZipFile(filename, 'r') as zf:
        # Extract to current directory
        npy_name = os.path.basename(filename.replace('.zip', '.npy'))
        zf.extract(npy_name)
    data = np.load(npy_name)
    os.remove(npy_name)
    return data


# ================================
# Data Processing Functions
# ================================
def get_data(file_path: str, preprocess_layer: PreprocessLayer) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess a single sample."""
    # Load Raw Data
    data = load_relevant_data_subset(file_path)
    # Process Data Using Tensorflow
    data, non_empty_frame_idxs = preprocess_layer(data)
    return data.numpy(), non_empty_frame_idxs.numpy()


def process_dataset(train_df: pd.DataFrame, 
                   config: Config,
                   preprocess_layer: PreprocessLayer) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process entire dataset through preprocessing pipeline."""
    n_samples = len(train_df)
    landmarks = LandmarkIndices()
    
    # Create arrays to save data
    X = np.zeros([n_samples, config.INPUT_SIZE, landmarks.N_COLS, config.N_DIMS], dtype=np.float32)
    y = np.zeros([n_samples], dtype=np.int32)
    NON_EMPTY_FRAME_IDXS = np.full([n_samples, config.INPUT_SIZE], -1, dtype=np.float32)

    # Fill X/y
    for row_idx, (file_path, sign_ord) in enumerate(tqdm(train_df[['file_path', 'sign_ord']].values)):
        # Log message every 5000 samples
        if row_idx % 5000 == 0:
            print(f'Generated {row_idx}/{n_samples}')

        data, non_empty_frame_idxs = get_data(file_path, preprocess_layer)
        X[row_idx] = data
        y[row_idx] = sign_ord
        NON_EMPTY_FRAME_IDXS[row_idx] = non_empty_frame_idxs
        
        # Sanity check, data should not contain NaN values
        if np.isnan(data).sum() > 0:
            print(f'Warning: NaN values found in sample {row_idx}')

    return X, NON_EMPTY_FRAME_IDXS, y


def split_train_val(X: np.ndarray, y: np.ndarray, 
                   NON_EMPTY_FRAME_IDXS: np.ndarray,
                   participant_ids: np.ndarray,
                   val_size: float = 0.10,
                   seed: int = 42) -> Dict:
    """Split data by participant for robust validation."""
    # Split by participant
    splitter = GroupShuffleSplit(test_size=val_size, n_splits=2, random_state=seed)
    train_idx, val_idx = next(splitter.split(X, y, groups=participant_ids))
    
    # Split data
    X_train = X[train_idx]
    X_val = X[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    NON_EMPTY_FRAME_IDXS_TRAIN = NON_EMPTY_FRAME_IDXS[train_idx]
    NON_EMPTY_FRAME_IDXS_VAL = NON_EMPTY_FRAME_IDXS[val_idx]
    
    # Verify no participant overlap
    participants_train = participant_ids[train_idx]
    participants_val = participant_ids[val_idx]
    overlap = set(participants_train).intersection(set(participants_val))
    print(f'Patient ID Intersection Train/Val: {overlap}')
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'NON_EMPTY_FRAME_IDXS_TRAIN': NON_EMPTY_FRAME_IDXS_TRAIN,
        'X_val': X_val,
        'y_val': y_val,
        'NON_EMPTY_FRAME_IDXS_VAL': NON_EMPTY_FRAME_IDXS_VAL,
        'validation_data': ({
            'frames': X_val, 
            'non_empty_frame_idxs': NON_EMPTY_FRAME_IDXS_VAL
        }, y_val)
    }


# ================================
# Statistics Functions
# ================================
def analyze_frame_statistics(train_df: pd.DataFrame, 
                           config: Config,
                           sample_size: int = 1000):
    """Analyze frame statistics in dataset."""
    percentiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999]
    
    # Sample files
    N = min(sample_size, len(train_df))
    sample_files = train_df['file_path'].sample(N, random_state=config.SEED)
    
    N_UNIQUE_FRAMES = np.zeros(N, dtype=np.uint16)
    N_MISSING_FRAMES = np.zeros(N, dtype=np.uint16)
    MAX_FRAME = np.zeros(N, dtype=np.uint16)
    
    for idx, file_path in enumerate(tqdm(sample_files)):
        df = pd.read_parquet(file_path)
        N_UNIQUE_FRAMES[idx] = df['frame'].nunique()
        N_MISSING_FRAMES[idx] = (df['frame'].max() - df['frame'].min()) - df['frame'].nunique() + 1
        MAX_FRAME[idx] = df['frame'].max()
    
    # Print statistics
    print("\nNumber of unique frames in each video:")
    print(pd.Series(N_UNIQUE_FRAMES).describe(percentiles=percentiles).to_frame('N_UNIQUE_FRAMES'))
    
    print("\nNumber of missing frames:")
    print(pd.Series(N_MISSING_FRAMES).describe(percentiles=percentiles).to_frame('N_MISSING_FRAMES'))
    
    print("\nMaximum frame number:")
    print(pd.Series(MAX_FRAME).describe(percentiles=percentiles).to_frame('MAX_FRAME'))
    
    # Plot if requested
    if config.SHOW_PLOTS:
        fig, axes = plt.subplots(1, 3, figsize=(15, 8))
        
        # Unique frames
        pd.Series(N_UNIQUE_FRAMES).plot(kind='hist', bins=128, ax=axes[0])
        axes[0].set_title('Number of Unique Frames', size=24)
        axes[0].grid()
        
        # Missing frames
        pd.Series(N_MISSING_FRAMES).plot(kind='hist', bins=128, ax=axes[1])
        axes[1].set_title('Number of Missing Frames', size=24)
        axes[1].grid()
        
        # Max frame
        pd.Series(MAX_FRAME).plot(kind='hist', bins=128, ax=axes[2])
        axes[2].set_title('Maximum Frame Index', size=24)
        axes[2].grid()
        
        plt.tight_layout()
        plt.show()


def calculate_mean_std_stats(X_train: np.ndarray, landmarks: LandmarkIndices, 
                           config: Config) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Calculate mean and std statistics for lips, hands, and pose."""
    stats = {}
    
    # Percentage of frames filled
    NON_EMPTY_FRAME_IDXS_TRAIN = X_train[:, :, 0, 0]  # Dummy extraction
    P_DATA_FILLED = (NON_EMPTY_FRAME_IDXS_TRAIN != -1).sum() / NON_EMPTY_FRAME_IDXS_TRAIN.size * 100
    print(f'P_DATA_FILLED: {P_DATA_FILLED:.2f}%')
    
    # LIPS
    print("\nCalculating lips statistics...")
    LIPS_MEAN_X = np.zeros([len(landmarks.LIPS_IDXS)], dtype=np.float32)
    LIPS_MEAN_Y = np.zeros([len(landmarks.LIPS_IDXS)], dtype=np.float32)
    LIPS_STD_X = np.zeros([len(landmarks.LIPS_IDXS)], dtype=np.float32)
    LIPS_STD_Y = np.zeros([len(landmarks.LIPS_IDXS)], dtype=np.float32)
    
    lips_data = np.transpose(X_train[:, :, landmarks.LIPS_IDXS], [2, 3, 0, 1]).reshape(
        [len(landmarks.LIPS_IDXS), config.N_DIMS, -1]
    )
    
    for col, ll in enumerate(tqdm(lips_data)):
        for dim, l in enumerate(ll):
            v = l[np.nonzero(l)]
            if dim == 0:  # X
                LIPS_MEAN_X[col] = v.mean()
                LIPS_STD_X[col] = v.std()
            if dim == 1:  # Y
                LIPS_MEAN_Y[col] = v.mean()
                LIPS_STD_Y[col] = v.std()
    
    stats['lips'] = (
        np.array([LIPS_MEAN_X, LIPS_MEAN_Y]).T,
        np.array([LIPS_STD_X, LIPS_STD_Y]).T
    )
    
    # LEFT HAND
    print("\nCalculating left hand statistics...")
    LEFT_HANDS_MEAN_X = np.zeros([len(landmarks.LEFT_HAND_IDXS)], dtype=np.float32)
    LEFT_HANDS_MEAN_Y = np.zeros([len(landmarks.LEFT_HAND_IDXS)], dtype=np.float32)
    LEFT_HANDS_STD_X = np.zeros([len(landmarks.LEFT_HAND_IDXS)], dtype=np.float32)
    LEFT_HANDS_STD_Y = np.zeros([len(landmarks.LEFT_HAND_IDXS)], dtype=np.float32)
    
    left_hand_data = np.transpose(X_train[:, :, landmarks.LEFT_HAND_IDXS], [2, 3, 0, 1]).reshape(
        [len(landmarks.LEFT_HAND_IDXS), config.N_DIMS, -1]
    )
    
    for col, ll in enumerate(tqdm(left_hand_data)):
        for dim, l in enumerate(ll):
            v = l[np.nonzero(l)]
            if dim == 0:  # X
                LEFT_HANDS_MEAN_X[col] = v.mean()
                LEFT_HANDS_STD_X[col] = v.std()
            if dim == 1:  # Y
                LEFT_HANDS_MEAN_Y[col] = v.mean()
                LEFT_HANDS_STD_Y[col] = v.std()
    
    stats['left_hand'] = (
        np.array([LEFT_HANDS_MEAN_X, LEFT_HANDS_MEAN_Y]).T,
        np.array([LEFT_HANDS_STD_X, LEFT_HANDS_STD_Y]).T
    )
    
    # POSE
    print("\nCalculating pose statistics...")
    POSE_MEAN_X = np.zeros([len(landmarks.POSE_IDXS)], dtype=np.float32)
    POSE_MEAN_Y = np.zeros([len(landmarks.POSE_IDXS)], dtype=np.float32)
    POSE_STD_X = np.zeros([len(landmarks.POSE_IDXS)], dtype=np.float32)
    POSE_STD_Y = np.zeros([len(landmarks.POSE_IDXS)], dtype=np.float32)
    
    pose_data = np.transpose(X_train[:, :, landmarks.POSE_IDXS], [2, 3, 0, 1]).reshape(
        [len(landmarks.POSE_IDXS), config.N_DIMS, -1]
    )
    
    for col, ll in enumerate(tqdm(pose_data)):
        for dim, l in enumerate(ll):
            v = l[np.nonzero(l)]
            if dim == 0:  # X
                POSE_MEAN_X[col] = v.mean()
                POSE_STD_X[col] = v.std()
            if dim == 1:  # Y
                POSE_MEAN_Y[col] = v.mean()
                POSE_STD_Y[col] = v.std()
    
    stats['pose'] = (
        np.array([POSE_MEAN_X, POSE_MEAN_Y]).T,
        np.array([POSE_STD_X, POSE_STD_Y]).T
    )
    
    return stats


# ================================
# Batch Generator
# ================================
def get_train_batch_all_signs(X: np.ndarray, y: np.ndarray, 
                            NON_EMPTY_FRAME_IDXS: np.ndarray,
                            config: Config):
    """Custom sampler to get a batch containing N times all signs."""
    landmarks = LandmarkIndices()
    n = config.BATCH_ALL_SIGNS_N
    
    # Arrays to store batch in
    X_batch = np.zeros(
        [config.NUM_CLASSES * n, config.INPUT_SIZE, landmarks.N_COLS, config.N_DIMS], 
        dtype=np.float32
    )
    y_batch = np.arange(0, config.NUM_CLASSES, step=1/n, dtype=np.float32).astype(np.int64)
    non_empty_frame_idxs_batch = np.zeros(
        [config.NUM_CLASSES * n, config.INPUT_SIZE], 
        dtype=np.float32
    )
    
    # Dictionary mapping ordinally encoded sign to corresponding sample indices
    CLASS2IDXS = {}
    for i in range(config.NUM_CLASSES):
        CLASS2IDXS[i] = np.argwhere(y == i).squeeze().astype(np.int32)
            
    while True:
        # Fill batch arrays
        for i in range(config.NUM_CLASSES):
            idxs = np.random.choice(CLASS2IDXS[i], n)
            X_batch[i*n:(i+1)*n] = X[idxs]
            non_empty_frame_idxs_batch[i*n:(i+1)*n] = NON_EMPTY_FRAME_IDXS[idxs]
        
        yield {
            'frames': X_batch, 
            'non_empty_frame_idxs': non_empty_frame_idxs_batch
        }, y_batch


# ================================
# Main Pipeline
# ================================
def prepare_data(preprocess: bool = True,
                use_validation: bool = False,
                show_plots: bool = False) -> Dict:
    """Main data preparation pipeline."""
    config = Config()
    config.SHOW_PLOTS = show_plots
    landmarks = LandmarkIndices()
    
    # Load metadata
    print("Loading metadata...")
    train_df = pd.read_csv('train.csv')
    train_df['file_path'] = train_df['path'].apply(lambda x: f'./{x}')
    train_df['sign_ord'] = train_df['sign'].astype('category').cat.codes
    
    # Create translation dictionaries
    SIGN2ORD = train_df[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
    ORD2SIGN = train_df[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()
    
    print(f"Dataset: {len(train_df)} samples, {train_df['sign'].nunique()} unique signs")
    print(train_df.head())
    
    # Analyze statistics
    analyze_frame_statistics(train_df, config)
    
    # Process or load data
    if preprocess:
        print("\nProcessing data from scratch...")
        preprocess_layer = PreprocessLayer()
        X, NON_EMPTY_FRAME_IDXS, y = process_dataset(train_df, config, preprocess_layer)
        
        # Save processed data
        print("Saving processed data...")
        # Save as compressed files for memory efficiency
        save_compressed(X, 'X.zip')
        save_compressed(y, 'y.zip')
        save_compressed(NON_EMPTY_FRAME_IDXS, 'NON_EMPTY_FRAME_IDXS.zip')
        
        # Also save in original format for backward compatibility
        np.save('X.npy', X)
        np.save('y.npy', y)
        np.save('NON_EMPTY_FRAME_IDXS.npy', NON_EMPTY_FRAME_IDXS)
    else:
        print("\nLoading preprocessed data...")
        # Try compressed format first, fall back to original
        try:
            X = load_compressed('X.zip')
            y = load_compressed('y.zip')
            NON_EMPTY_FRAME_IDXS = load_compressed('NON_EMPTY_FRAME_IDXS.zip')
        except FileNotFoundError:
            X = np.load('X.npy')
            y = np.load('y.npy')
            NON_EMPTY_FRAME_IDXS = np.load('NON_EMPTY_FRAME_IDXS.npy')
    
    print_shape_dtype([X, y, NON_EMPTY_FRAME_IDXS], ['X', 'y', 'NON_EMPTY_FRAME_IDXS'])
    print(f"Memory usage: {(X.nbytes + y.nbytes + NON_EMPTY_FRAME_IDXS.nbytes) / 1e9:.2f} GB")
    
    # Split data
    if use_validation:
        print("\nSplitting data with validation set...")
        split_data = split_train_val(
            X, y, NON_EMPTY_FRAME_IDXS, 
            train_df['participant_id'].values,
            seed=config.SEED
        )
        
        # Save splits
        print("Saving train/validation splits...")
        for key in ['X_train', 'y_train', 'NON_EMPTY_FRAME_IDXS_TRAIN',
                    'X_val', 'y_val', 'NON_EMPTY_FRAME_IDXS_VAL']:
            np.save(f'{key}.npy', split_data[key])
            save_compressed(split_data[key], f'{key}.zip')
        
        print_shape_dtype(
            [split_data['X_train'], split_data['X_val']], 
            ['X_train', 'X_val']
        )
        
        result = split_data
    else:
        print("\nUsing all data for training...")
        result = {
            'X_train': X,
            'y_train': y,
            'NON_EMPTY_FRAME_IDXS_TRAIN': NON_EMPTY_FRAME_IDXS,
            'validation_data': None
        }
    
    # Calculate mean/std statistics
    print("\nCalculating landmark statistics...")
    stats = calculate_mean_std_stats(result['X_train'], landmarks, config)
    
    # Add metadata to result
    result.update({
        'train': train_df,
        'SIGN2ORD': SIGN2ORD,
        'ORD2SIGN': ORD2SIGN,
        'stats': stats,
        'config': config,
        'landmarks': landmarks
    })
    
    print("\nData preparation complete!")
    return result


# ================================
# Backward Compatibility Functions
# ================================
def preprocess_data():
    """Backward compatibility wrapper for original function."""
    prepare_data(preprocess=True, use_validation=False)


if __name__ == "__main__":
    # Test with example configuration
    data = prepare_data(
        preprocess=False,  # Load existing data
        use_validation=False,
        show_plots=False
    )
    
    # Test batch generator
    config = Config()
    batch_gen = get_train_batch_all_signs(
        data['X_train'], 
        data['y_train'], 
        data['NON_EMPTY_FRAME_IDXS_TRAIN'],
        config
    )
    X_batch, y_batch = next(batch_gen)
    print("\nBatch test:")
    print_shape_dtype(
        [X_batch['frames'], X_batch['non_empty_frame_idxs'], y_batch],
        ['X_batch["frames"]', 'X_batch["non_empty_frame_idxs"]', 'y_batch']
    )
    print(f"Unique classes in batch: {len(np.unique(y_batch))}")