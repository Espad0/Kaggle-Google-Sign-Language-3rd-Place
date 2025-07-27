import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import zipfile
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
import math
import gc

# MatplotLib Global Settings
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['axes.titlesize'] = 24

# Constants
N_ROWS = 543
N_DIMS = 3
DIM_NAMES = ['x', 'y', 'z']
SEED = 42
NUM_CLASSES = 250
INPUT_SIZE = 32
MASK_VAL = 4237

# Landmark indices
USE_TYPES = ['left_hand', 'pose', 'right_hand']
START_IDX = 468
LIPS_IDXS0 = np.array([
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    ])
# Landmark indices in original data
LEFT_HAND_IDXS0 = np.arange(468,489)
RIGHT_HAND_IDXS0 = np.arange(522,543)
LEFT_POSE_IDXS0 = np.array([502, 504, 506, 508, 510])
RIGHT_POSE_IDXS0 = np.array([503, 505, 507, 509, 511])
LANDMARK_IDXS_LEFT_DOMINANT0 = np.concatenate((LIPS_IDXS0, LEFT_HAND_IDXS0, LEFT_POSE_IDXS0))
LANDMARK_IDXS_RIGHT_DOMINANT0 = np.concatenate((LIPS_IDXS0, RIGHT_HAND_IDXS0, RIGHT_POSE_IDXS0))
HAND_IDXS0 = np.concatenate((LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0), axis=0)
N_COLS = LANDMARK_IDXS_LEFT_DOMINANT0.size
# Landmark indices in processed data
LIPS_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_LEFT_DOMINANT0, LIPS_IDXS0)).squeeze()
LEFT_HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_LEFT_DOMINANT0, LEFT_HAND_IDXS0)).squeeze()
RIGHT_HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_LEFT_DOMINANT0, RIGHT_HAND_IDXS0)).squeeze()
HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_LEFT_DOMINANT0, HAND_IDXS0)).squeeze()
POSE_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_LEFT_DOMINANT0, LEFT_POSE_IDXS0)).squeeze()

LIPS_START = 0
LEFT_HAND_START = LIPS_IDXS.size
RIGHT_HAND_START = LEFT_HAND_START + LEFT_HAND_IDXS.size
POSE_START = RIGHT_HAND_START + RIGHT_HAND_IDXS.size

# Source: https://www.kaggle.com/competitions/asl-signs/overview/evaluation
ROWS_PER_FRAME = 543  # number of landmarks per frame


def print_shape_dtype(l, names):
    """Prints Shape and Dtype For List Of Variables"""
    for e, n in zip(l, names):
        print(f'{n} shape: {e.shape}, dtype: {e.dtype}')


def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


class PreprocessLayer(tf.keras.layers.Layer):
    """Tensorflow layer to process data in TFLite
    Data needs to be processed in the model itself, so we can not use Python
    """ 
    def __init__(self):
        super(PreprocessLayer, self).__init__()
        normalisation_correction = tf.constant([
                    # Add 0.50 to left hand (original right hand) and substract 0.50 of right hand (original left hand)
                    [0] * len(LIPS_IDXS) + [0.50] * len(LEFT_HAND_IDXS) + [0.50] * len(POSE_IDXS),
                    # Y coordinates stay intact
                    [0] * len(LANDMARK_IDXS_LEFT_DOMINANT0),
                    # Z coordinates stay intact
                    [0] * len(LANDMARK_IDXS_LEFT_DOMINANT0),
                ],
                dtype=tf.float32,
            )
        self._normalisation_correction = tf.transpose(normalisation_correction, [1,0])
        self._landmark_idxs_left_dominant0 = tf.constant(LANDMARK_IDXS_LEFT_DOMINANT0, dtype=tf.int32)
        self._landmark_idxs_right_dominant0 = tf.constant(LANDMARK_IDXS_RIGHT_DOMINANT0, dtype=tf.int32)
        self._hand_idxs0 = tf.constant(HAND_IDXS0, dtype=tf.int32)
        self._n_hand_points = tf.constant(len(HAND_IDXS0), dtype=tf.int32)
        self._lips_idxs = tf.constant(LIPS_IDXS, dtype=tf.int32)
        self._hand_idxs = tf.constant(HAND_IDXS, dtype=tf.int32)
        self._pose_idxs = tf.constant(POSE_IDXS, dtype=tf.int32)
        
    @tf.function(
        input_signature=(tf.TensorSpec(shape=[None, ROWS_PER_FRAME, 3], dtype=tf.float32),),
    )
    def call(self, data):
        # Number of Frames in Video
        n_frames = tf.shape(data)[0]
        
        # Find dominant hand by comparing summed absolute coordinates
        left_hand_sum = tf.math.reduce_sum(tf.where(tf.math.is_nan(tf.gather(data, LEFT_HAND_IDXS0, axis=1)), 0, 1))
        right_hand_sum = tf.math.reduce_sum(tf.where(tf.math.is_nan(tf.gather(data, RIGHT_HAND_IDXS0, axis=1)), 0, 1))
        left_dominant = left_hand_sum >= right_hand_sum
        
        # Count non NaN Hand values in each frame for the dominant hand
        if left_dominant:
            frames_hands_non_nan_sum = tf.math.reduce_sum(
                    tf.where(
                        tf.math.is_nan(tf.gather(data, LEFT_HAND_IDXS0, axis=1)),
                        0,
                        1,
                    ),
                    axis=[1, 2],
                )
        else:
            frames_hands_non_nan_sum = tf.math.reduce_sum(
                    tf.where(
                        tf.math.is_nan(tf.gather(data, RIGHT_HAND_IDXS0, axis=1)),
                        0,
                        1,
                    ),
                    axis=[1, 2],
                )
        
        # Find frames indices with coordinates of dominant hand
        # If at least 60% of the 21 hand points have non-NaN values then keep it
        # The mean number of non-NaN values in the hand for all samples is 19 (out of 21)
        # 0.8*21 = 17 is still very close to the mean so it should keep the vast majority of useful frames
        frames_non_empty_idxs = tf.where(tf.cast(frames_hands_non_nan_sum, tf.float32) >= tf.cast(self._n_hand_points, tf.float32)*0.60)
        frames_non_empty_idxs = tf.squeeze(frames_non_empty_idxs, axis=1)
        # Filter data to only keep non-empty frames
        data = tf.gather(data, frames_non_empty_idxs, axis=0)
        n_frames = tf.shape(data)[0]
        
        # Cast Indices in float32 to be compatible with Tensorflow Lite
        frames_non_empty_idxs = tf.cast(frames_non_empty_idxs, tf.float32)
        # Normalize to start with 0
        frames_non_empty_idxs = frames_non_empty_idxs - tf.reduce_min(frames_non_empty_idxs)
        
        # Gather Relevant Landmark Columns
        if left_dominant:
            data = tf.gather(data, self._landmark_idxs_left_dominant0, axis=1)
        else:
            data = tf.gather(data, self._landmark_idxs_right_dominant0, axis=1)
        
        if not left_dominant:
            # Need to adjust coordinates for right dominant hand
            data = (
                self._normalisation_correction + (
                    tf.where(
                        tf.math.is_nan(data),
                        0.0,
                        data,
                    ) * tf.constant([-1,1,1], dtype=tf.float32)
                )
            )
        
        # Video fits in INPUT_SIZE
        if n_frames < INPUT_SIZE:
            # Pad With -1 to indicate padding
            frames_non_empty_idxs = tf.pad(frames_non_empty_idxs, [[0, INPUT_SIZE-n_frames]], constant_values=-1)
            # Pad Data With Zeros
            data = tf.pad(data, [[0, INPUT_SIZE-n_frames], [0,0], [0,0]], constant_values=0)
        # Video needs to be downsampled to INPUT_SIZE
        else:
            # Randomly downsample video frames without replacement
            # Lower probability for the first and last frames to prevent cutting out non-empty frames
            frame_probabilities = tf.concat([
                    tf.constant([0.05], dtype=tf.float32),
                    tf.fill([n_frames-2], tf.constant(0.95, dtype=tf.float32)),
                    tf.constant([0.05], dtype=tf.float32),
                ],
                axis=0,
            )
            frames_sampled_idxs = tf.reshape(
                tf.random.stateless_categorical(
                    tf.math.log([frame_probabilities]),
                    INPUT_SIZE,
                    (tf.cast(tf.math.reduce_sum(frames_non_empty_idxs), tf.int32), tf.cast(tf.math.reduce_max(frames_non_empty_idxs), tf.int32)),
                ),
                [INPUT_SIZE],
            )
            data = tf.gather(data, frames_sampled_idxs, axis=0)
            frames_non_empty_idxs = tf.gather(frames_non_empty_idxs, frames_sampled_idxs, axis=0)
        
        # If a frame contains at least 1 NaN coordinate after preprocessing then we replace it entirely with zeros
        frames_with_nan = tf.reduce_any(tf.reduce_any(tf.math.is_nan(data), axis=2), axis=1)
        data = tf.where(frames_with_nan[..., tf.newaxis, tf.newaxis], 0.0, data)
        frames_non_empty_idxs = tf.where(frames_with_nan, -1.0, frames_non_empty_idxs)
        
        # Normalise data
        # Mean
        data_mean = tf.math.reduce_mean(data, axis=[0,1], keepdims=True)
        data_mean = tf.where(tf.math.is_nan(data_mean), 0.0, data_mean)
        # Standard Deviation
        data_std = tf.math.reduce_std(data, axis=[0,1], keepdims=True)
        data_std = tf.where(tf.math.is_nan(data_std), 1.0, data_std)
        data_std = tf.where(data_std < 0.01, 1.0, data_std)
        
        data = (data - data_mean) / data_std
        
        # Fill NaN Values With 0
        data = tf.where(tf.math.is_nan(data), 0.0, data)
        
        # Clip Values to [-10,10] to fix outliers
        data = tf.clip_by_value(data, -10.0, 10.0)
        
        return data, frames_non_empty_idxs


def create_nan_samples():
    """Create samples with MASK_VAL value masked (NaN replaced with 0)"""
    samples_with_nan = np.zeros((10,INPUT_SIZE,N_COLS,N_DIMS), dtype=np.float32)
    samples_with_nan_frame_idxs = np.full((10,INPUT_SIZE), -1, dtype=np.float32)
    samples_with_nan[0,0:7,:,:] = MASK_VAL
    samples_with_nan[1,0:30,:,:] = MASK_VAL
    samples_with_nan[2,0:INPUT_SIZE,:,:] = MASK_VAL
    samples_with_nan[3,0:1,:,:] = MASK_VAL
    samples_with_nan[4,0:INPUT_SIZE-1,:,:] = MASK_VAL
    samples_with_nan[5,16:18,:,:] = MASK_VAL
    samples_with_nan[6,0:INPUT_SIZE:2,:,:] = MASK_VAL
    samples_with_nan[7,1:INPUT_SIZE:2,:,:] = MASK_VAL
    samples_with_nan[8,0:INPUT_SIZE:3,:,:] = MASK_VAL
    samples_with_nan[9,1:INPUT_SIZE:5,:,:] = MASK_VAL
    samples_with_nan_frame_idxs[0,0:7] = np.arange(7, dtype=np.float32)
    samples_with_nan_frame_idxs[1,0:30] = np.arange(30, dtype=np.float32)
    samples_with_nan_frame_idxs[2,0:INPUT_SIZE] = np.arange(INPUT_SIZE, dtype=np.float32)
    samples_with_nan_frame_idxs[3,0:1] = np.arange(1, dtype=np.float32)
    samples_with_nan_frame_idxs[4,0:INPUT_SIZE-1] = np.arange(INPUT_SIZE-1, dtype=np.float32)
    samples_with_nan_frame_idxs[5,16:18] = np.arange(16,18, dtype=np.float32)
    samples_with_nan_frame_idxs[6,0:INPUT_SIZE:2] = np.arange(0,INPUT_SIZE,2, dtype=np.float32)
    samples_with_nan_frame_idxs[7,1:INPUT_SIZE:2] = np.arange(1,INPUT_SIZE,2, dtype=np.float32)
    samples_with_nan_frame_idxs[8,0:INPUT_SIZE:3] = np.arange(0,INPUT_SIZE,3, dtype=np.float32)
    samples_with_nan_frame_idxs[9,1:INPUT_SIZE:5] = np.arange(1,INPUT_SIZE,5, dtype=np.float32)
    return samples_with_nan, samples_with_nan_frame_idxs


def preprocess_and_save_data(train, show_plots=False):
    """Process data from scratch and save to disk"""
    X = []
    NON_EMPTY_FRAME_IDXS = []
    y = []
    
    preprocess_layer = PreprocessLayer()
    
    # Process each data file
    for row_idx, (file_path, file_label_ord) in enumerate(tqdm(train[['file_path', 'sign_ord']].values)):
        data = load_relevant_data_subset(file_path)
        
        # Apply Preprocessing
        data, non_empty_frame_idxs = preprocess_layer(data)
        
        X.append(data)
        NON_EMPTY_FRAME_IDXS.append(non_empty_frame_idxs)
        y.append(file_label_ord)
    
    # Cast to numpy
    X = np.array(X, dtype=np.float32)
    NON_EMPTY_FRAME_IDXS = np.array(NON_EMPTY_FRAME_IDXS, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    # 10 GB
    # X = 100k samples x 64 frames x 66 landmarks x 3 coordinates x 4 bytes = 5 GB
    # NON_EMPTY_FRAME_IDXS = 100k samples x 64 frames x 4 bytes = 25 MB
    # y = 100k samples x 1 byte = 0.1 MB
    
    print_shape_dtype([X, NON_EMPTY_FRAME_IDXS, y], ['X', 'NON_EMPTY_FRAME_IDXS', 'y'])
    print(f'TOTAL MEMORY USAGE: {(X.nbytes + NON_EMPTY_FRAME_IDXS.nbytes + y.nbytes) / 1024**3:.2f} GB')
    
    # Save X
    with zipfile.ZipFile('X.zip', 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        np.save('X.npy', X)
        zf.write('X.npy')
        os.remove('X.npy')

    # Save NON_EMPTY_FRAME_IDXS
    with zipfile.ZipFile('NON_EMPTY_FRAME_IDXS.zip', 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        np.save('NON_EMPTY_FRAME_IDXS.npy', NON_EMPTY_FRAME_IDXS)
        zf.write('NON_EMPTY_FRAME_IDXS.npy')
        os.remove('NON_EMPTY_FRAME_IDXS.npy')

    # Save y
    with zipfile.ZipFile('y.zip', 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        np.save('y.npy', y)
        zf.write('y.npy')
        os.remove('y.npy')
    
    return X, NON_EMPTY_FRAME_IDXS, y


def load_preprocessed_data():
    """Load preprocessed data from disk"""
    # Read X
    with zipfile.ZipFile('X.zip', 'r', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.extractall()
    X = np.load('X.npy')
    os.remove('X.npy')

    # Read NON_EMPTY_FRAME_IDXS
    with zipfile.ZipFile('NON_EMPTY_FRAME_IDXS.zip', 'r', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.extractall()
    NON_EMPTY_FRAME_IDXS = np.load('NON_EMPTY_FRAME_IDXS.npy')
    os.remove('NON_EMPTY_FRAME_IDXS.npy')

    # Read y
    with zipfile.ZipFile('y.zip', 'r', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.extractall()
    y = np.load('y.npy')
    os.remove('y.npy')
    
    print_shape_dtype([X, NON_EMPTY_FRAME_IDXS, y], ['X', 'NON_EMPTY_FRAME_IDXS', 'y'])
    
    return X, NON_EMPTY_FRAME_IDXS, y


def split_data(X, y, NON_EMPTY_FRAME_IDXS, train, use_val=False):
    """Split data into train/validation sets"""
    samples_with_nan, samples_with_nan_frame_idxs = create_nan_samples()
    
    if use_val:
        # Split by participant_id so that the validation set contains new users not seen in the training data
        # A more robust model is produced if the model learns to generalize to new users
        PARTICIPANT_IDS = train['participant_id'].values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
        for train_index, val_index in gss.split(X, y, groups=PARTICIPANT_IDS):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            NON_EMPTY_FRAME_IDXS_TRAIN, NON_EMPTY_FRAME_IDXS_VAL = NON_EMPTY_FRAME_IDXS[train_index], NON_EMPTY_FRAME_IDXS[val_index]
            PARTICIPANT_IDS_TRAIN, PARTICIPANT_IDS_VAL = PARTICIPANT_IDS[train_index], PARTICIPANT_IDS[val_index]
        
        print_shape_dtype([X_train, X_val, y_train, y_val, NON_EMPTY_FRAME_IDXS_TRAIN, NON_EMPTY_FRAME_IDXS_VAL], 
                         ['X_train', 'X_val', 'y_train', 'y_val', 'NON_EMPTY_FRAME_IDXS_TRAIN', 'NON_EMPTY_FRAME_IDXS_VAL'])
        
        # Add NaN samples
        X_train = np.concatenate([X_train, samples_with_nan], axis=0)
        y_train = np.concatenate([y_train, np.zeros(len(samples_with_nan), dtype=np.int32)], axis=0)
        NON_EMPTY_FRAME_IDXS_TRAIN = np.concatenate([NON_EMPTY_FRAME_IDXS_TRAIN, samples_with_nan_frame_idxs], axis=0)
        
        validation_data = ({ 'frames': X_val, 'non_empty_frame_idxs': NON_EMPTY_FRAME_IDXS_VAL }, y_val)
        return X_train, y_train, NON_EMPTY_FRAME_IDXS_TRAIN, validation_data, y_val
    else:
        # Use all data for training
        X_train = X
        y_train = y
        NON_EMPTY_FRAME_IDXS_TRAIN = NON_EMPTY_FRAME_IDXS
        
        print_shape_dtype([X_train, y_train, NON_EMPTY_FRAME_IDXS_TRAIN], ['X_train', 'y_train', 'NON_EMPTY_FRAME_IDXS_TRAIN'])
        
        # Add NaN samples
        X_train = np.concatenate([X_train, samples_with_nan], axis=0)
        y_train = np.concatenate([y_train, np.zeros(len(samples_with_nan), dtype=np.int32)], axis=0)
        NON_EMPTY_FRAME_IDXS_TRAIN = np.concatenate([NON_EMPTY_FRAME_IDXS_TRAIN, samples_with_nan_frame_idxs], axis=0)
        
        validation_data = None
        return X_train, y_train, NON_EMPTY_FRAME_IDXS_TRAIN, validation_data, None


def analyze_data_statistics(train, is_interactive=True, show_plots=False):
    """Analyze frame statistics from the dataset"""
    N = int(1e3) if is_interactive else int(10e3)
    N_UNIQUE_FRAMES = np.zeros(N, dtype=np.uint16)
    N_MISSING_FRAMES = np.zeros(N, dtype=np.uint16)
    MAX_FRAME = np.zeros(N, dtype=np.uint16)
    
    PERCENTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999]
    
    for idx, file_path in enumerate(tqdm(train['file_path'].sample(N, random_state=SEED))):
        df = pd.read_parquet(file_path)
        N_UNIQUE_FRAMES[idx] = df['frame'].nunique()
        N_MISSING_FRAMES[idx] = (df['frame'].max() - df['frame'].min()) - df['frame'].nunique() + 1
        MAX_FRAME[idx] = df['frame'].max()
    
    # Number of unique frames in each video
    print(pd.Series(N_UNIQUE_FRAMES).describe(percentiles=PERCENTILES).to_frame('N_UNIQUE_FRAMES'))
    
    if show_plots:
        plt.figure(figsize=(15,8))
        plt.title('Number of Unique Frames', size=24)
        pd.Series(N_UNIQUE_FRAMES).plot(kind='hist', bins=128)
        plt.grid()
        xlim = math.ceil(plt.xlim()[1])
        plt.xlim(0, xlim)
        plt.xticks(np.arange(0, xlim+25, 25))
        plt.show()
    
    # Number of missing frames
    print(pd.Series(N_MISSING_FRAMES).describe(percentiles=PERCENTILES).to_frame('N_MISSING_FRAMES'))
    
    if show_plots:
        plt.figure(figsize=(15,8))
        plt.title('Number of Missing Frames', size=24)
        pd.Series(N_MISSING_FRAMES).plot(kind='hist', bins=128)
        plt.grid()
        plt.xlim(0, math.ceil(plt.xlim()[1]))
        plt.show()
    
    # Maximum frame number
    print(pd.Series(MAX_FRAME).describe(percentiles=PERCENTILES).to_frame('MAX_FRAME'))
    
    if show_plots:
        plt.figure(figsize=(15,8))
        plt.title('Maximum Frames Index', size=24)
        pd.Series(MAX_FRAME).plot(kind='hist', bins=128)
        plt.grid()
        plt.xlim(0, math.ceil(plt.xlim()[1]))
        plt.show()


def main():
    """Main preprocessing function"""
    # If True, processing data from scratch
    # If False, loads preprocessed data
    PREPROCESS_DATA = True
    IS_INTERACTIVE = True
    SHOW_PLOTS = False
    USE_VAL = False
    
    # Read Training Data
    train = pd.read_csv('train.csv')
    
    N_SAMPLES = len(train)
    print(f'N_SAMPLES: {N_SAMPLES}')
    
    # Get complete file path to file
    def get_file_path(path):
        return f'./{path}'
    
    train['file_path'] = train['path'].apply(get_file_path)
    
    # Add ordinally Encoded Sign (assign number to each sign name)
    train['sign_ord'] = train['sign'].astype('category').cat.codes
    
    # Dictionaries to translate sign <-> ordinal encoded sign
    SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
    ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()
    
    print(train.head(30))
    print(train.info())
    
    # Analyze data statistics
    analyze_data_statistics(train, IS_INTERACTIVE, SHOW_PLOTS)
    
    print(f'# HAND_IDXS: {len(HAND_IDXS)}, N_COLS: {N_COLS}')
    print(f'LIPS_START: {LIPS_START}, LEFT_HAND_START: {LEFT_HAND_START}, RIGHT_HAND_START: {RIGHT_HAND_START}, POSE_START: {POSE_START}')
    
    gc.collect()
    
    # Load or preprocess data
    if PREPROCESS_DATA:
        X, NON_EMPTY_FRAME_IDXS, y = preprocess_and_save_data(train, SHOW_PLOTS)
    else:
        X, NON_EMPTY_FRAME_IDXS, y = load_preprocessed_data()
    
    gc.collect()
    
    # Split data
    X_train, y_train, NON_EMPTY_FRAME_IDXS_TRAIN, validation_data, y_val = split_data(X, y, NON_EMPTY_FRAME_IDXS, train, USE_VAL)
    
    del X, y, NON_EMPTY_FRAME_IDXS
    gc.collect()
    
    print("Data preprocessing complete!")
    
    # Save some global variables for import by other modules
    return {
        'X_train': X_train,
        'y_train': y_train,
        'NON_EMPTY_FRAME_IDXS_TRAIN': NON_EMPTY_FRAME_IDXS_TRAIN,
        'validation_data': validation_data,
        'y_val': y_val,
        'SIGN2ORD': SIGN2ORD,
        'ORD2SIGN': ORD2SIGN,
        'train': train
    }


if __name__ == "__main__":
    main()