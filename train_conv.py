import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sn
import zipfile
import tflite_runtime.interpreter as tflite
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GroupShuffleSplit 

import glob
import sys
import math
import gc
import sys
import sklearn
import scipy
import time

print(f'Tensorflow V{tf.__version__}')
print(f'Python V{sys.version}')


# MatplotLib Global Settings
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['axes.titlesize'] = 24

# If True, processing data from scratch
# If False, loads preprocessed data
PREPROCESS_DATA = False  # Set to True to re-preprocess with INPUT_SIZE=32
TRAIN_MODEL = True
# True: use 10% of participants as validation set
# False: use all data for training -> gives better LB result
USE_VAL = False

N_ROWS = 543
N_DIMS = 3
DIM_NAMES = ['x', 'y', 'z']
SEED = 42
NUM_CLASSES = 250
IS_INTERACTIVE = True
VERBOSE = 1 if IS_INTERACTIVE else 2

INPUT_SIZE = 32  # Reduced from 64 to save memory

BATCH_ALL_SIGNS_N = 2  # Reduced from 4
BATCH_SIZE = 64  # Reduced from 256
N_EPOCHS = 100
LR_MAX = 1e-3
N_WARMUP_EPOCHS = 0
WD_RATIO = 0.05
MASK_VAL = 4237

# Visualization flag - set to True to show plots during training
SHOW_PLOTS = False

# Prints Shape and Dtype For List Of Variables
def print_shape_dtype(l, names):
    for e, n in zip(l, names):
        print(f'{n} shape: {e.shape}, dtype: {e.dtype}')


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


N = int(1e3) if (IS_INTERACTIVE or not PREPROCESS_DATA) else int(10e3)
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

if SHOW_PLOTS:
    plt.figure(figsize=(15,8))
    plt.title('Number of Unique Frames', size=24)
    pd.Series(N_UNIQUE_FRAMES).plot(kind='hist', bins=128)
    plt.grid()
    xlim = math.ceil(plt.xlim()[1])
    plt.xlim(0, xlim)
    plt.xticks(np.arange(0, xlim+25, 25))
    plt.show()

# Number of missing frames, consecutive frames with missing intermediate frame, i.e. 1,2,4,5 -> 3 is missing
print(pd.Series(N_MISSING_FRAMES).describe(percentiles=PERCENTILES).to_frame('N_MISSING_FRAMES'))

if SHOW_PLOTS:
    plt.figure(figsize=(15,8))
    plt.title('Number of Missing Frames', size=24)
    pd.Series(N_MISSING_FRAMES).plot(kind='hist', bins=128)
    plt.grid()
    plt.xlim(0, math.ceil(plt.xlim()[1]))
    plt.show()

# Maximum frame number
print(pd.Series(MAX_FRAME).describe(percentiles=PERCENTILES).to_frame('MAX_FRAME'))

if SHOW_PLOTS:
    plt.figure(figsize=(15,8))
    plt.title('Maximum Frames Index', size=24)
    pd.Series(MAX_FRAME).plot(kind='hist', bins=128)
    plt.grid()
    plt.xlim(0, math.ceil(plt.xlim()[1]))
    plt.show()


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

print(f'# HAND_IDXS: {len(HAND_IDXS)}, N_COLS: {N_COLS}')


LIPS_START = 0
LEFT_HAND_START = LIPS_IDXS.size
RIGHT_HAND_START = LEFT_HAND_START + LEFT_HAND_IDXS.size
POSE_START = RIGHT_HAND_START + RIGHT_HAND_IDXS.size

print(f'LIPS_START: {LIPS_START}, LEFT_HAND_START: {LEFT_HAND_START}, RIGHT_HAND_START: {RIGHT_HAND_START}, POSE_START: {POSE_START}')


# Source: https://www.kaggle.com/competitions/asl-signs/overview/evaluation
ROWS_PER_FRAME = 543  # number of landmarks per frame

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


"""
    Tensorflow layer to process data in TFLite
    Data needs to be processed in the model itself, so we can not use Python
""" 
class PreprocessLayer(tf.keras.layers.Layer):
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


preprocess_layer = None

gc.collect()

# Load processed dataset
if PREPROCESS_DATA:
    # Processing Data From Scratch
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
else:
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


gc.collect()

# Create samples with MASK_VAL value masked (NaN replaced with 0)
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

gc.collect()

# Train/Validation Split
if USE_VAL:
    # Split by participant_id so that the validation set contains new users not seen in the training data
    # A more robust model is produced if the model learns to generalize to new users
    PARTICIPANT_IDS = train['participant_id'].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
    for train_index, val_index in gss.split(X, y, groups=PARTICIPANT_IDS):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        NON_EMPTY_FRAME_IDXS_TRAIN, NON_EMPTY_FRAME_IDXS_VAL = NON_EMPTY_FRAME_IDXS[train_index], NON_EMPTY_FRAME_IDXS[val_index]
        PARTICIPANT_IDS_TRAIN, PARTICIPANT_IDS_VAL = PARTICIPANT_IDS[train_index], PARTICIPANT_IDS[val_index]
    
    print_shape_dtype([X_train, X_val, y_train, y_val, NON_EMPTY_FRAME_IDXS_TRAIN, NON_EMPTY_FRAME_IDXS_VAL], ['X_train', 'X_val', 'y_train', 'y_val', 'NON_EMPTY_FRAME_IDXS_TRAIN', 'NON_EMPTY_FRAME_IDXS_VAL'])
    
    # Add NaN samples
    X_train = np.concatenate([X_train, samples_with_nan], axis=0)
    y_train = np.concatenate([y_train, np.zeros(len(samples_with_nan), dtype=np.int32)], axis=0)
    NON_EMPTY_FRAME_IDXS_TRAIN = np.concatenate([NON_EMPTY_FRAME_IDXS_TRAIN, samples_with_nan_frame_idxs], axis=0)
    
    validation_data = ({ 'frames': X_val, 'non_empty_frame_idxs': NON_EMPTY_FRAME_IDXS_VAL }, y_val)
    del X_val, y_val, NON_EMPTY_FRAME_IDXS_VAL
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

del X, y, NON_EMPTY_FRAME_IDXS

gc.collect()

# Custom sampler to get a batch containing N times all signs
def get_train_batch_all_signs(X, y, NON_EMPTY_FRAME_IDXS, n=BATCH_ALL_SIGNS_N):
    # Arrays to store batch in
    X_batch = np.zeros([NUM_CLASSES*n, INPUT_SIZE, N_COLS, N_DIMS], dtype=np.float32)
    y_batch = np.arange(0, NUM_CLASSES, step=1/n, dtype=np.float32).astype(np.int64)
    non_empty_frame_idxs_batch = np.zeros([NUM_CLASSES*n, INPUT_SIZE], dtype=np.float32)
    
    # Dictionary mapping ordinally encoded sign to corresponding sample indices
    CLASS2IDXS = {}
    for i in range(NUM_CLASSES):
        CLASS2IDXS[i] = np.argwhere(y == i).squeeze().astype(np.int32)
            
    while True:
        # Fill batch arrays
        for i in range(NUM_CLASSES):
            idxs = np.random.choice(CLASS2IDXS[i], n)
            X_batch[i*n:(i+1)*n] = X[idxs]
            non_empty_frame_idxs_batch[i*n:(i+1)*n] = NON_EMPTY_FRAME_IDXS[idxs]
        
        yield { 'frames': X_batch, 'non_empty_frame_idxs': non_empty_frame_idxs_batch }, y_batch


# Get a batch
X_batch, y_batch = next(get_train_batch_all_signs(X_train, y_train, NON_EMPTY_FRAME_IDXS_TRAIN))
print_shape_dtype([X_batch['frames'], X_batch['non_empty_frame_idxs'], y_batch], ['X_batch["frames"]', 'X_batch["non_empty_frame_idxs"]', 'y_batch'])


# Sparse Categorical Cross Entropy With Label Smoothing
@tf.function
def scce_with_ls(y_true, y_pred):
    # One Hot Encode Sparsely Encoded Target Sign
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, NUM_CLASSES, axis=1)
    y_true = tf.squeeze(y_true, axis=2)
    # Categorical Crossentropy with native label smoothing support
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=0.25)


# Conv1D Model Architecture
def get_model():
    # Inputs
    frames = tf.keras.layers.Input([INPUT_SIZE, N_COLS, N_DIMS], dtype=tf.float32, name='frames')
    non_empty_frame_idxs = tf.keras.layers.Input([INPUT_SIZE], dtype=tf.float32, name='non_empty_frame_idxs')
    
    # Extract features from frames
    x = frames
    # Take only x,y coordinates (drop z)
    x = tf.slice(x, [0,0,0,0], [-1, INPUT_SIZE, N_COLS, 2])
    
    # Reshape for Conv1D: (batch, INPUT_SIZE, N_COLS*2)
    x = tf.reshape(x, [-1, INPUT_SIZE, N_COLS*2])
    
    # Conv1D Model Architecture as specified
    do = 0.5
    
    # First block
    x = tf.keras.layers.Conv1D(64, 1, strides=1, padding='valid', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.DepthwiseConv1D(3, strides=1, padding='valid', depth_multiplier=1, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Second block
    x = tf.keras.layers.Conv1D(64, 1, strides=1, padding='valid', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.DepthwiseConv1D(5, strides=2, padding='valid', depth_multiplier=4, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Pooling
    x = tf.keras.layers.MaxPool1D(2, 2)(x)
    
    # Third block
    x = tf.keras.layers.Conv1D(256, 1, strides=1, padding='valid', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.DepthwiseConv1D(3, strides=1, padding='valid', depth_multiplier=1, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Fourth block
    x = tf.keras.layers.Conv1D(256, 1, strides=1, padding='valid', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.DepthwiseConv1D(3, strides=2, padding='valid', depth_multiplier=4, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Global pooling
    x = tf.keras.layers.GlobalAvgPool1D()(x)
    x = tf.keras.layers.Dropout(rate=do)(x)
    
    # Dense layers
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(rate=do)(x)
    
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(rate=do)(x)
    
    # Output layer
    x = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    outputs = x
    
    # Create Tensorflow Model
    model = tf.keras.models.Model(inputs=[frames, non_empty_frame_idxs], outputs=outputs)
    
    # Sparse Categorical Cross Entropy With Label Smoothing
    loss = scce_with_ls
    
    # Adam Optimizer with weight decay
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5, clipnorm=1.0)
    
    # TopK Metrics
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top_10_acc'),
    ]
    
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
    return model


tf.keras.backend.clear_session()

model = get_model()


# Plot model summary
model.summary(expand_nested=True)


# tf.keras.utils.plot_model(model, show_shapes=True, show_dtype=True, show_layer_names=True, expand_nested=True, show_layer_activations=True)
# Note: Requires graphviz to be installed (sudo apt-get install graphviz)


if not PREPROCESS_DATA and TRAIN_MODEL:
    y_pred = model.predict_on_batch(X_batch).flatten()

    print(f'# NaN Values In Prediction: {np.isnan(y_pred).sum()}')


if not PREPROCESS_DATA and TRAIN_MODEL and SHOW_PLOTS:
    plt.figure(figsize=(12,5))
    plt.title(f'Softmax Output Initialized Model | µ={y_pred.mean():.3f}, σ={y_pred.std():.3f}', pad=25)
    pd.Series(y_pred).plot(kind='hist', bins=128, label='Class Probability')
    plt.xlim(0, max(y_pred) * 1.1)
    plt.vlines([1 / NUM_CLASSES], 0, plt.ylim()[1], color='red', label=f'Random Guessing Baseline 1/NUM_CLASSES={1 / NUM_CLASSES:.3f}')
    plt.grid()
    plt.legend()
    plt.show()


def lrfn(current_step, num_warmup_steps, lr_max, num_cycles=0.50, num_training_steps=N_EPOCHS):
    
    if current_step < num_warmup_steps:
        if WARMUP_METHOD == 'log':
            return lr_max * 0.10 ** (num_warmup_steps - current_step)
        else:
            return lr_max * 2 ** -(num_warmup_steps - current_step)
    else:
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr_max


def plot_lr_schedule(lr_schedule, epochs):
    if not SHOW_PLOTS:
        return
    fig = plt.figure(figsize=(20, 10))
    plt.plot([None] + lr_schedule + [None])
    # X Labels
    x = np.arange(1, epochs + 1)
    x_axis_labels = [i if epochs <= 40 or i % 5 == 0 or i == 1 else None for i in range(1, epochs + 1)]
    plt.xlim([1, epochs])
    plt.xticks(x, x_axis_labels) # set tick step to 1 and let x axis start at 1
    
    # Increase y-limit for better readability
    plt.ylim([0, max(lr_schedule) * 1.1])
    
    # Title
    schedule_info = f'start: {lr_schedule[0]:.1E}, max: {max(lr_schedule):.1E}, final: {lr_schedule[-1]:.1E}'
    plt.title(f'Step Learning Rate Schedule, {schedule_info}', size=18, pad=12)
    
    # Plot Learning Rates
    for x, val in enumerate(lr_schedule):
        if epochs <= 40 or x % 5 == 0 or x is epochs - 1:
            if x < len(lr_schedule) - 1:
                if lr_schedule[x - 1] < val:
                    ha = 'right'
                else:
                    ha = 'left'
            elif x == 0:
                ha = 'right'
            else:
                ha = 'left'
            plt.plot(x + 1, val, 'o', color='black');
            offset_y = (max(lr_schedule) - min(lr_schedule)) * 0.02
            plt.annotate(f'{val:.1E}', xy=(x + 1, val + offset_y), size=12, ha=ha)
    
    plt.xlabel('Epoch', size=16, labelpad=5)
    plt.ylabel('Learning Rate', size=16, labelpad=5)
    plt.grid()
    plt.show()

# Add missing WARMUP_METHOD definition
WARMUP_METHOD = 'log'

# Learning rate for encoder
LR_SCHEDULE = [lrfn(step, num_warmup_steps=N_WARMUP_EPOCHS, lr_max=LR_MAX, num_cycles=0.50) for step in range(N_EPOCHS)]
# Plot Learning Rate Schedule
if SHOW_PLOTS:
    plot_lr_schedule(LR_SCHEDULE, epochs=N_EPOCHS)
# Learning Rate Callback
lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda step: LR_SCHEDULE[step], verbose=1)


# Custom callback to update weight decay with learning rate
class WeightDecayCallback(tf.keras.callbacks.Callback):
    def __init__(self, wd_ratio=WD_RATIO):
        self.step_counter = 0
        self.wd_ratio = wd_ratio
    
    def on_epoch_begin(self, epoch, logs=None):
        model.optimizer.weight_decay = model.optimizer.learning_rate * self.wd_ratio
        print(f'learning rate: {model.optimizer.learning_rate.numpy():.2e}, weight decay: {model.optimizer.weight_decay.numpy():.2e}')




if TRAIN_MODEL:
    # Verify model prediction is <<<100ms
    start_time = time.time()
    for _ in range(100):
        model.predict_on_batch({ 'frames': X_train[:1], 'non_empty_frame_idxs': NON_EMPTY_FRAME_IDXS_TRAIN[:1] })
    end_time = time.time()
    avg_time = (end_time - start_time) / 100
    print(f'Average prediction time: {avg_time*1000:.2f}ms')


if USE_VAL:
    # Verify Validation Dataset Covers All Signs
    print(f'# Unique Signs in Validation Set: {pd.Series(y_val).nunique()}')
    # Value Counts
    print(pd.Series(y_val).value_counts().to_frame('Count').iloc[[1,2,3,-3,-2,-1]])


# Sanity Check
if TRAIN_MODEL and USE_VAL:
    _ = model.evaluate(*validation_data, verbose=2)


if TRAIN_MODEL:
    # Clear all models in GPU
    tf.keras.backend.clear_session()

    # Get new fresh model
    model = get_model()
    
    # Sanity Check
    model.summary()

    # Actual Training
    history = model.fit(
            x=get_train_batch_all_signs(X_train, y_train, NON_EMPTY_FRAME_IDXS_TRAIN),
            steps_per_epoch=len(X_train) // (NUM_CLASSES * BATCH_ALL_SIGNS_N),
            epochs=N_EPOCHS,
            # Only used for validation data since training data is a generator
            batch_size=BATCH_SIZE,
            validation_data=validation_data,
            callbacks=[
                lr_callback,
                WeightDecayCallback(),
            ],
            verbose = VERBOSE,
        )


# Save Model Weights
model.save_weights('model_conv.h5')


if USE_VAL:
    # Validation Predictions
    y_val_pred = model.predict({ 'frames': X_val, 'non_empty_frame_idxs': NON_EMPTY_FRAME_IDXS_VAL }, verbose=2).argmax(axis=1)
    # Label
    labels = [ORD2SIGN.get(i).replace(' ', '_') for i in range(NUM_CLASSES)]


def print_classification_report():
    # Classification report for all signs
    classification_report = sklearn.metrics.classification_report(
            y_val,
            y_val_pred,
            target_names=labels,
            output_dict=True,
        )
    # Round Data for better readability
    classification_report = pd.DataFrame(classification_report).T
    classification_report = classification_report.round(2)
    classification_report = classification_report.astype({
            'support': np.uint16,
        })
    # Add signs
    classification_report['sign'] = [e if e in SIGN2ORD else -1 for e in classification_report.index]
    classification_report['sign_ord'] = classification_report['sign'].apply(SIGN2ORD.get).fillna(-1).astype(np.int16)
    # Sort on F1-score
    classification_report = pd.concat((
        classification_report.head(NUM_CLASSES).sort_values('f1-score', ascending=False),
        classification_report.tail(3),
    ))

    pd.options.display.max_rows = 999
    print(classification_report)

if USE_VAL:
    print_classification_report()


def plot_history_metric(metric, f_best=np.argmax, ylim=None, yscale=None, yticks=None):
    if not SHOW_PLOTS:
        return
    plt.figure(figsize=(20, 10))
    
    values = history.history[metric]
    N_EPOCHS = len(values)
    val = 'val' in ''.join(history.history.keys())
    # Epoch Ticks
    if N_EPOCHS <= 20:
        x = np.arange(1, N_EPOCHS + 1)
    else:
        x = [1, 5] + [10 + 5 * idx for idx in range((N_EPOCHS - 10) // 5 + 1)]

    x_ticks = np.arange(1, N_EPOCHS+1)

    # Validation
    if val:
        val_values = history.history[f'val_{metric}']
        val_argmin = f_best(val_values)
        plt.plot(x_ticks, val_values, label=f'val')

    # summarize history for accuracy
    plt.plot(x_ticks, values, label=f'train')
    argmin = f_best(values)
    plt.scatter(argmin + 1, values[argmin], color='red', s=75, marker='o', label=f'train_best')
    if val:
        plt.scatter(val_argmin + 1, val_values[val_argmin], color='purple', s=75, marker='o', label=f'val_best')

    plt.title(f'Model {metric}', fontsize=24, pad=10)
    plt.ylabel(metric, fontsize=20, labelpad=10)

    if ylim:
        plt.ylim(ylim)

    if yscale is not None:
        plt.yscale(yscale)
        
    if yticks is not None:
        plt.yticks(yticks, fontsize=16)

    plt.xlabel('epoch', fontsize=20, labelpad=10)        
    plt.tick_params(axis='x', labelsize=8)
    plt.xticks(x, fontsize=16) # set tick step to 1 and let x axis start at 1
    plt.yticks(fontsize=16)
    
    plt.legend(prop={'size': 10})
    plt.grid()
    plt.show()


if TRAIN_MODEL and SHOW_PLOTS:
    plot_history_metric('loss', f_best=np.argmin)


if TRAIN_MODEL and SHOW_PLOTS:
    plot_history_metric('acc', ylim=[0,1], yticks=np.arange(0.0, 1.1, 0.1))

if TRAIN_MODEL and SHOW_PLOTS:
    plot_history_metric('top_5_acc', ylim=[0,1], yticks=np.arange(0.0, 1.1, 0.1))

if TRAIN_MODEL and SHOW_PLOTS:
    plot_history_metric('top_10_acc', ylim=[0,1], yticks=np.arange(0.0, 1.1, 0.1))



### SUBMISSION ###


# TFLite model for submission
class TFLiteModel(tf.Module):
    def __init__(self, model):
        super(TFLiteModel, self).__init__()

        # Load the feature generation and main models
        global preprocess_layer
        if preprocess_layer is None:
            print("PREPARING PREPROCESS LAYER FOR TFLITE MODEL") 
            preprocess_layer = PreprocessLayer()
        self.preprocess_layer = preprocess_layer
        self.model = model
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, N_ROWS, N_DIMS], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs):
        # Preprocess Data
        x, non_empty_frame_idxs = self.preprocess_layer(inputs)
        # Add Batch Dimension
        x = tf.expand_dims(x, axis=0)
        non_empty_frame_idxs = tf.expand_dims(non_empty_frame_idxs, axis=0)
        # Make Prediction
        outputs = self.model({ 'frames': x, 'non_empty_frame_idxs': non_empty_frame_idxs })
        # Squeeze Output 1x250 -> 250
        outputs = tf.squeeze(outputs, axis=0)

        # Return a dictionary with the output tensor
        return {'outputs': outputs}


def convert_tflite(tflite_model, tflite_path):
    # Concatenate all data for calibration dataset
    calibration_data = np.concatenate([
        X_train[:1000],
        np.zeros([10, INPUT_SIZE, N_COLS, 3], dtype=np.float32),
        np.ones([10, INPUT_SIZE, N_COLS, 3], dtype=np.float32),
    ])

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_concrete_functions([tflite_model.__call__.get_concrete_function()])

    # TFLite requires a batch axis: (543,3) -> (1,543,3). Model outputs shape (250,), it must output shape (1, 250) for inference
    def representative_data_gen():
        for frame_data in calibration_data:
            yield [frame_data]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # TFLite Model requires dynamic range quantization with int8/uint8 activations and int8 weights
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    # The inference is broken if we don't restrict all intermediate operations to int8
    converter._experimental_disable_per_channel = True
    tflite_model_ser = converter.convert()

    # Check size
    model_size_mb = len(tflite_model_ser) / (1024 * 1024)
    print(f'Model size: {model_size_mb:.2f} MB')

    # Save the model
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model_ser)


# Initialize model
keras_model = get_model()
keras_model.load_weights('model_conv.h5')
# Wrap the keras model with preprocessing using TFLiteModel
tflite_model = TFLiteModel(keras_model)

# Convert
convert_tflite(tflite_model, 'model_conv.tflite')


# Verify TFLite model was exported correctly
# Load the model
interpreter = tflite.Interpreter('model_conv.tflite')

# List of found signatures
found_signatures = list(interpreter.get_signature_list().keys())

# Serving signature
prediction_fn = interpreter.get_signature_runner("serving_default")

# Example prediction
output = prediction_fn(inputs=X_train[0])
print('outputs' in output)


# Verify output matches (there will be some difference due to quantization)
demo_data = tf.constant(X_train[0:1].astype(np.float32))
keras_output = tflite_model(demo_data)['outputs'].numpy()
tflite_output = prediction_fn(inputs=demo_data[0])['outputs']

print(f'Keras: {keras_output.shape}, {keras_output.argmax()}, {keras_output.max():.3f}')
print(f'TFLite: {tflite_output.shape}, {tflite_output.argmax()}, {tflite_output.max():.3f}')
print(f'DIFFERENCE: {np.abs(keras_output - tflite_output).max():.3f}')

plt.figure(figsize=(20,4))
plt.plot(keras_output, alpha=0.50, label='keras', linewidth=4)
plt.plot(tflite_output, alpha=0.75, label='tflite', linewidth=2)
plt.vlines(x=[keras_output.argmax(), tflite_output.argmax()], ymin=0, ymax=1, color='red', alpha=0.25, linewidth=5)
plt.legend()
plt.show()


print("Conv1D model training complete!")