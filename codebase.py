import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pandas as pd
import tensorflow as tf
# import tensorflow_addons as tfa  # No longer needed
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sn

from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split, GroupShuffleSplit 
import tensorflow.keras.backend as K
import glob
import sys
import os
import math
import gc
import sys
import sklearn
import scipy

# If True, processing data from scratch
# If False, loads preprocessed data
PREPROCESS_DATA = False
TRAIN_MODEL = True

N_FOLDS = 5
N_ROWS = 543
N_DIMS = 3
DIM_NAMES = ['x', 'y', 'z']
SEED = 42
NUM_CLASSES = 250
VERBOSE = 1

INPUT_SIZE = 32

NUM_HEADS = 8

BATCH_ALL_SIGNS_N = 4
BATCH_SIZE = 256
N_EPOCHS = 50
LR_MAX = 1e-3
N_WARMUP_EPOCHS = 0
WD_RATIO = 0.05
MASK_VAL = 4237

# Label smoothing for loss function
LABEL_SMOOTHING = 0.0

train = pd.read_csv('train.csv')

N_SAMPLES = len(train)
print(f'N_SAMPLES: {N_SAMPLES}')

# Prints Shape and Dtype For List Of Variables
def print_shape_dtype(l, names):
    for e, n in zip(l, names):
        print(f'{n} shape: {e.shape}, dtype: {e.dtype}')
        
def get_file_path(path):
    return f'/kaggle/input/asl-signs/{path}'

train['file_path'] = train['path'].apply(get_file_path)

# Ordinal Encoding: Convert sign names to numerical codes for ML model training
# Each unique sign name gets a unique integer (0, 1, 2, ...) based on alphabetical order
# This creates a numerical representation that the model can process

# Add ordinally Encoded Sign (assign number to each sign name)
train['sign_ord'] = train['sign'].astype('category').cat.codes

# Dictionaries to translate sign <-> ordinal encoded sign
SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

# Source: https://www.kaggle.com/competitions/asl-signs/overview/evaluation
ROWS_PER_FRAME = 543  # number of landmarks per frame

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

    
class PreprocessLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(PreprocessLayer, self).__init__()
        
    def pad_edge(self, t, repeats, side):
        if side == 'LEFT':
            return tf.concat((tf.repeat(t[:1], repeats=repeats, axis=0), t), axis=0)
        elif side == 'RIGHT':
            return tf.concat((t, tf.repeat(t[-1:], repeats=repeats, axis=0)), axis=0)
    
    @tf.function(
        input_signature=(tf.TensorSpec(shape=[None,N_ROWS,N_DIMS], dtype=tf.float32),),
    )
    def call(self, data0):
        # Number of Frames in Video
        N_FRAMES0 = tf.shape(data0)[0]
        # Count non NaN Hand values in each frame
        frames_hands_non_nan_sum = tf.math.reduce_sum(
                tf.cast(tf.math.is_nan(tf.gather(data0, HAND_IDXS0, axis=1)) == False, tf.int32),
                axis=[1, 2],
            )
        # Get indices of frames with at least 1 non NaN Hand Measurement
        non_empty_frames_idxs = tf.where(frames_hands_non_nan_sum > 0)
        non_empty_frames_idxs = tf.squeeze(non_empty_frames_idxs, axis=1)
        # Gather all frame indices with at least 1 non NaN Hand Measurement
        data = tf.gather(data0, non_empty_frames_idxs, axis=0)
        
        # Cast Indices in float32 to be compatible with Tensorflow Lite
        non_empty_frames_idxs = tf.cast(non_empty_frames_idxs, tf.float32) 

        
        # Number of Frames in Filtered Video
        N_FRAMES = tf.shape(data)[0]
        
        # Gather Relevant Landmark Columns
        data = tf.gather(data, LANDMARK_IDXS0, axis=1)
        
        # Video fits in INPUT_SIZE
        if N_FRAMES < INPUT_SIZE:
            # Pad With -1 to indicate padding
            non_empty_frames_idxs = tf.pad(non_empty_frames_idxs, [[0, INPUT_SIZE-N_FRAMES]], constant_values=-1)
            # Pad Data With Zeros
            data = tf.pad(data, [[0, INPUT_SIZE-N_FRAMES], [0,0], [0,0]], constant_values=0)
            
            data = tf.where(tf.math.equal(data,0.0), np.nan, data)
            xyz_ref = tf.gather(data, center_idx, axis=-2)
            xyz_ref = tf.reshape(xyz_ref, [-1, N_DIMS])
            center_mean = tf.experimental.numpy.nanmean(xyz_ref, axis=0, keepdims=True)
            center_std = tf.experimental.numpy.sqrt(tf.experimental.numpy.nanmean((xyz_ref - center_mean)**2, axis=0, keepdims=True))
            center_std = tf.math.reduce_mean(tf.gather(center_std, [0,1], axis=-1), axis=1, keepdims=True)
            center_mean = tf.reshape(tf.cast(center_mean, tf.float32), [1,1,3])
            center_std = tf.reshape(tf.cast(center_std, tf.float32), [1,1,1])
            data_normalized = (data - center_mean) / center_std
            
            # Fill NaN Values With 0
            data_normalized = tf.where(tf.math.is_nan(data_normalized), 0.0, data_normalized)
            
            
            return data_normalized, non_empty_frames_idxs
        # Video needs to be downsampled to INPUT_SIZE
        else:
            # Repeat
            if N_FRAMES < INPUT_SIZE**2:
                repeats = tf.math.floordiv(INPUT_SIZE * INPUT_SIZE, N_FRAMES0)
                data = tf.repeat(data, repeats=repeats, axis=0)
                non_empty_frames_idxs = tf.repeat(non_empty_frames_idxs, repeats=repeats, axis=0)

            # Pad To Multiple Of Input Size
            pool_size = tf.math.floordiv(len(data), INPUT_SIZE)
            if tf.math.mod(len(data), INPUT_SIZE) > 0:
                pool_size += 1

            if pool_size == 1:
                pad_size = (pool_size * INPUT_SIZE) - len(data)
            else:
                pad_size = (pool_size * INPUT_SIZE) % len(data)

            # Pad Start/End with Start/End value
            pad_left = tf.math.floordiv(pad_size, 2) + tf.math.floordiv(INPUT_SIZE, 2)
            pad_right = tf.math.floordiv(pad_size, 2) + tf.math.floordiv(INPUT_SIZE, 2)
            if tf.math.mod(pad_size, 2) > 0:
                pad_right += 1

            # Pad By Concatenating Left/Right Edge Values
            data = self.pad_edge(data, pad_left, 'LEFT')
            data = self.pad_edge(data, pad_right, 'RIGHT')

            # Pad Non Empty Frame Indices
            non_empty_frames_idxs = self.pad_edge(non_empty_frames_idxs, pad_left, 'LEFT')
            non_empty_frames_idxs = self.pad_edge(non_empty_frames_idxs, pad_right, 'RIGHT')

            # # better normalization
            # data = data - tf.experimental.numpy.nanmean(data, axis=-2)

            # Reshape to Mean Pool
            data = tf.reshape(data, [INPUT_SIZE, -1, N_COLS, N_DIMS])
            non_empty_frames_idxs = tf.reshape(non_empty_frames_idxs, [INPUT_SIZE, -1])

            # Mean Pool
            data = tf.experimental.numpy.nanmean(data, axis=1)
            non_empty_frames_idxs = tf.experimental.numpy.nanmean(non_empty_frames_idxs, axis=1)


            data = tf.where(tf.math.equal(data,0.0), np.nan, data)
            xyz_ref = tf.gather(data, center_idx, axis=-2)
            xyz_ref = tf.reshape(xyz_ref, [-1, N_DIMS])
            center_mean = tf.experimental.numpy.nanmean(xyz_ref, axis=0, keepdims=True)
            center_std = tf.experimental.numpy.sqrt(tf.experimental.numpy.nanmean((xyz_ref - center_mean)**2, axis=0, keepdims=True))
            center_std = tf.math.reduce_mean(tf.gather(center_std, [0,1], axis=-1), axis=1, keepdims=True)
            center_mean = tf.reshape(tf.cast(center_mean, tf.float32), [1,1,3])
            center_std = tf.reshape(tf.cast(center_std, tf.float32), [1,1,1])
            data_normalized = (data - center_mean) / center_std
            
            # Fill NaN Values With 0
            data_normalized = tf.where(tf.math.is_nan(data_normalized), 0.0, data_normalized)
            
            return data_normalized, non_empty_frames_idxs
    
preprocess_layer = PreprocessLayer()


def get_data(file_path):
    # Load Raw Data
    data = load_relevant_data_subset(file_path)
    # Process Data Using Tensorflow
    data = preprocess_layer(data)
    
    return data


REYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    246, 161, 160, 159, 158, 157, 173,
]
LEYE = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    466, 388, 387, 386, 385, 384, 398,
]
NOSE=[
    1,2,98,327
]
SLIP = [
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        191, 80, 81, 82, 13, 312, 311, 310, 415,
        ]


LIPS_IDXS0 = np.array(REYE+LEYE+NOSE+SLIP) #
LEFT_HAND_IDXS0 = np.arange(468,489) # 21
RIGHT_HAND_IDXS0 = np.arange(522,543) # 21
POSE_IDXS0 = np.array([11,13,15,12,14,16,23,24,])+489 # 8

LANDMARK_IDXS0 = np.concatenate((LIPS_IDXS0, LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0, POSE_IDXS0))
HAND_IDXS0 = np.concatenate((LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0), axis=0)
N_COLS = LANDMARK_IDXS0.size
# Landmark indices in processed data
LIPS_IDXS = np.argwhere(np.isin(LANDMARK_IDXS0, LIPS_IDXS0)).squeeze()
LEFT_HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS0, LEFT_HAND_IDXS0)).squeeze()
RIGHT_HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS0, RIGHT_HAND_IDXS0)).squeeze()
HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS0, HAND_IDXS0)).squeeze()
POSE_IDXS = np.argwhere(np.isin(LANDMARK_IDXS0, POSE_IDXS0)).squeeze()

print(f'# HAND_IDXS: {len(HAND_IDXS)}, N_COLS: {N_COLS}')

LIPS_START = 0
LEFT_HAND_START = LIPS_IDXS.size
RIGHT_HAND_START = LEFT_HAND_START + LEFT_HAND_IDXS.size
POSE_START = RIGHT_HAND_START + RIGHT_HAND_IDXS.size

print(f'LIPS_START: {LIPS_START}, LEFT_HAND_START: {LEFT_HAND_START}, RIGHT_HAND_START: {RIGHT_HAND_START}, POSE_START: {POSE_START}')

leye_center_idx = np.arange(len(LEYE))
reye_center_idx = np.arange(len(REYE)) + len(REYE)
nose_center_idx = np.arange(len(NOSE)) + len(REYE) + len(LEYE)
lip_center_idx = np.arange(len(SLIP)) + len(REYE)+len(LEYE)+len(NOSE)
pose_center_idx = np.array([0,1,6,7]) + POSE_START
center_idx = np.array(leye_center_idx.tolist()+reye_center_idx.tolist()+nose_center_idx.tolist()+lip_center_idx.tolist()+pose_center_idx.tolist())




# Epsilon value for layer normalisation
LAYER_NORM_EPS = 1e-6

# Dense layer units for landmarks
MOTION_UNITS = 128
LIPS_UNITS = 256
HANDS_UNITS = 256
POSE_UNITS = 256
# final embedding and transformer embedding size
UNITS = 256
XYZ_UNITS = 384
# Transformer
NUM_BLOCKS = 2
MLP_RATIO = 2

# Dropout
EMBEDDING_DROPOUT = 0.00
MLP_DROPOUT_RATIO = 0.30
CLASSIFIER_DROPOUT_RATIO = 0.10

# Initiailizers
INIT_HE_UNIFORM = tf.keras.initializers.he_uniform
INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform
INIT_ZEROS = tf.keras.initializers.constant(0.0)
# Activations
GELU = tf.keras.activations.gelu



# %% [code] {"execution":{"iopub.status.busy":"2023-03-24T17:24:03.552905Z","iopub.execute_input":"2023-03-24T17:24:03.553892Z","iopub.status.idle":"2023-03-24T17:24:03.560104Z","shell.execute_reply.started":"2023-03-24T17:24:03.553842Z","shell.execute_reply":"2023-03-24T17:24:03.558879Z"}}
def loss_fn(y_true, y_pred):
    loss = tf.keras.losses.categorical_crossentropy(tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])[:,0,:], y_pred, label_smoothing=LABEL_SMOOTHING)
    loss = tf.reduce_mean(loss)
    return loss

# %% [code] {"execution":{"iopub.status.busy":"2023-03-24T17:24:04.095699Z","iopub.execute_input":"2023-03-24T17:24:04.096388Z","iopub.status.idle":"2023-03-24T17:24:04.113226Z","shell.execute_reply.started":"2023-03-24T17:24:04.096352Z","shell.execute_reply":"2023-03-24T17:24:04.112053Z"}}
def get_model():

    def scaled_dot_product(q,k,v, softmax, attention_mask):
        #calculates Q . K(transpose)
        qkt = tf.matmul(q,k,transpose_b=True)
        #caculates scaling factor
        dk = tf.math.sqrt(tf.cast(q.shape[-1],dtype=tf.float32))
        scaled_qkt = qkt/dk
        softmax = softmax(scaled_qkt, mask=attention_mask)

        z = tf.matmul(softmax,v)
        #shape: (m,Tx,depth), same shape as q,k,v
        return z

    class MultiHeadAttention(tf.keras.layers.Layer):
        def __init__(self,d_model,num_of_heads):
            super(MultiHeadAttention,self).__init__()
            self.d_model = d_model
            self.num_of_heads = num_of_heads
            self.depth = d_model//num_of_heads
            self.wq = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
            self.wk = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
            self.wv = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
            self.wo = tf.keras.layers.Dense(d_model)
            self.softmax = tf.keras.layers.Softmax()

        def call(self,x, attention_mask):

            multi_attn = []
            for i in range(self.num_of_heads):
                Q = self.wq[i](x)
                K = self.wk[i](x)
                V = self.wv[i](x)
                multi_attn.append(scaled_dot_product(Q,K,V, self.softmax, attention_mask))

            multi_head = tf.concat(multi_attn,axis=-1)
            multi_head_attention = self.wo(multi_head)
            return multi_head_attention

    # %% [code] {"execution":{"iopub.status.busy":"2023-03-24T17:24:01.886612Z","iopub.execute_input":"2023-03-24T17:24:01.886981Z","iopub.status.idle":"2023-03-24T17:24:01.899889Z","shell.execute_reply.started":"2023-03-24T17:24:01.886946Z","shell.execute_reply":"2023-03-24T17:24:01.898849Z"}}
    # Full Transformer
    class Transformer(tf.keras.Model):
        def __init__(self, num_blocks):
            super(Transformer, self).__init__(name='transformer')
            self.num_blocks = num_blocks

        def build(self, input_shape):
            self.ln_1s = []
            self.mhas = []
            self.ln_2s = []
            self.mlps = []
            # Make Transformer Blocks
            for i in range(self.num_blocks):
                # First Layer Normalisation
                self.ln_1s.append(tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS))
                # Multi Head Attention
                self.mhas.append(MultiHeadAttention(UNITS, NUM_HEADS))
                # Second Layer Normalisation
                self.ln_2s.append(tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS))
                # Multi Layer Perception
                self.mlps.append(tf.keras.Sequential([
                    tf.keras.layers.Dense(384 * MLP_RATIO, activation=GELU, kernel_initializer=INIT_GLOROT_UNIFORM),
                    LateDropout(MLP_DROPOUT_RATIO),
                    tf.keras.layers.Dense(UNITS, kernel_initializer=INIT_HE_UNIFORM),
                ]))

        def call(self, x, attention_mask):
            # Iterate input over transformer blocks
            for ln_1, mha, ln_2, mlp in zip(self.ln_1s, self.mhas, self.ln_2s, self.mlps):
                x1 = ln_1(x)
                attention_output = mha(x1, attention_mask)
                x2 = x1 + attention_output
                x3 = ln_2(x2)
                x3 = mlp(x3)
                x = x3 + x2

            return x

    class LateDropout(tf.keras.layers.Layer):
        def __init__(self, rate, noise_shape=None, start_step=160*N_EPOCHS//2, **kwargs):
            super().__init__(**kwargs)
            self.rate = rate
            self.start_step = start_step
            self.dropout = tf.keras.layers.Dropout(rate, noise_shape=noise_shape)

        def build(self, input_shape):
            super().build(input_shape)
            agg = tf.VariableAggregation.ONLY_FIRST_REPLICA
            self._train_counter = tf.Variable(0, dtype="int64", aggregation=agg, trainable=False)

        def call(self, inputs, training=False):
            if training:
                x = tf.cond(self._train_counter < self.start_step, lambda:inputs,  lambda:self.dropout(inputs,training=training))
                self._train_counter.assign_add(1)
            else:
                x = inputs
            return x
    # %% [markdown]
    # # Landmark Embedding

    # %% [code] {"execution":{"iopub.status.busy":"2023-03-24T17:24:02.513568Z","iopub.execute_input":"2023-03-24T17:24:02.514655Z","iopub.status.idle":"2023-03-24T17:24:02.523575Z","shell.execute_reply.started":"2023-03-24T17:24:02.514603Z","shell.execute_reply":"2023-03-24T17:24:02.522523Z"}}
    class LandmarkEmbedding(tf.keras.Model):
        def __init__(self, units, name):
            super(LandmarkEmbedding, self).__init__(name=f'{name}_embedding')
            self.units = units

        def build(self, input_shape):
            # Embedding for missing landmark in frame, initizlied with zeros
            self.empty_embedding = self.add_weight(
                name=f'{self.name}_empty_embedding',
                shape=[self.units],
                initializer=INIT_ZEROS,
            )
            self.per_cls_embedding = tf.Variable(tf.zeros([self.units], dtype=tf.float32), name='per_cls_embedding')

            # Embedding
            self.dense = tf.keras.Sequential([
                tf.keras.layers.Dense(384, name=f'{self.name}_dense_1', use_bias=False, kernel_initializer=INIT_GLOROT_UNIFORM, activation=GELU),
                tf.keras.layers.Dense(self.units, name=f'{self.name}_dense_2', use_bias=False, kernel_initializer=INIT_HE_UNIFORM),
            ], name=f'{self.name}_dense')

        def call(self, x):
            return tf.where(
                    # Checks whether landmark is missing in frame
                    tf.reduce_sum(x, axis=2, keepdims=True) == 0,
                    # If so, the empty embedding is used
                    self.empty_embedding,
                    # Otherwise the landmark data is embedded
                    self.dense(x),
                ) + self.per_cls_embedding

    # %% [markdown]
    # # Embedding

    # %% [code] {"execution":{"iopub.status.busy":"2023-03-24T17:24:03.177702Z","iopub.execute_input":"2023-03-24T17:24:03.178116Z","iopub.status.idle":"2023-03-24T17:24:03.191425Z","shell.execute_reply.started":"2023-03-24T17:24:03.178084Z","shell.execute_reply":"2023-03-24T17:24:03.190242Z"}}
    class Embedding(tf.keras.Model):
        def __init__(self):
            super(Embedding, self).__init__()

        def get_diffs(self, l):
            S = l.shape[2]
            other = tf.expand_dims(l, 3)
            other = tf.repeat(other, S, axis=3)
            other = tf.transpose(other, [0,1,3,2])
            diffs = tf.expand_dims(l, 3) - other
            diffs = tf.reshape(diffs, [-1, INPUT_SIZE, S*S])
            return diffs

        def build(self, input_shape):
            # Positional Embedding, initialized with zeros
            self.positional_embedding = tf.keras.layers.Embedding(INPUT_SIZE+1, UNITS, embeddings_initializer=INIT_ZEROS)
            # Embedding layer for Landmarks
            self.motion_embedding = LandmarkEmbedding(MOTION_UNITS, 'motion')

            self.lips_embedding = LandmarkEmbedding(LIPS_UNITS, 'lips')
            self.left_hand_embedding = LandmarkEmbedding(HANDS_UNITS, 'left_hand')
            self.right_hand_embedding = LandmarkEmbedding(HANDS_UNITS, 'right_hand')
            self.pose_embedding = LandmarkEmbedding(POSE_UNITS, 'pose')
            # Landmark Weights

            self.cls_embedding = tf.Variable(tf.zeros([UNITS], dtype=tf.float32), name='cls_embedding')
            # self.landmark_weights = tf.Variable(tf.zeros([4], dtype=tf.float32), name='landmark_weights')
            # Fully Connected Layers for combined landmarks
            self.fc = tf.keras.Sequential([
                tf.keras.layers.Dense(384, name='fully_connected_1', use_bias=False, kernel_initializer=INIT_GLOROT_UNIFORM, activation=GELU),
                tf.keras.layers.Dense(UNITS, name='fully_connected_2', use_bias=False, kernel_initializer=INIT_HE_UNIFORM),
            ], name='fc')
            self.weight = tf.keras.layers.Dense(1, name=f'{self.name}_dense_3', use_bias=False, kernel_initializer=INIT_HE_UNIFORM)
            self.dropout = LateDropout(0.2)

        def call(self, lips0, left_hand0, right_hand0, pose0, motion0, non_empty_frame_idxs, training=False):
            motion_embedding = self.motion_embedding(motion0)
            # Lips
            lips_embedding = self.lips_embedding(lips0)
            w_lips = self.weight(lips_embedding)
            # Left Hand
            left_hand_embedding = self.left_hand_embedding(left_hand0)
            w_left_hand = self.weight(left_hand_embedding)
            # Right Hand
            right_hand_embedding = self.right_hand_embedding(right_hand0)
            w_right_hand = self.weight(right_hand_embedding)
            # Pose
            # [bs N 2]  # [bs N_frame N 2] # [bs N_frame//SIZE, SIZE, N, 2]
            pose_embedding = self.pose_embedding(pose0)
            w_pose = self.weight(pose_embedding)
            # Merge Embeddings of all landmarks with mean pooling
            x = tf.stack((lips_embedding, left_hand_embedding, right_hand_embedding, pose_embedding), axis=3) #[bs, units, 32]
            landmark_weights = tf.stack((w_lips, w_left_hand, w_right_hand, w_pose), axis=3) # [bs, 4]
            # Merge Landmarks with trainable attention weights
            x = x * tf.nn.softmax(landmark_weights, axis=3)
            x = tf.reduce_sum(x, axis=3)
            x = tf.concat((x, motion_embedding), axis=-1)
            # Fully Connected Layers
            x = self.fc(x)
            x = self.dropout(x) 


            # Add Positional Embedding
            normalised_non_empty_frame_idxs = tf.where(
                tf.math.equal(non_empty_frame_idxs, -1.0),
                INPUT_SIZE,
                tf.cast(
                    non_empty_frame_idxs / tf.reduce_max(non_empty_frame_idxs, axis=1, keepdims=True) * INPUT_SIZE,
                    tf.int32,
                ),
            )
            x = x + self.positional_embedding(normalised_non_empty_frame_idxs)
            x = x + self.cls_embedding
            return x
    # Inputs
    frames = tf.keras.layers.Input([INPUT_SIZE, N_COLS, N_DIMS], dtype=tf.float32, name='frames')
    non_empty_frame_idxs = tf.keras.layers.Input([INPUT_SIZE], dtype=tf.float32, name='non_empty_frame_idxs')
    # Padding Mask
    mask = tf.cast(tf.math.not_equal(non_empty_frame_idxs, -1), tf.float32)
    mask = tf.expand_dims(mask, axis=2)
    
    """
        left_hand: 468:489
        pose: 489:522
        right_hand: 522:543
    """
    x = frames
    x = tf.slice(x, [0,0,0,0], [-1,INPUT_SIZE, N_COLS, 2])
    left = np.arange(INPUT_SIZE-1)
    right = np.arange(1, INPUT_SIZE)
    motion = tf.pad(tf.gather(x, left, axis=1) - tf.gather(x, right, axis=1), [[0,0],[0,1],[0,0],[0,0]])
    motion = tf.where(tf.math.equal(x, 0.0), 0.0, motion)
    motion_dist = tf.math.sqrt(tf.math.reduce_mean(motion**2, axis=-1, keepdims=True))
    motion = tf.concat((motion, motion_dist), axis=-1)
    motion = tf.reshape(motion, [-1, INPUT_SIZE, 106*3])
    # x = tf.concat((x, motion), axis=-1)
    # LIPS
    lips = tf.slice(x, [0,0,LIPS_START,0], [-1,INPUT_SIZE, 56, 2])
    # lips = tf.where(
    #         tf.math.equal(lips, 0.0),
    #         0.0,
    #         (lips - LIPS_MEAN) / LIPS_STD,
    #     )
    lips = tf.reshape(lips, [-1, INPUT_SIZE, 56*2])
    # LEFT HAND
    left_hand = tf.slice(x, [0,0,56,0], [-1,INPUT_SIZE, 21, 2])
    # left_hand = tf.where(
    #         tf.math.equal(left_hand, 0.0),
    #         0.0,
    #         (left_hand - LEFT_HANDS_MEAN) / LEFT_HANDS_STD,
    #     )
    left_hand = tf.reshape(left_hand, [-1, INPUT_SIZE, 21*2])
    # RIGHT HAND
    right_hand = tf.slice(x, [0,0,77,0], [-1,INPUT_SIZE, 21, 2])
    # right_hand = tf.where(
    #         tf.math.equal(right_hand, 0.0),
    #         0.0,
    #         (right_hand - RIGHT_HANDS_MEAN) / RIGHT_HANDS_STD,
    #     )
    right_hand = tf.reshape(right_hand, [-1, INPUT_SIZE, 21*2])
    # POSE
    pose = tf.slice(x, [0,0,98,0], [-1,INPUT_SIZE, 8, 2])
    # pose = tf.where(
    #         tf.math.equal(pose, 0.0),
    #         0.0,
    #         (pose - POSE_MEAN) / POSE_STD,
    #     )
    pose = tf.reshape(pose, [-1, INPUT_SIZE, 8*2])
    
    # x = lips, left_hand, right_hand, pose    
    x = Embedding()(lips, left_hand, right_hand, pose, motion, non_empty_frame_idxs)
    
    # Encoder Transformer Blocks
    x = Transformer(NUM_BLOCKS)(x, mask) + x
    
    # Pooling
    x = tf.reduce_sum(x * mask, axis=1) / tf.reduce_sum(mask, axis=1)
    # Classifier Dropout
    x = LateDropout(CLASSIFIER_DROPOUT_RATIO)(x)

    # Classification Layer
    x = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax, kernel_initializer=INIT_GLOROT_UNIFORM)(x)
    
    outputs = x
    
    # Create Tensorflow Model
    model = tf.keras.models.Model(inputs=[frames, non_empty_frame_idxs], outputs=outputs)
    
    # Adam Optimizer with weight decay
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5, clipnorm=1.0)

    # TopK Metrics
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top_10_acc'),
    ]
    
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)
    
    return model


def get_model_global():
    def scaled_dot_product(q,k,v, softmax, attention_mask):
        #calculates Q . K(transpose)
        qkt = tf.matmul(q,k,transpose_b=True)
        #caculates scaling factor
        dk = tf.math.sqrt(tf.cast(q.shape[-1],dtype=tf.float32))
        scaled_qkt = qkt/dk
        softmax = softmax(scaled_qkt, mask=attention_mask)

        z = tf.matmul(softmax,v)
        #shape: (m,Tx,depth), same shape as q,k,v
        return z

    class MultiHeadAttention(tf.keras.layers.Layer):
        def __init__(self,d_model,num_of_heads):
            super(MultiHeadAttention,self).__init__()
            self.d_model = d_model
            self.num_of_heads = num_of_heads
            self.depth = d_model//num_of_heads
            self.wq = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
            self.wk = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
            self.wv = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
            self.wo = tf.keras.layers.Dense(d_model)
            self.softmax = tf.keras.layers.Softmax()

        def call(self,x, attention_mask):

            multi_attn = []
            for i in range(self.num_of_heads):
                Q = self.wq[i](x)
                K = self.wk[i](x)
                V = self.wv[i](x)
                multi_attn.append(scaled_dot_product(Q,K,V, self.softmax, attention_mask))

            multi_head = tf.concat(multi_attn,axis=-1)
            multi_head_attention = self.wo(multi_head)
            return multi_head_attention

    # %% [code] {"execution":{"iopub.status.busy":"2023-03-24T17:24:01.886612Z","iopub.execute_input":"2023-03-24T17:24:01.886981Z","iopub.status.idle":"2023-03-24T17:24:01.899889Z","shell.execute_reply.started":"2023-03-24T17:24:01.886946Z","shell.execute_reply":"2023-03-24T17:24:01.898849Z"}}
    # Full Transformer
    class Transformer(tf.keras.Model):
        def __init__(self, num_blocks):
            super(Transformer, self).__init__(name='transformer')
            self.num_blocks = num_blocks

        def build(self, input_shape):
            self.ln_1s = []
            self.mhas = []
            self.ln_2s = []
            self.mlps = []
            # Make Transformer Blocks
            for i in range(self.num_blocks):
                # First Layer Normalisation
                self.ln_1s.append(tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS))
                # Multi Head Attention
                self.mhas.append(MultiHeadAttention(UNITS, NUM_HEADS))
                # Second Layer Normalisation
                self.ln_2s.append(tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS))
                # Multi Layer Perception
                self.mlps.append(tf.keras.Sequential([
                    tf.keras.layers.Dense(384 * MLP_RATIO, activation=GELU, kernel_initializer=INIT_GLOROT_UNIFORM),
                    LateDropout(MLP_DROPOUT_RATIO),
                    tf.keras.layers.Dense(UNITS, kernel_initializer=INIT_HE_UNIFORM),
                ]))

        def call(self, x, attention_mask):
            # Iterate input over transformer blocks
            for ln_1, mha, ln_2, mlp in zip(self.ln_1s, self.mhas, self.ln_2s, self.mlps):
                x1 = ln_1(x)
                attention_output = mha(x1, attention_mask)
                x2 = x1 + attention_output
                x3 = ln_2(x2)
                x3 = mlp(x3)
                x = x3 + x2

            return x

    class LateDropout(tf.keras.layers.Layer):
        def __init__(self, rate, noise_shape=None, start_step=160*N_EPOCHS//2, **kwargs):
            super().__init__(**kwargs)
            self.rate = rate
            self.start_step = start_step
            self.dropout = tf.keras.layers.Dropout(rate, noise_shape=noise_shape)

        def build(self, input_shape):
            super().build(input_shape)
            agg = tf.VariableAggregation.ONLY_FIRST_REPLICA
            self._train_counter = tf.Variable(0, dtype="int64", aggregation=agg, trainable=False)

        def call(self, inputs, training=False):
            if training:
                x = tf.cond(self._train_counter < self.start_step, lambda:inputs,  lambda:self.dropout(inputs,training=training))
                self._train_counter.assign_add(1)
            else:
                x = inputs
            return x
    # %% [markdown]
    # # Landmark Embedding

    # %% [code] {"execution":{"iopub.status.busy":"2023-03-24T17:24:02.513568Z","iopub.execute_input":"2023-03-24T17:24:02.514655Z","iopub.status.idle":"2023-03-24T17:24:02.523575Z","shell.execute_reply.started":"2023-03-24T17:24:02.514603Z","shell.execute_reply":"2023-03-24T17:24:02.522523Z"}}
    class LandmarkEmbedding(tf.keras.Model):
        def __init__(self, units, name):
            super(LandmarkEmbedding, self).__init__(name=f'{name}_embedding')
            self.units = units

        def build(self, input_shape):
            # Embedding for missing landmark in frame, initizlied with zeros
            self.empty_embedding = self.add_weight(
                name=f'{self.name}_empty_embedding',
                shape=[self.units],
                initializer=INIT_ZEROS,
            )
            self.per_cls_embedding = tf.Variable(tf.zeros([self.units], dtype=tf.float32), name='per_cls_embedding')

            # Embedding
            self.dense = tf.keras.Sequential([
                tf.keras.layers.Dense(384, name=f'{self.name}_dense_1', use_bias=False, kernel_initializer=INIT_GLOROT_UNIFORM, activation=GELU),
                tf.keras.layers.Dense(self.units, name=f'{self.name}_dense_2', use_bias=False, kernel_initializer=INIT_HE_UNIFORM),
            ], name=f'{self.name}_dense')

        def call(self, x):
            return tf.where(
                    # Checks whether landmark is missing in frame
                    tf.reduce_sum(x, axis=2, keepdims=True) == 0,
                    # If so, the empty embedding is used
                    self.empty_embedding,
                    # Otherwise the landmark data is embedded
                    self.dense(x),
                ) + self.per_cls_embedding

    # %% [markdown]
    # # Embedding

    # %% [code] {"execution":{"iopub.status.busy":"2023-03-24T17:24:03.177702Z","iopub.execute_input":"2023-03-24T17:24:03.178116Z","iopub.status.idle":"2023-03-24T17:24:03.191425Z","shell.execute_reply.started":"2023-03-24T17:24:03.178084Z","shell.execute_reply":"2023-03-24T17:24:03.190242Z"}}
    class Embedding(tf.keras.Model):
        def __init__(self):
            super(Embedding, self).__init__()

        def get_diffs(self, l):
            S = l.shape[2]
            other = tf.expand_dims(l, 3)
            other = tf.repeat(other, S, axis=3)
            other = tf.transpose(other, [0,1,3,2])
            diffs = tf.expand_dims(l, 3) - other
            diffs = tf.reshape(diffs, [-1, INPUT_SIZE, S*S])
            return diffs

        def build(self, input_shape):
            # Positional Embedding, initialized with zeros
            self.positional_embedding = tf.keras.layers.Embedding(INPUT_SIZE+1, UNITS, embeddings_initializer=INIT_ZEROS)
            # Embedding layer for Landmarks
            self.motion_embedding = LandmarkEmbedding(MOTION_UNITS, 'motion')
            self.xyz_embedding = LandmarkEmbedding(XYZ_UNITS, 'xyz')
            # Landmark Weights

            self.cls_embedding = tf.Variable(tf.zeros([UNITS], dtype=tf.float32), name='cls_embedding')
            # self.landmark_weights = tf.Variable(tf.zeros([4], dtype=tf.float32), name='landmark_weights')
            # Fully Connected Layers for combined landmarks
            self.fc = tf.keras.Sequential([
                tf.keras.layers.Dense(384, name='fully_connected_1', use_bias=False, kernel_initializer=INIT_GLOROT_UNIFORM, activation=GELU),
                tf.keras.layers.Dense(UNITS, name='fully_connected_2', use_bias=False, kernel_initializer=INIT_HE_UNIFORM),
            ], name='fc')
            # self.weight = tf.keras.layers.Dense(1, name=f'{self.name}_dense_3', use_bias=False, kernel_initializer=INIT_HE_UNIFORM)
            self.dropout = LateDropout(0.2)

        def call(self, xyz0, motion0, non_empty_frame_idxs, training=False):
            motion_embedding = self.motion_embedding(motion0)
            xyz_embedding = self.xyz_embedding(xyz0)
            x = tf.concat((xyz_embedding, motion_embedding), axis=-1)
            # Fully Connected Layers
            x = self.fc(x)
            x = self.dropout(x) 


            # Add Positional Embedding
            normalised_non_empty_frame_idxs = tf.where(
                tf.math.equal(non_empty_frame_idxs, -1.0),
                INPUT_SIZE,
                tf.cast(
                    non_empty_frame_idxs / tf.reduce_max(non_empty_frame_idxs, axis=1, keepdims=True) * INPUT_SIZE,
                    tf.int32,
                ),
            )
            x = x + self.positional_embedding(normalised_non_empty_frame_idxs)
            x = x + self.cls_embedding
            return x
    # Inputs
    frames = tf.keras.layers.Input([INPUT_SIZE, N_COLS, N_DIMS], dtype=tf.float32, name='frames')
    non_empty_frame_idxs = tf.keras.layers.Input([INPUT_SIZE], dtype=tf.float32, name='non_empty_frame_idxs')
    # Padding Mask
    mask = tf.cast(tf.math.not_equal(non_empty_frame_idxs, -1), tf.float32)
    mask = tf.expand_dims(mask, axis=2)
    
    """
        left_hand: 468:489
        pose: 489:522
        right_hand: 522:543
    """
    x = frames
    x = tf.slice(x, [0,0,0,0], [-1,INPUT_SIZE, N_COLS, 2])
    left = np.arange(INPUT_SIZE-1)
    right = np.arange(1, INPUT_SIZE)
    motion = tf.pad(tf.gather(x, left, axis=1) - tf.gather(x, right, axis=1), [[0,0],[0,1],[0,0],[0,0]])
    motion = tf.where(tf.math.equal(x, 0.0), 0.0, motion)
    motion_dist = tf.math.sqrt(tf.math.reduce_mean(motion**2, axis=-1, keepdims=True))
    motion = tf.concat((motion, motion_dist), axis=-1)
    motion = tf.reshape(motion, [-1, INPUT_SIZE, 106*3])

    xyz = tf.reshape(x, [-1, INPUT_SIZE, N_COLS*2])

    x = Embedding()(xyz, motion, non_empty_frame_idxs)
    
    # Encoder Transformer Blocks
    x = Transformer(NUM_BLOCKS)(x, mask) + x
    
    # Pooling
    x = tf.reduce_sum(x * mask, axis=1) / tf.reduce_sum(mask, axis=1)
    # Classifier Dropout
    x = LateDropout(CLASSIFIER_DROPOUT_RATIO)(x)

    # Classification Layer
    x = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax, kernel_initializer=INIT_GLOROT_UNIFORM)(x)
    
    outputs = x
    
    # Create Tensorflow Model
    model = tf.keras.models.Model(inputs=[frames, non_empty_frame_idxs], outputs=outputs)
    
    # Adam Optimizer with weight decay
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5, clipnorm=1.0)

    # TopK Metrics
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top_10_acc'),
    ]
    
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)
    
    return model

tf.keras.backend.clear_session()
model = get_model()


from tensorflow.keras.layers import (
    InputLayer, Activation, Conv1D, Dropout, BatchNormalization, 
    MaxPool1D, GlobalAvgPool1D, Flatten, Reshape,
    Conv2D, MaxPool2D, GlobalAvgPool2D, AvgPool1D,
    DepthwiseConv1D, Dense, DepthwiseConv2D
)
from tensorflow.keras.models import Sequential, model_from_json



def get_conv1d_model():
    class LateDropout(tf.keras.layers.Layer):
        def __init__(self, rate, noise_shape=None, start_step=160*N_EPOCHS//2, **kwargs):
            super().__init__(**kwargs)
            self.rate = rate
            self.start_step = start_step
            self.dropout = tf.keras.layers.Dropout(rate, noise_shape=noise_shape)

        def build(self, input_shape):
            super().build(input_shape)
            agg = tf.VariableAggregation.ONLY_FIRST_REPLICA
            self._train_counter = tf.Variable(0, dtype="int64", aggregation=agg, trainable=False)

        def call(self, inputs, training=False):
            if training:
                x = tf.cond(self._train_counter < self.start_step, lambda:inputs,  lambda:self.dropout(inputs,training=training))
                self._train_counter.assign_add(1)
            else:
                x = inputs
            return x
    # %% [markdown]
    # # Landmark Embedding

    # %% [code] {"execution":{"iopub.status.busy":"2023-03-24T17:24:02.513568Z","iopub.execute_input":"2023-03-24T17:24:02.514655Z","iopub.status.idle":"2023-03-24T17:24:02.523575Z","shell.execute_reply.started":"2023-03-24T17:24:02.514603Z","shell.execute_reply":"2023-03-24T17:24:02.522523Z"}}
    class LandmarkEmbedding(tf.keras.Model):
        def __init__(self, units, name):
            super(LandmarkEmbedding, self).__init__(name=f'{name}_embedding')
            self.units = units

        def build(self, input_shape):
            # Embedding for missing landmark in frame, initizlied with zeros
            self.empty_embedding = self.add_weight(
                name=f'{self.name}_empty_embedding',
                shape=[self.units],
                initializer=INIT_ZEROS,
            )
            self.per_cls_embedding = tf.Variable(tf.zeros([self.units], dtype=tf.float32), name='per_cls_embedding')

            # Embedding
            self.dense = tf.keras.Sequential([
                tf.keras.layers.Dense(384, name=f'{self.name}_dense_1', use_bias=False, kernel_initializer=INIT_GLOROT_UNIFORM, activation=GELU),
                tf.keras.layers.Dense(self.units, name=f'{self.name}_dense_2', use_bias=False, kernel_initializer=INIT_HE_UNIFORM),
            ], name=f'{self.name}_dense')

        def call(self, x):
            return tf.where(
                    # Checks whether landmark is missing in frame
                    tf.reduce_sum(x, axis=2, keepdims=True) == 0,
                    # If so, the empty embedding is used
                    self.empty_embedding,
                    # Otherwise the landmark data is embedded
                    self.dense(x),
                ) + self.per_cls_embedding

    # %% [markdown]
    # # Embedding

    # %% [code] {"execution":{"iopub.status.busy":"2023-03-24T17:24:03.177702Z","iopub.execute_input":"2023-03-24T17:24:03.178116Z","iopub.status.idle":"2023-03-24T17:24:03.191425Z","shell.execute_reply.started":"2023-03-24T17:24:03.178084Z","shell.execute_reply":"2023-03-24T17:24:03.190242Z"}}
    class Embedding(tf.keras.Model):
        def __init__(self):
            super(Embedding, self).__init__()

        def get_diffs(self, l):
            S = l.shape[2]
            other = tf.expand_dims(l, 3)
            other = tf.repeat(other, S, axis=3)
            other = tf.transpose(other, [0,1,3,2])
            diffs = tf.expand_dims(l, 3) - other
            diffs = tf.reshape(diffs, [-1, INPUT_SIZE, S*S])
            return diffs

        def build(self, input_shape):
            # Positional Embedding, initialized with zeros
            self.positional_embedding = tf.keras.layers.Embedding(INPUT_SIZE+1, UNITS, embeddings_initializer=INIT_ZEROS)
            # Embedding layer for Landmarks
            self.motion_embedding = LandmarkEmbedding(MOTION_UNITS, 'motion')

            self.lips_embedding = LandmarkEmbedding(LIPS_UNITS, 'lips')
            self.left_hand_embedding = LandmarkEmbedding(HANDS_UNITS, 'left_hand')
            self.right_hand_embedding = LandmarkEmbedding(HANDS_UNITS, 'right_hand')
            self.pose_embedding = LandmarkEmbedding(POSE_UNITS, 'pose')
            # Landmark Weights

            self.cls_embedding = tf.Variable(tf.zeros([UNITS], dtype=tf.float32), name='cls_embedding')
            # self.landmark_weights = tf.Variable(tf.zeros([4], dtype=tf.float32), name='landmark_weights')
            # Fully Connected Layers for combined landmarks
            self.fc = tf.keras.Sequential([
                tf.keras.layers.Dense(384, name='fully_connected_1', use_bias=False, kernel_initializer=INIT_GLOROT_UNIFORM, activation=GELU),
                tf.keras.layers.Dense(UNITS, name='fully_connected_2', use_bias=False, kernel_initializer=INIT_HE_UNIFORM),
            ], name='fc')
            self.weight = tf.keras.layers.Dense(1, name=f'{self.name}_dense_3', use_bias=False, kernel_initializer=INIT_HE_UNIFORM)
            self.dropout = LateDropout(0.2)

        def call(self, lips0, left_hand0, right_hand0, pose0, motion0, non_empty_frame_idxs, training=False):
            motion_embedding = self.motion_embedding(motion0)
            # Lips
            lips_embedding = self.lips_embedding(lips0)
            w_lips = self.weight(lips_embedding)
            # Left Hand
            left_hand_embedding = self.left_hand_embedding(left_hand0)
            w_left_hand = self.weight(left_hand_embedding)
            # Right Hand
            right_hand_embedding = self.right_hand_embedding(right_hand0)
            w_right_hand = self.weight(right_hand_embedding)
            # Pose
            # [bs N 2]  # [bs N_frame N 2] # [bs N_frame//SIZE, SIZE, N, 2]
            pose_embedding = self.pose_embedding(pose0)
            w_pose = self.weight(pose_embedding)
            # Merge Embeddings of all landmarks with mean pooling
            x = tf.stack((lips_embedding, left_hand_embedding, right_hand_embedding, pose_embedding), axis=3) #[bs, units, 32]
            landmark_weights = tf.stack((w_lips, w_left_hand, w_right_hand, w_pose), axis=3) # [bs, 4]
            # Merge Landmarks with trainable attention weights
            x = x * tf.nn.softmax(landmark_weights, axis=3)
            x = tf.reduce_sum(x, axis=3)
            x = tf.concat((x, motion_embedding), axis=-1)
            # Fully Connected Layers
            x = self.fc(x)
            x = self.dropout(x) 


            # Add Positional Embedding
            normalised_non_empty_frame_idxs = tf.where(
                tf.math.equal(non_empty_frame_idxs, -1.0),
                INPUT_SIZE,
                tf.cast(
                    non_empty_frame_idxs / tf.reduce_max(non_empty_frame_idxs, axis=1, keepdims=True) * INPUT_SIZE,
                    tf.int32,
                ),
            )
            x = x + self.positional_embedding(normalised_non_empty_frame_idxs)
            x = x + self.cls_embedding
            return x
    # Inputs
    frames = tf.keras.layers.Input([INPUT_SIZE, N_COLS, N_DIMS], dtype=tf.float32, name='frames')
    non_empty_frame_idxs = tf.keras.layers.Input([INPUT_SIZE], dtype=tf.float32, name='non_empty_frame_idxs')
    # Padding Mask
    mask = tf.cast(tf.math.not_equal(non_empty_frame_idxs, -1), tf.float32)
    mask = tf.expand_dims(mask, axis=2)
    
    """
        left_hand: 468:489
        pose: 489:522
        right_hand: 522:543
    """
    x = frames
    x = tf.slice(x, [0,0,0,0], [-1,INPUT_SIZE, N_COLS, 2])
    left = np.arange(INPUT_SIZE-1)
    right = np.arange(1, INPUT_SIZE)
    motion = tf.pad(tf.gather(x, left, axis=1) - tf.gather(x, right, axis=1), [[0,0],[0,1],[0,0],[0,0]])
    motion = tf.where(tf.math.equal(x, 0.0), 0.0, motion)
    motion_dist = tf.math.sqrt(tf.math.reduce_mean(motion**2, axis=-1, keepdims=True))
    motion = tf.concat((motion, motion_dist), axis=-1)
    motion = tf.reshape(motion, [-1, INPUT_SIZE, 106*3])
    # x = tf.concat((x, motion), axis=-1)
    # LIPS
    lips = tf.slice(x, [0,0,LIPS_START,0], [-1,INPUT_SIZE, 56, 2])
    # lips = tf.where(
    #         tf.math.equal(lips, 0.0),
    #         0.0,
    #         (lips - LIPS_MEAN) / LIPS_STD,
    #     )
    lips = tf.reshape(lips, [-1, INPUT_SIZE, 56*2])
    # LEFT HAND
    left_hand = tf.slice(x, [0,0,56,0], [-1,INPUT_SIZE, 21, 2])
    # left_hand = tf.where(
    #         tf.math.equal(left_hand, 0.0),
    #         0.0,
    #         (left_hand - LEFT_HANDS_MEAN) / LEFT_HANDS_STD,
    #     )
    left_hand = tf.reshape(left_hand, [-1, INPUT_SIZE, 21*2])
    # RIGHT HAND
    right_hand = tf.slice(x, [0,0,77,0], [-1,INPUT_SIZE, 21, 2])
    # right_hand = tf.where(
    #         tf.math.equal(right_hand, 0.0),
    #         0.0,
    #         (right_hand - RIGHT_HANDS_MEAN) / RIGHT_HANDS_STD,
    #     )
    right_hand = tf.reshape(right_hand, [-1, INPUT_SIZE, 21*2])
    # POSE
    pose = tf.slice(x, [0,0,98,0], [-1,INPUT_SIZE, 8, 2])
    # pose = tf.where(
    #         tf.math.equal(pose, 0.0),
    #         0.0,
    #         (pose - POSE_MEAN) / POSE_STD,
    #     )
    pose = tf.reshape(pose, [-1, INPUT_SIZE, 8*2])
    
    # x = lips, left_hand, right_hand, pose    
    x = Embedding()(lips, left_hand, right_hand, pose, motion, non_empty_frame_idxs)
    
    do = 0.20
    model = Sequential()

    model.add(Reshape((1, 32, 256)))

    model.add(Conv2D(256, 1, strides=1, padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(DepthwiseConv2D((1,3), strides=1, padding='valid', depth_multiplier=1, activation='relu'))
    model.add(BatchNormalization())

    model.add(MaxPool2D((1,2), (1,2)))

    model.add(Conv2D(256, 1, strides=1, padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(DepthwiseConv2D((1,3), strides=1, padding='valid', depth_multiplier=1, activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(256, 1, strides=1, padding='valid', activation='relu'))
    model.add(BatchNormalization())
    model.add(DepthwiseConv2D((1,3), strides=1, padding='valid', depth_multiplier=4, activation='relu'))
    model.add(BatchNormalization())

    model.add(GlobalAvgPool2D())
    model.add(Dropout(rate=do))

    model.add(Dense(768, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=do))

    model.add(Dense(768, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=do))

    model.add(Dense(250, activation='softmax'))
    
    outputs = model(x)
    
    model = tf.keras.models.Model(inputs=[frames, non_empty_frame_idxs], outputs=outputs)

    # model = tf.keras.models.Model(inputs=[frames, non_empty_frame_idxs], outputs=x)

    # Adam Optimizer with weight decay
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5, clipnorm=1.0)

    # TopK Metrics
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top_10_acc'),
    ]
    
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)
    
    return model


class PreprocessLayerV0(tf.keras.layers.Layer):
    def __init__(self, max_len):
        super(PreprocessLayerV0, self).__init__()
        self._max_len = max_len
        
        self.REF = [500, 501, 512, 513, 159,  386, 13,]

        self.LIP = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        ]
        self.LLIP = self.LIP[10::-1] + self.LIP[19:10:-1] + self.LIP[29:19:-1] + self.LIP[39:29:-1]
        # LLIP = LIP
        self.LHAND = np.arange(468, 489).tolist()
        self.RHAND = np.arange(522, 543).tolist()
        self.POSE = np.arange(500, 512).tolist()
        self.LPOSE = [(i + 1) if (n % 2 == 0) else (i - 1) for n, i in enumerate(self.POSE)]        
        
    @tf.function(
        input_signature=(tf.TensorSpec(shape=[None, ROWS_PER_FRAME, 3], dtype=tf.float32),),
    )
    def call(self, frames):
        ref = tf.gather(frames, self.REF, axis=1)
        
        K = tf.shape(frames)[-1]

        frames_flat = tf.reshape(ref, (-1, K))

        nnan_idxs = ~tf.reduce_any(tf.math.is_nan(frames_flat), -1)

        m = tf.reshape(tf.reduce_mean(frames_flat[nnan_idxs], 0), (1, 1, K))
        s = tf.reduce_mean(tf.math.reduce_std(frames_flat[nnan_idxs], 0))

        frames = frames - m
        frames = frames / s

        lhand_fs = ~tf.reduce_any(tf.math.is_nan(tf.gather(frames, self.LHAND, axis=1)), axis=(-2, -1))
        rhand_fs = ~tf.reduce_any(tf.math.is_nan(tf.gather(frames, self.RHAND, axis=1)), axis=(-2, -1))
        hand_fs = lhand_fs | rhand_fs

        lhanded = tf.reduce_sum(tf.cast(lhand_fs, tf.float32)) > tf.reduce_sum(tf.cast(rhand_fs, tf.float32))

        lframes = tf.concat([
            tf.gather(frames, self.LLIP, axis=1),
            tf.gather(frames, self.LHAND, axis=1),    
        ], axis=1)

        rframes = tf.concat([
            tf.gather(frames, self.LIP, axis=1),
            tf.gather(frames, self.RHAND, axis=1),    
        ], axis=1)

        frames = tf.where(lhanded, lframes, rframes)

        lMf = tf.constant([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)
        rMf = tf.constant([[ 1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)

        Mf = tf.where(lhanded, lMf, rMf)

        frames = frames @ Mf

        sh = tf.shape(frames)
        n_frames = sh[0]

        if n_frames < self._max_len:
            add = self._max_len - n_frames
            # add_b = tf.random.uniform((), 0, add, dtype=tf.int32)
            # add_b = tf.cast(tf.floor(tf.random.uniform((), 0, add)), tf.int32)
            # add_b = tf.cast(tf.floor(add / 2), tf.int32)
            add_b = add // 2
            add_a = add - add_b
            frames = tf.concat([
                tf.fill((add_b, sh[1], sh[2]), math.nan),
                frames,
                tf.fill((add_a, sh[1], sh[2]), math.nan),    
            ], axis=0)    
        elif n_frames > self._max_len:
            f_idxs = tf.cast(tf.math.round(tf.cast(n_frames, dtype=tf.float32) * (tf.range(0, self._max_len, dtype=tf.float32) / self._max_len)), tf.int32)
            # f_idxs = tf.sort(tf.random.shuffle(tf.range(0, n_frames, 1))[:self._max_len])
            # f_idxs = tf.range(0, n_frames, 1)[:self._max_len]
            frames = tf.gather(frames, f_idxs, axis=0)

        out_frames = frames
        
        out_frames = tf.where(tf.math.is_nan(out_frames), 0.0, out_frames)

        return out_frames[..., :2]


class PreprocessLayerV0Pose(tf.keras.layers.Layer):
    def __init__(self, max_len):
        super(PreprocessLayerV0Pose, self).__init__()
        self._max_len = max_len
        
        self.REF = [500, 501, 512, 513, 159,  386, 13,]

        self.LIP = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        ]
        self.LLIP = self.LIP[10::-1] + self.LIP[19:10:-1] + self.LIP[29:19:-1] + self.LIP[39:29:-1]
        # LLIP = LIP
        self.LHAND = np.arange(468, 489).tolist()
        self.RHAND = np.arange(522, 543).tolist()
        self.POSE = np.arange(500, 512).tolist()
        self.LPOSE = [(i + 1) if (n % 2 == 0) else (i - 1) for n, i in enumerate(self.POSE)]        
        
    @tf.function(
        input_signature=(tf.TensorSpec(shape=[None, ROWS_PER_FRAME, 3], dtype=tf.float32),),
    )
    def call(self, frames):
        ref = tf.gather(frames, self.REF, axis=1)
        
        K = tf.shape(frames)[-1]

        frames_flat = tf.reshape(ref, (-1, K))

        nnan_idxs = ~tf.reduce_any(tf.math.is_nan(frames_flat), -1)

        m = tf.reshape(tf.reduce_mean(frames_flat[nnan_idxs], 0), (1, 1, K))
        s = tf.reduce_mean(tf.math.reduce_std(frames_flat[nnan_idxs], 0))

        frames = frames - m
        frames = frames / s

        lhand_fs = ~tf.reduce_any(tf.math.is_nan(tf.gather(frames, self.LHAND, axis=1)), axis=(-2, -1))
        rhand_fs = ~tf.reduce_any(tf.math.is_nan(tf.gather(frames, self.RHAND, axis=1)), axis=(-2, -1))
        hand_fs = lhand_fs | rhand_fs

        lhanded = tf.reduce_sum(tf.cast(lhand_fs, tf.float32)) > tf.reduce_sum(tf.cast(rhand_fs, tf.float32))

        lframes = tf.concat([
            tf.gather(frames, self.LLIP, axis=1),
            tf.gather(frames, self.LHAND, axis=1), 
            tf.gather(frames, self.LPOSE, axis=1), 
        ], axis=1)

        rframes = tf.concat([
            tf.gather(frames, self.LIP, axis=1),
            tf.gather(frames, self.RHAND, axis=1),
            tf.gather(frames, self.POSE, axis=1), 
        ], axis=1)

        frames = tf.where(lhanded, lframes, rframes)

        lMf = tf.constant([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)
        rMf = tf.constant([[ 1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)

        Mf = tf.where(lhanded, lMf, rMf)

        frames = frames @ Mf

        sh = tf.shape(frames)
        n_frames = sh[0]

        if n_frames < self._max_len:
            add = self._max_len - n_frames
            # add_b = tf.random.uniform((), 0, add, dtype=tf.int32)
            # add_b = tf.cast(tf.floor(tf.random.uniform((), 0, add)), tf.int32)
            # add_b = tf.cast(tf.floor(add / 2), tf.int32)
            add_b = add // 2
            add_a = add - add_b
            frames = tf.concat([
                tf.fill((add_b, sh[1], sh[2]), math.nan),
                frames,
                tf.fill((add_a, sh[1], sh[2]), math.nan),    
            ], axis=0)    
        elif n_frames > self._max_len:
            f_idxs = tf.cast(tf.math.round(tf.cast(n_frames, dtype=tf.float32) * (tf.range(0, self._max_len, dtype=tf.float32) / self._max_len)), tf.int32)
            # f_idxs = tf.sort(tf.random.shuffle(tf.range(0, n_frames, 1))[:self._max_len])
            # f_idxs = tf.range(0, n_frames, 1)[:self._max_len]
            frames = tf.gather(frames, f_idxs, axis=0)

        out_frames = frames
        
        out_frames = tf.where(tf.math.is_nan(out_frames), 0.0, out_frames)

        return out_frames[..., :2]


class PreprocessLayerV0Eyes(tf.keras.layers.Layer):
    def __init__(self, max_len):
        super(PreprocessLayerV0Eyes, self).__init__()
        self._max_len = max_len
        
        self.REF = [500, 501, 512, 513, 159,  386, 13,]

        self.LIP = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        ]
        self.LLIP = self.LIP[10::-1] + self.LIP[19:10:-1] + self.LIP[29:19:-1] + self.LIP[39:29:-1]
        # LLIP = LIP
        self.LHAND = np.arange(468, 489).tolist()
        self.RHAND = np.arange(522, 543).tolist()
        #self.POSE = np.arange(500, 512).tolist()
        #self.LPOSE = [(i + 1) if (n % 2 == 0) else (i - 1) for n, i in enumerate(self.POSE)]    
        
        self.EYES = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249, 33, 245, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
        self.LEYES = [33, 245, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]        
        
        
    @tf.function(
        input_signature=(tf.TensorSpec(shape=[None, ROWS_PER_FRAME, 3], dtype=tf.float32),),
    )
    def call(self, frames):
        ref = tf.gather(frames, self.REF, axis=1)
        
        K = tf.shape(frames)[-1]

        frames_flat = tf.reshape(ref, (-1, K))

        nnan_idxs = ~tf.reduce_any(tf.math.is_nan(frames_flat), -1)

        m = tf.reshape(tf.reduce_mean(frames_flat[nnan_idxs], 0), (1, 1, K))
        s = tf.reduce_mean(tf.math.reduce_std(frames_flat[nnan_idxs], 0))

        frames = frames - m
        frames = frames / s

        lhand_fs = ~tf.reduce_any(tf.math.is_nan(tf.gather(frames, self.LHAND, axis=1)), axis=(-2, -1))
        rhand_fs = ~tf.reduce_any(tf.math.is_nan(tf.gather(frames, self.RHAND, axis=1)), axis=(-2, -1))
        hand_fs = lhand_fs | rhand_fs

        lhanded = tf.reduce_sum(tf.cast(lhand_fs, tf.float32)) > tf.reduce_sum(tf.cast(rhand_fs, tf.float32))

        lframes = tf.concat([
            tf.gather(frames, self.LLIP, axis=1),
            tf.gather(frames, self.LHAND, axis=1), 
            tf.gather(frames, self.LEYES, axis=1), 
        ], axis=1)

        rframes = tf.concat([
            tf.gather(frames, self.LIP, axis=1),
            tf.gather(frames, self.RHAND, axis=1),
            tf.gather(frames, self.EYES, axis=1), 
        ], axis=1)

        frames = tf.where(lhanded, lframes, rframes)

        lMf = tf.constant([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)
        rMf = tf.constant([[ 1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)

        Mf = tf.where(lhanded, lMf, rMf)

        frames = frames @ Mf

        sh = tf.shape(frames)
        n_frames = sh[0]

        if n_frames < self._max_len:
            add = self._max_len - n_frames
            # add_b = tf.random.uniform((), 0, add, dtype=tf.int32)
            # add_b = tf.cast(tf.floor(tf.random.uniform((), 0, add)), tf.int32)
            # add_b = tf.cast(tf.floor(add / 2), tf.int32)
            add_b = add // 2
            add_a = add - add_b
            frames = tf.concat([
                tf.fill((add_b, sh[1], sh[2]), math.nan),
                frames,
                tf.fill((add_a, sh[1], sh[2]), math.nan),    
            ], axis=0)    
        elif n_frames > self._max_len:
            f_idxs = tf.cast(tf.math.round(tf.cast(n_frames, dtype=tf.float32) * (tf.range(0, self._max_len, dtype=tf.float32) / self._max_len)), tf.int32)
            # f_idxs = tf.sort(tf.random.shuffle(tf.range(0, n_frames, 1))[:self._max_len])
            # f_idxs = tf.range(0, n_frames, 1)[:self._max_len]
            frames = tf.gather(frames, f_idxs, axis=0)

        out_frames = frames
        
        out_frames = tf.where(tf.math.is_nan(out_frames), 0.0, out_frames)

        return out_frames[..., :2]


class PreprocessLayerV0EyesSparce(tf.keras.layers.Layer):
    def __init__(self, max_len):
        super(PreprocessLayerV0EyesSparce, self).__init__()
        self._max_len = max_len
        
        self.REF = [500, 501, 512, 513, 159,  386, 13,]

        self.LIP = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        ]
        self.LLIP = self.LIP[10::-1] + self.LIP[19:10:-1] + self.LIP[29:19:-1] + self.LIP[39:29:-1]
        # LLIP = LIP
        self.LHAND = np.arange(468, 489).tolist()
        self.RHAND = np.arange(522, 543).tolist()
        #self.POSE = np.arange(500, 512).tolist()
        #self.LPOSE = [(i + 1) if (n % 2 == 0) else (i - 1) for n, i in enumerate(self.POSE)]    
        
        self.EYES = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249, 33, 245, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
        self.LEYES = [33, 245, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]        
        
        
    @tf.function(
        input_signature=(tf.TensorSpec(shape=[None, ROWS_PER_FRAME, 3], dtype=tf.float32),),
    )
    def call(self, frames):
        ref = tf.gather(frames, self.REF, axis=1)
        
        K = tf.shape(frames)[-1]

        frames_flat = tf.reshape(ref, (-1, K))

        nnan_idxs = ~tf.reduce_any(tf.math.is_nan(frames_flat), -1)

        m = tf.reshape(tf.reduce_mean(frames_flat[nnan_idxs], 0), (1, 1, K))
        s = tf.reduce_mean(tf.math.reduce_std(frames_flat[nnan_idxs], 0))

        frames = frames - m
        frames = frames / s

        lhand_fs = ~tf.reduce_any(tf.math.is_nan(tf.gather(frames, self.LHAND, axis=1)), axis=(-2, -1))
        rhand_fs = ~tf.reduce_any(tf.math.is_nan(tf.gather(frames, self.RHAND, axis=1)), axis=(-2, -1))
        hand_fs = lhand_fs | rhand_fs

        lhanded = tf.reduce_sum(tf.cast(lhand_fs, tf.float32)) > tf.reduce_sum(tf.cast(rhand_fs, tf.float32))

        lframes = tf.concat([
            tf.gather(frames, self.LLIP[::2], axis=1),
            tf.gather(frames, self.LHAND, axis=1), 
            tf.gather(frames, self.LEYES[::2], axis=1), 
        ], axis=1)

        rframes = tf.concat([
            tf.gather(frames, self.LIP[::2], axis=1),
            tf.gather(frames, self.RHAND, axis=1),
            tf.gather(frames, self.EYES[::2], axis=1), 
        ], axis=1)

        frames = tf.where(lhanded, lframes, rframes)

        lMf = tf.constant([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)
        rMf = tf.constant([[ 1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)

        Mf = tf.where(lhanded, lMf, rMf)

        frames = frames @ Mf

        sh = tf.shape(frames)
        n_frames = sh[0]

        if n_frames < self._max_len:
            add = self._max_len - n_frames
            # add_b = tf.random.uniform((), 0, add, dtype=tf.int32)
            # add_b = tf.cast(tf.floor(tf.random.uniform((), 0, add)), tf.int32)
            # add_b = tf.cast(tf.floor(add / 2), tf.int32)
            add_b = add // 2
            add_a = add - add_b
            frames = tf.concat([
                tf.fill((add_b, sh[1], sh[2]), math.nan),
                frames,
                tf.fill((add_a, sh[1], sh[2]), math.nan),    
            ], axis=0)    
        elif n_frames > self._max_len:
            f_idxs = tf.cast(tf.math.round(tf.cast(n_frames, dtype=tf.float32) * (tf.range(0, self._max_len, dtype=tf.float32) / self._max_len)), tf.int32)
            # f_idxs = tf.sort(tf.random.shuffle(tf.range(0, n_frames, 1))[:self._max_len])
            # f_idxs = tf.range(0, n_frames, 1)[:self._max_len]
            frames = tf.gather(frames, f_idxs, axis=0)

        out_frames = frames
        
        out_frames = tf.where(tf.math.is_nan(out_frames), 0.0, out_frames)

        return out_frames[..., :2]


my_preprocess_layer96 = PreprocessLayerV0(max_len=96)
my_preprocess_layer32 = PreprocessLayerV0(max_len=32)
my_preprocess_layer32pose = PreprocessLayerV0Pose(max_len=32)
my_preprocess_layer32eyes = PreprocessLayerV0Eyes(max_len=32)
my_preprocess_layer32eyes_s =PreprocessLayerV0EyesSparce(max_len=32)

from tensorflow.keras.models import load_model
from copy import deepcopy

conv1_models_path = "/kaggle/input/conv1dmodels/conv1dmodels/conv1dmodels"


modelname = 'model0sm1_r1_midle_v3_all'
my_model1 = load_model(f"{conv1_models_path}/{modelname}.hdf5")
my_model1.load_weights(f"{conv1_models_path}/{modelname}/best_weights-560.hdf5")
my_preprocess_layer1 = deepcopy(my_preprocess_layer96)

modelname = 'model0sm1_r1_midle_v4_maxlen32_all'
my_model3 = load_model(f"{conv1_models_path}/{modelname}.hdf5")
my_model3.load_weights(f"{conv1_models_path}/{modelname}/best_weights-768.hdf5")
my_preprocess_layer3 = deepcopy(my_preprocess_layer32)

modelname = 'model0sm3_r1_midle_v4_maxlen32_withpose_all'
my_model4 = load_model(f"{conv1_models_path}/{modelname}.hdf5")
my_model4.load_weights(f"{conv1_models_path}/{modelname}/best_weights-280.hdf5")
my_preprocess_layer4 = deepcopy(my_preprocess_layer32pose)

modelname = 'model0sm2_r1_midle_v4_maxlen32_test5_frank_v2_all'
my_model5 = load_model(f"{conv1_models_path}/{modelname}.hdf5")
my_model5.load_weights(f"{conv1_models_path}/{modelname}/best_weights-560.hdf5")
my_preprocess_layer5 = deepcopy(my_preprocess_layer32)


modelname = 'model0sm2_r1_midle_v4_maxlen32_test5_frank_v3_all'
my_model6 = load_model(f"{conv1_models_path}/{modelname}.hdf5")
my_model6.load_weights(f"{conv1_models_path}/{modelname}/best_weights-768.hdf5")
my_preprocess_layer6 = deepcopy(my_preprocess_layer32eyes)

modelname = 'model0sm2_r1_midle_v4_maxlen32_test5_frank_v2_mu0_all'
my_model7 = load_model(f"{conv1_models_path}/{modelname}.hdf5")
my_model7.load_weights(f"{conv1_models_path}/{modelname}/best_weights-980.hdf5")
my_preprocess_layer7 = deepcopy(my_preprocess_layer32)

modelname = 'model0sm2_r1_midle_v4_maxlen32_test5_frank_v4_pfaug_all'
my_model8 = load_model(f"{conv1_models_path}/{modelname}.hdf5")
my_model8.load_weights(f"{conv1_models_path}/{modelname}/best_weights-908.hdf5")
my_preprocess_layer8 = deepcopy(my_preprocess_layer32eyes_s)

modelname = 'model0sm2_r1_midle_v4_maxlen32_test5_frank_v2_smv1_all'
my_model5_sm = load_model(f"{conv1_models_path}/{modelname}.hdf5")
my_model5_sm.load_weights(f"{conv1_models_path}/{modelname}/best_weights-908.hdf5")
my_preprocess_layer5_sm = deepcopy(my_preprocess_layer32)

modelname = 'model0sm2_r1_midle_v4_maxlen32_test5_frank_v3_smv1_all'
my_model6_sm = load_model(f"{conv1_models_path}/{modelname}.hdf5")
my_model6_sm.load_weights(f"{conv1_models_path}/{modelname}/best_weights-768.hdf5")
my_preprocess_layer6_sm = deepcopy(my_preprocess_layer32eyes)

modelname = 'model0sm2_r1_midle_v4_maxlen32_test5_frank_v2_mu0_smv1_all'
my_model7_sm = load_model(f"{conv1_models_path}/{modelname}.hdf5")
my_model7_sm.load_weights(f"{conv1_models_path}/{modelname}/best_weights-980.hdf5")
my_preprocess_layer7_sm = deepcopy(my_preprocess_layer32)