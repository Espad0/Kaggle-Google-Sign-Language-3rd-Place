"""
Transformer Model Training for Sign Language Recognition

This module contains the transformer-based model architecture and training
functions for sign language recognition.
"""

import os
import gc
import time
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.metrics
import scipy.special

from preprocess_data import (
    Config as PreprocessConfig,
    LandmarkIndices,
    prepare_data,
    get_train_batch_all_signs,
    print_shape_dtype
)


# ======================== Configuration ========================
@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    # Model architecture
    layer_norm_eps: float = 1e-6
    lips_units: int = 384
    hands_units: int = 384
    pose_units: int = 384
    units: int = 512  # Final embedding and transformer embedding size
    num_blocks: int = 2
    mlp_ratio: int = 2
    num_heads: int = 8
    
    # Dropout rates
    embedding_dropout: float = 0.00
    mlp_dropout_ratio: float = 0.30
    classifier_dropout_ratio: float = 0.10
    
    # Training parameters
    batch_all_signs_n: int = 2
    batch_size: int = 128
    n_epochs: int = 100
    lr_max: float = 1e-3
    n_warmup_epochs: int = 0
    wd_ratio: float = 0.05
    warmup_method: str = 'log'
    label_smoothing: float = 0.25
    
    # Other settings
    train_model: bool = True
    use_validation: bool = False
    show_plots: bool = False
    verbose: int = 1


# Create config instances
training_config = TrainingConfig()
preprocess_config = PreprocessConfig()
landmarks = LandmarkIndices()

# Map constants for compatibility
NUM_CLASSES = preprocess_config.NUM_CLASSES
INPUT_SIZE = preprocess_config.INPUT_SIZE
N_COLS = landmarks.N_COLS
N_DIMS = preprocess_config.N_DIMS


# ======================== Model Components ========================
def scaled_dot_product(q, k, v, softmax, attention_mask):
    """Scaled dot product attention."""
    # Calculate Q . K(transpose)
    qkt = tf.matmul(q, k, transpose_b=True)
    # Calculate scaling factor
    dk = tf.math.sqrt(tf.cast(q.shape[-1], dtype=tf.float32))
    scaled_qkt = qkt / dk
    softmax = softmax(scaled_qkt, mask=attention_mask)
    
    z = tf.matmul(softmax, v)
    return z


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention layer."""
    
    def __init__(self, d_model, num_of_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_of_heads = num_of_heads
        self.depth = d_model // num_of_heads
        self.wq = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wk = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wv = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wo = tf.keras.layers.Dense(d_model)
        self.softmax = tf.keras.layers.Softmax()
        
    def call(self, x, attention_mask):
        multi_attn = []
        for i in range(self.num_of_heads):
            Q = self.wq[i](x)
            K = self.wk[i](x)
            V = self.wv[i](x)
            multi_attn.append(scaled_dot_product(Q, K, V, self.softmax, attention_mask))
            
        multi_head = tf.concat(multi_attn, axis=-1)
        multi_head_attention = self.wo(multi_head)
        return multi_head_attention


class Transformer(tf.keras.Model):
    """Full Transformer model with multiple blocks."""
    
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
            # Multi Head Attention
            self.mhas.append(MultiHeadAttention(training_config.units, training_config.num_heads))
            # Multi Layer Perceptron
            self.mlps.append(tf.keras.Sequential([
                tf.keras.layers.Dense(
                    training_config.units * training_config.mlp_ratio, 
                    activation=tf.keras.activations.gelu,
                    kernel_initializer=tf.keras.initializers.glorot_uniform()
                ),
                tf.keras.layers.Dropout(training_config.mlp_dropout_ratio),
                tf.keras.layers.Dense(
                    training_config.units,
                    kernel_initializer=tf.keras.initializers.he_uniform()
                ),
            ]))
    
    def call(self, x, attention_mask):
        # Iterate input over transformer blocks
        for mha, mlp in zip(self.mhas, self.mlps):
            x = x + mha(x, attention_mask)
            x = x + mlp(x)
        return x


class LandmarkEmbedding(tf.keras.Model):
    """Embedding layer for individual landmark types."""
    
    def __init__(self, units, name):
        super(LandmarkEmbedding, self).__init__(name=f'{name}_embedding')
        self.units = units
        
    def build(self, input_shape):
        # Embedding for missing landmark in frame, initialized with zeros
        self.empty_embedding = self.add_weight(
            name=f'{self.name}_empty_embedding',
            shape=[self.units],
            initializer=tf.keras.initializers.constant(0.0),
        )
        # Embedding
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.units, 
                name=f'{self.name}_dense_1', 
                use_bias=False,
                kernel_initializer=tf.keras.initializers.glorot_uniform()
            ),
            tf.keras.layers.Activation(tf.keras.activations.gelu),
            tf.keras.layers.Dense(
                self.units, 
                name=f'{self.name}_dense_2', 
                use_bias=False,
                kernel_initializer=tf.keras.initializers.he_uniform()
            ),
        ], name=f'{self.name}_dense')

    def call(self, x):
        return tf.where(
            # Checks whether landmark is missing in frame
            tf.reduce_sum(x, axis=2, keepdims=True) == 0,
            # If so, the empty embedding is used
            self.empty_embedding,
            # Otherwise the landmark data is embedded
            self.dense(x),
        )


class Embedding(tf.keras.Model):
    """Main embedding layer combining all landmarks."""
    
    def __init__(self):
        super(Embedding, self).__init__()
        
    def get_diffs(self, l):
        """Calculate pairwise differences (not used in final model)."""
        S = l.shape[2]
        other = tf.expand_dims(l, 3)
        other = tf.repeat(other, S, axis=3)
        other = tf.transpose(other, [0, 1, 3, 2])
        diffs = tf.expand_dims(l, 3) - other
        diffs = tf.reshape(diffs, [-1, INPUT_SIZE, S * S])
        return diffs

    def build(self, input_shape):
        # Positional Embedding, initialized with zeros
        self.positional_embedding = tf.keras.layers.Embedding(
            INPUT_SIZE + 1, 
            training_config.units, 
            embeddings_initializer=tf.keras.initializers.constant(0.0)
        )
        # Embedding layer for Landmarks
        self.lips_embedding = LandmarkEmbedding(training_config.lips_units, 'lips')
        self.left_hand_embedding = LandmarkEmbedding(training_config.hands_units, 'left_hand')
        self.pose_embedding = LandmarkEmbedding(training_config.pose_units, 'pose')
        # Landmark Weights
        self.landmark_weights = tf.Variable(
            tf.zeros([3], dtype=tf.float32), 
            name='landmark_weights'
        )
        # Fully Connected Layers for combined landmarks
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(
                training_config.units, 
                name='fully_connected_1', 
                use_bias=False,
                kernel_initializer=tf.keras.initializers.glorot_uniform()
            ),
            tf.keras.layers.Activation(tf.keras.activations.gelu),
            tf.keras.layers.Dense(
                training_config.units, 
                name='fully_connected_2', 
                use_bias=False,
                kernel_initializer=tf.keras.initializers.he_uniform()
            ),
        ], name='fc')

    def call(self, lips0, left_hand0, pose0, non_empty_frame_idxs, training=False):
        # Lips
        lips_embedding = self.lips_embedding(lips0)
        # Left Hand
        left_hand_embedding = self.left_hand_embedding(left_hand0)
        # Pose
        pose_embedding = self.pose_embedding(pose0)
        
        # Merge Embeddings of all landmarks with mean pooling
        x = tf.stack((
            lips_embedding, left_hand_embedding, pose_embedding,
        ), axis=3)
        x = x * tf.nn.softmax(self.landmark_weights)
        x = tf.reduce_sum(x, axis=3)
        
        # Fully Connected Layers
        x = self.fc(x)
        
        # Add Positional Embedding
        max_frame_idxs = tf.clip_by_value(
            tf.reduce_max(non_empty_frame_idxs, axis=1, keepdims=True),
            1,
            np.PINF,
        )
        normalised_non_empty_frame_idxs = tf.where(
            tf.math.equal(non_empty_frame_idxs, -1.0),
            INPUT_SIZE,
            tf.cast(
                non_empty_frame_idxs / max_frame_idxs * INPUT_SIZE,
                tf.int32,
            ),
        )
        x = x + self.positional_embedding(normalised_non_empty_frame_idxs)
        
        return x


# ======================== Model Architecture ========================
def get_model(stats: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> tf.keras.Model:
    """Build the complete transformer model."""
    # Extract statistics
    LIPS_MEAN, LIPS_STD = stats['lips']
    LEFT_HANDS_MEAN, LEFT_HANDS_STD = stats['left_hand']
    POSE_MEAN, POSE_STD = stats['pose']
    
    # Inputs
    frames = tf.keras.layers.Input([INPUT_SIZE, N_COLS, N_DIMS], dtype=tf.float32, name='frames')
    non_empty_frame_idxs = tf.keras.layers.Input([INPUT_SIZE], dtype=tf.float32, name='non_empty_frame_idxs')
    
    # Padding Mask
    mask0 = tf.cast(tf.math.not_equal(non_empty_frame_idxs, -1), tf.float32)
    mask0 = tf.expand_dims(mask0, axis=2)
    
    # Random Frame Masking (for training robustness)
    mask = tf.where(
        (tf.random.uniform(tf.shape(mask0)) > 0.25) & tf.math.not_equal(mask0, 0.0),
        1.0,
        0.0,
    )
    # Correct Samples Which are all masked now...
    mask = tf.where(
        tf.math.equal(tf.reduce_sum(mask, axis=[1, 2], keepdims=True), 0.0),
        mask0,
        mask,
    )
    
    # Extract x,y coordinates only (drop z)
    x = frames
    x = tf.slice(x, [0, 0, 0, 0], [-1, INPUT_SIZE, N_COLS, 2])
    
    # LIPS
    lips = tf.slice(x, [0, 0, landmarks.LIPS_START, 0], [-1, INPUT_SIZE, 40, 2])
    lips = tf.where(
        tf.math.equal(lips, 0.0),
        0.0,
        (lips - LIPS_MEAN) / LIPS_STD,
    )
    
    # LEFT HAND
    left_hand = tf.slice(x, [0, 0, 40, 0], [-1, INPUT_SIZE, 21, 2])
    left_hand = tf.where(
        tf.math.equal(left_hand, 0.0),
        0.0,
        (left_hand - LEFT_HANDS_MEAN) / LEFT_HANDS_STD,
    )
    
    # POSE
    pose = tf.slice(x, [0, 0, 61, 0], [-1, INPUT_SIZE, 5, 2])
    pose = tf.where(
        tf.math.equal(pose, 0.0),
        0.0,
        (pose - POSE_MEAN) / POSE_STD,
    )
    
    # Flatten
    lips = tf.reshape(lips, [-1, INPUT_SIZE, 40 * 2])
    left_hand = tf.reshape(left_hand, [-1, INPUT_SIZE, 21 * 2])
    pose = tf.reshape(pose, [-1, INPUT_SIZE, 5 * 2])
    
    # Embedding
    x = Embedding()(lips, left_hand, pose, non_empty_frame_idxs)
    
    # Encoder Transformer Blocks
    x = Transformer(training_config.num_blocks)(x, mask)
    
    # Pooling
    x = tf.reduce_sum(x * mask, axis=1) / tf.reduce_sum(mask, axis=1)
    
    # Classifier Dropout
    x = tf.keras.layers.Dropout(training_config.classifier_dropout_ratio)(x)
    
    # Classification Layer
    x = tf.keras.layers.Dense(
        NUM_CLASSES, 
        activation=tf.keras.activations.softmax,
        kernel_initializer=tf.keras.initializers.glorot_uniform()
    )(x)
    
    outputs = x
    
    # Create Tensorflow Model
    model = tf.keras.models.Model(inputs=[frames, non_empty_frame_idxs], outputs=outputs)
    
    return model


# ======================== Loss and Callbacks ========================
def scce_with_ls(y_true, y_pred):
    """Sparse categorical crossentropy with label smoothing."""
    # One Hot Encode Sparsely Encoded Target Sign
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, NUM_CLASSES, axis=1)
    y_true = tf.squeeze(y_true, axis=2)
    # Categorical Crossentropy with native label smoothing support
    return tf.keras.losses.categorical_crossentropy(
        y_true, y_pred, label_smoothing=training_config.label_smoothing
    )


def lrfn(current_step, num_warmup_steps, lr_max, num_cycles=0.50, num_training_steps=None):
    """Learning rate schedule with warmup and cosine decay."""
    if num_training_steps is None:
        num_training_steps = training_config.n_epochs
        
    if current_step < num_warmup_steps:
        if training_config.warmup_method == 'log':
            return lr_max * 0.10 ** (num_warmup_steps - current_step)
        else:
            return lr_max * 2 ** -(num_warmup_steps - current_step)
    else:
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr_max


class WeightDecayCallback(tf.keras.callbacks.Callback):
    """Update weight decay with learning rate."""
    def __init__(self, wd_ratio=0.05):
        self.step_counter = 0
        self.wd_ratio = wd_ratio
    
    def on_epoch_begin(self, epoch, logs=None):
        self.model.optimizer.weight_decay = self.model.optimizer.learning_rate * self.wd_ratio
        print(f'learning rate: {self.model.optimizer.learning_rate.numpy():.2e}, '
              f'weight decay: {self.model.optimizer.weight_decay.numpy():.2e}')


# ======================== Training Functions ========================
def compile_model(model: tf.keras.Model) -> None:
    """Compile model with optimizer, loss, and metrics."""
    model.compile(
        loss=scce_with_ls,
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=1e-3, 
            weight_decay=1e-5, 
            clipnorm=1.0
        ),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_acc'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top_10_acc'),
        ]
    )


def train_model(model: tf.keras.Model, 
                X_train: np.ndarray, 
                y_train: np.ndarray, 
                NON_EMPTY_FRAME_IDXS_TRAIN: np.ndarray, 
                validation_data: Optional[Tuple],
                config: TrainingConfig) -> tf.keras.callbacks.History:
    """Train the model."""
    # Create learning rate schedule
    LR_SCHEDULE = [
        lrfn(step, num_warmup_steps=config.n_warmup_epochs, lr_max=config.lr_max) 
        for step in range(config.n_epochs)
    ]
    
    # Plot learning rate schedule if requested
    if config.show_plots:
        plot_lr_schedule(LR_SCHEDULE, config.n_epochs)
    
    # Create callbacks
    lr_callback = tf.keras.callbacks.LearningRateScheduler(
        lambda step: LR_SCHEDULE[step], 
        verbose=1
    )
    wd_callback = WeightDecayCallback(config.wd_ratio)
    
    # Create batch generator with all preprocess config
    batch_config = PreprocessConfig()
    batch_config.BATCH_ALL_SIGNS_N = config.batch_all_signs_n
    
    # Train
    history = model.fit(
        x=get_train_batch_all_signs(X_train, y_train, NON_EMPTY_FRAME_IDXS_TRAIN, batch_config),
        steps_per_epoch=len(X_train) // (NUM_CLASSES * config.batch_all_signs_n),
        epochs=config.n_epochs,
        batch_size=config.batch_size,
        validation_data=validation_data,
        callbacks=[lr_callback, wd_callback],
        verbose=config.verbose,
    )
    
    return history


# ======================== Evaluation Functions ========================
def print_classification_report(model: tf.keras.Model, 
                              X_val: np.ndarray, 
                              y_val: np.ndarray,
                              NON_EMPTY_FRAME_IDXS_VAL: np.ndarray,
                              ORD2SIGN: Dict) -> None:
    """Print classification report for validation data."""
    # Get predictions
    y_val_pred = model.predict({
        'frames': X_val, 
        'non_empty_frame_idxs': NON_EMPTY_FRAME_IDXS_VAL
    }, verbose=2).argmax(axis=1)
    
    # Create labels
    labels = [ORD2SIGN.get(i, '').replace(' ', '_') for i in range(NUM_CLASSES)]
    
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
    SIGN2ORD = {v: k for k, v in ORD2SIGN.items()}
    classification_report['sign'] = [e if e in SIGN2ORD else -1 for e in classification_report.index]
    classification_report['sign_ord'] = classification_report['sign'].apply(SIGN2ORD.get).fillna(-1).astype(np.int16)
    
    # Sort on F1-score
    classification_report = pd.concat((
        classification_report.head(NUM_CLASSES).sort_values('f1-score', ascending=False),
        classification_report.tail(3),
    ))

    pd.options.display.max_rows = 999
    print(classification_report)


def print_landmark_weights(model: tf.keras.Model) -> None:
    """Print the learned landmark weights."""
    for w in model.get_layer('embedding').weights:
        if 'landmark_weights' in w.name:
            weights = scipy.special.softmax(w)

    landmarks = ['lips_embedding', 'left_hand_embedding', 'pose_embedding']
    
    for w, lm in zip(weights, landmarks):
        print(f'{lm} weight: {(w*100):.1f}%')


# ======================== Visualization Functions ========================
def plot_lr_schedule(lr_schedule: List[float], epochs: int):
    """Plot learning rate schedule."""
    fig = plt.figure(figsize=(20, 10))
    plt.plot([None] + lr_schedule + [None])
    
    # X Labels
    x = np.arange(1, epochs + 1)
    x_axis_labels = [i if epochs <= 40 or i % 5 == 0 or i == 1 else None for i in range(1, epochs + 1)]
    plt.xlim([1, epochs])
    plt.xticks(x, x_axis_labels)
    
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
            plt.plot(x + 1, val, 'o', color='black')
            offset_y = (max(lr_schedule) - min(lr_schedule)) * 0.02
            plt.annotate(f'{val:.1E}', xy=(x + 1, val + offset_y), size=12, ha=ha)
    
    plt.xlabel('Epoch', size=16, labelpad=5)
    plt.ylabel('Learning Rate', size=16, labelpad=5)
    plt.grid()
    plt.show()


def plot_history_metric(history, metric: str, f_best=np.argmax, ylim=None, 
                       yscale=None, yticks=None, show_plots: bool = True):
    """Plot training history for a specific metric."""
    if not show_plots:
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

    x_ticks = np.arange(1, N_EPOCHS + 1)

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
    plt.xticks(x, fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.legend(prop={'size': 10})
    plt.grid()
    plt.show()


# ======================== Main Execution ========================
def main():
    """Main training pipeline."""
    # Load configuration
    config = TrainingConfig()
    
    # Load data
    print("Loading data...")
    data = prepare_data(
        preprocess=False,  # Assuming data is already preprocessed
        use_validation=config.use_validation,
        show_plots=config.show_plots
    )
    
    X_train = data['X_train']
    y_train = data['y_train']
    NON_EMPTY_FRAME_IDXS_TRAIN = data['NON_EMPTY_FRAME_IDXS_TRAIN']
    validation_data = data.get('validation_data')
    y_val = data.get('y_val')
    SIGN2ORD = data['SIGN2ORD']
    ORD2SIGN = data['ORD2SIGN']
    stats = data['stats']
    
    # Print data info
    print_shape_dtype(
        [X_train, y_train, NON_EMPTY_FRAME_IDXS_TRAIN], 
        ['X_train', 'y_train', 'NON_EMPTY_FRAME_IDXS_TRAIN']
    )
    
    # Sanity Check
    print(f'# NaN Values X_train: {np.isnan(X_train).sum()}')
    
    # Class Count
    print("\nClass distribution:")
    print(pd.Series(y_train).value_counts().to_frame('Class Count').iloc[[0, 1, 2, 3, 4, -5, -4, -3, -2, -1]])
    
    # Test batch generator
    batch_config = PreprocessConfig()
    batch_config.BATCH_ALL_SIGNS_N = config.batch_all_signs_n
    dummy_dataset = get_train_batch_all_signs(X_train, y_train, NON_EMPTY_FRAME_IDXS_TRAIN, batch_config)
    X_batch, y_batch = next(dummy_dataset)
    
    for k, v in X_batch.items():
        print(f'{k} shape: {v.shape}, dtype: {v.dtype}')
    print(f'y_batch shape: {y_batch.shape}, dtype: {y_batch.dtype}')
    
    # Verify each batch contains each sign exactly N times
    print("\nBatch class distribution:")
    print(pd.Series(y_batch).value_counts().to_frame('Counts'))
    
    if not config.train_model:
        print("\nTraining skipped (train_model=False)")
        return
    
    # Build model
    print("\nBuilding model...")
    tf.keras.backend.clear_session()
    model = get_model(stats)
    model.summary(expand_nested=True)
    
    # Compile model
    compile_model(model)
    
    # Test prediction batch
    y_pred = model.predict_on_batch(X_batch).flatten()
    print(f'\n# NaN Values In Prediction: {np.isnan(y_pred).sum()}')
    
    # Plot initial predictions if requested
    if config.show_plots:
        plt.figure(figsize=(12, 5))
        plt.title(f'Softmax Output Initialized Model | µ={y_pred.mean():.3f}, σ={y_pred.std():.3f}', pad=25)
        pd.Series(y_pred).plot(kind='hist', bins=128, label='Class Probability')
        plt.xlim(0, max(y_pred) * 1.1)
        plt.vlines([1 / NUM_CLASSES], 0, plt.ylim()[1], color='red', 
                   label=f'Random Guessing Baseline 1/NUM_CLASSES={1 / NUM_CLASSES:.3f}')
        plt.grid()
        plt.legend()
        plt.show()
    
    # Verify prediction speed
    print("\nTesting prediction speed...")
    start_time = time.time()
    for _ in range(100):
        model.predict_on_batch({
            'frames': X_train[:1], 
            'non_empty_frame_idxs': NON_EMPTY_FRAME_IDXS_TRAIN[:1]
        })
    end_time = time.time()
    avg_time = (end_time - start_time) / 100
    print(f'Average prediction time: {avg_time*1000:.2f}ms')
    
    # Validate if using validation set
    if config.use_validation and validation_data:
        print(f'\n# Unique Signs in Validation Set: {pd.Series(y_val).nunique()}')
        print(pd.Series(y_val).value_counts().to_frame('Count').iloc[[1, 2, 3, -3, -2, -1]])
        
        # Sanity check
        _ = model.evaluate(*validation_data, verbose=2)
    
    # Train model
    print("\nTraining model...")
    history = train_model(model, X_train, y_train, NON_EMPTY_FRAME_IDXS_TRAIN, 
                         validation_data, config)
    
    # Save model weights
    model.save_weights('model.h5')
    print("\nModel weights saved to model.h5")
    
    # Print landmark weights
    print("\nLandmark weights:")
    print_landmark_weights(model)
    
    # Evaluate on validation set
    if config.use_validation and validation_data:
        X_val = validation_data[0]['frames']
        NON_EMPTY_FRAME_IDXS_VAL = validation_data[0]['non_empty_frame_idxs']
        print("\nClassification Report:")
        print_classification_report(model, X_val, y_val, NON_EMPTY_FRAME_IDXS_VAL, ORD2SIGN)
    
    # Plot training history
    if config.show_plots:
        plot_history_metric(history, 'loss', f_best=np.argmin, show_plots=config.show_plots)
        plot_history_metric(history, 'acc', ylim=[0, 1], yticks=np.arange(0.0, 1.1, 0.1), 
                          show_plots=config.show_plots)
        plot_history_metric(history, 'top_5_acc', ylim=[0, 1], yticks=np.arange(0.0, 1.1, 0.1), 
                          show_plots=config.show_plots)
        plot_history_metric(history, 'top_10_acc', ylim=[0, 1], yticks=np.arange(0.0, 1.1, 0.1), 
                          show_plots=config.show_plots)
    
    print("\nTraining complete!")
    return model


if __name__ == "__main__":
    # Run training
    model = main()