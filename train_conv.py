"""
Conv1D Model Training for Sign Language Recognition

A clean, well-structured implementation for training a Conv1D model
on preprocessed sign language data.
"""

import os
import gc
import time
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Generator

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.metrics

from preprocess_data_conv import (
    Config as PreprocessConfig, 
    LandmarkIndices,
    load_compressed,
    split_train_val,
    prepare_data as preprocess_prepare_data,
    print_shape_dtype,
    load_preprocessed_data,
    split_data
)

# Create instances for constants
preprocess_config = PreprocessConfig()
landmarks = LandmarkIndices()

# Map constants for compatibility
N_ROWS = preprocess_config.N_ROWS
N_DIMS = preprocess_config.N_DIMS  
SEED = preprocess_config.SEED
NUM_CLASSES = preprocess_config.NUM_CLASSES
INPUT_SIZE = preprocess_config.INPUT_SIZE
N_COLS = landmarks.n_cols


# ======================== Configuration ========================
@dataclass
class Config:
    """Training configuration parameters"""
    # Model training
    train_model: bool = True
    use_validation: bool = False
    show_plots: bool = False
    
    # Data preprocessing
    generate_data: bool = False  # Set to True to generate data from raw files
    
    # Batch settings
    batch_all_signs_n: int = 2
    batch_size: int = 64
    
    # Training parameters
    n_epochs: int = 100
    lr_max: float = 1e-3
    n_warmup_epochs: int = 0
    wd_ratio: float = 0.05
    warmup_method: str = 'log'
    
    # Other settings
    dropout_rate: float = 0.5
    label_smoothing: float = 0.25
    
    @property
    def verbose(self) -> int:
        return 1 if self.show_plots else 2


config = Config()

# Configure matplotlib
plt.rcParams.update({
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 24
})

print(f'TensorFlow V{tf.__version__}')


# ======================== Data Loading ========================
def prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Tuple], Optional[np.ndarray], Dict, Dict]:
    """Load and prepare training data"""
    # Load metadata
    train_df = pd.read_csv('train.csv')
    train_df['file_path'] = train_df['path'].apply(lambda path: f'./{path}')
    train_df['sign_ord'] = train_df['sign'].astype('category').cat.codes
    
    SIGN2ORD = train_df[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
    ORD2SIGN = train_df[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()
    
    # Check if we need to generate data or if files exist
    data_files = ['X.zip', 'NON_EMPTY_FRAME_IDXS.zip', 'y.zip']
    files_exist = all(os.path.exists(f) for f in data_files)
    
    if config.generate_data or not files_exist:
        if not files_exist:
            print("Preprocessed data files not found. Generating from raw data...")
        else:
            print("Generating data from raw files (generate_data=True)...")
        
        # Generate data using preprocessing pipeline
        preprocess_config = {
            'preprocess': True,
            'use_validation': config.use_validation,
            'show_plots': config.show_plots,
            'analyze_stats': True
        }
        data = preprocess_prepare_data(preprocess_config)
        
        # Extract the arrays (remove synthetic NaN samples added by preprocessing)
        X = load_compressed('X.zip')
        non_empty_frame_idxs = load_compressed('NON_EMPTY_FRAME_IDXS.zip')
        y = load_compressed('y.zip')
        
        print("Data generation complete!")
    else:
        # Load existing preprocessed data
        print("Loading existing preprocessed data...")
        X, non_empty_frame_idxs, y = load_preprocessed_data()
    
    gc.collect()
    
    # Split data
    X_train, y_train, nef_train, val_data, y_val = split_data(
        X, y, non_empty_frame_idxs, train_df, config.use_validation
    )
    
    del X, y, non_empty_frame_idxs
    gc.collect()
    
    return X_train, y_train, nef_train, val_data, y_val, SIGN2ORD, ORD2SIGN


# ======================== Data Generator ========================
def create_batch_generator(
    X: np.ndarray, 
    y: np.ndarray, 
    non_empty_frames: np.ndarray,
    n: int = 2
) -> Generator:
    """Generate batches with n samples per class"""
    # Map classes to sample indices
    class_to_idxs = {
        i: np.where(y == i)[0].astype(np.int32)
        for i in range(NUM_CLASSES)
    }
    
    batch_size = NUM_CLASSES * n
    X_batch = np.zeros([batch_size, INPUT_SIZE, N_COLS, N_DIMS], dtype=np.float32)
    y_batch = np.repeat(np.arange(NUM_CLASSES), n)
    nef_batch = np.zeros([batch_size, INPUT_SIZE], dtype=np.float32)
    
    while True:
        for i in range(NUM_CLASSES):
            idxs = np.random.choice(class_to_idxs[i], n)
            start_idx = i * n
            end_idx = start_idx + n
            X_batch[start_idx:end_idx] = X[idxs]
            nef_batch[start_idx:end_idx] = non_empty_frames[idxs]
        
        yield {'frames': X_batch, 'non_empty_frame_idxs': nef_batch}, y_batch


# ======================== Model Architecture ========================
def create_conv_block(x, filters: int, kernel_size: int, strides: int = 1, 
                     depth_multiplier: int = 1) -> tf.Tensor:
    """Create a convolutional block with batch normalization"""
    x = tf.keras.layers.Conv1D(filters, 1, padding='valid', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    if kernel_size > 1:
        x = tf.keras.layers.DepthwiseConv1D(
            kernel_size, strides=strides, padding='valid', 
            depth_multiplier=depth_multiplier, activation='relu'
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
    
    return x


def build_model() -> tf.keras.Model:
    """Build the Conv1D model architecture"""
    # Input layers
    frames = tf.keras.layers.Input([INPUT_SIZE, N_COLS, N_DIMS], dtype=tf.float32, name='frames')
    non_empty_frame_idxs = tf.keras.layers.Input([INPUT_SIZE], dtype=tf.float32, name='non_empty_frame_idxs')
    
    # Extract x,y coordinates only (drop z)
    x = tf.slice(frames, [0, 0, 0, 0], [-1, INPUT_SIZE, N_COLS, 2])
    x = tf.reshape(x, [-1, INPUT_SIZE, N_COLS * 2])
    
    # Convolutional blocks
    x = create_conv_block(x, 64, 3, depth_multiplier=1)
    x = create_conv_block(x, 64, 5, strides=2, depth_multiplier=4)
    x = tf.keras.layers.MaxPool1D(2, 2)(x)
    
    x = create_conv_block(x, 256, 3, depth_multiplier=1)
    x = create_conv_block(x, 256, 3, strides=2, depth_multiplier=4)
    
    # Global pooling and dense layers
    x = tf.keras.layers.GlobalAvgPool1D()(x)
    x = tf.keras.layers.Dropout(rate=config.dropout_rate)(x)
    
    for _ in range(2):
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(rate=config.dropout_rate)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    return tf.keras.models.Model(inputs=[frames, non_empty_frame_idxs], outputs=outputs)


# ======================== Loss and Callbacks ========================
@tf.function
def sparse_categorical_crossentropy_with_label_smoothing(y_true, y_pred):
    """Sparse categorical crossentropy with label smoothing"""
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, NUM_CLASSES, axis=1)
    y_true = tf.squeeze(y_true, axis=2)
    return tf.keras.losses.categorical_crossentropy(
        y_true, y_pred, label_smoothing=config.label_smoothing
    )


def create_lr_schedule(step: int) -> float:
    """Create learning rate schedule with warmup and cosine decay"""
    if step < config.n_warmup_epochs:
        if config.warmup_method == 'log':
            return config.lr_max * 0.10 ** (config.n_warmup_epochs - step)
        else:
            return config.lr_max * 2 ** -(config.n_warmup_epochs - step)
    
    progress = (step - config.n_warmup_epochs) / max(1, config.n_epochs - config.n_warmup_epochs)
    return max(0.0, 0.5 * (1.0 + np.cos(np.pi * 0.5 * 2.0 * progress))) * config.lr_max


class WeightDecayCallback(tf.keras.callbacks.Callback):
    """Update weight decay with learning rate"""
    def __init__(self, wd_ratio: float = 0.05):
        self.wd_ratio = wd_ratio
    
    def on_epoch_begin(self, epoch, logs=None):
        self.model.optimizer.weight_decay = self.model.optimizer.learning_rate * self.wd_ratio
        if config.verbose == 1:
            print(f'LR: {self.model.optimizer.learning_rate.numpy():.2e}, '
                  f'WD: {self.model.optimizer.weight_decay.numpy():.2e}')


# ======================== Training Functions ========================
def compile_model(model: tf.keras.Model) -> None:
    """Compile model with optimizer, loss, and metrics"""
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=1e-3, 
            weight_decay=1e-5, 
            clipnorm=1.0
        ),
        loss=sparse_categorical_crossentropy_with_label_smoothing,
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_acc'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top_10_acc'),
        ]
    )


def train_model(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray, 
                nef_train: np.ndarray, val_data: Optional[Tuple]) -> tf.keras.callbacks.History:
    """Train the model"""
    # Create callbacks
    lr_schedule = [create_lr_schedule(step) for step in range(config.n_epochs)]
    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(lambda step: lr_schedule[step], verbose=1),
        WeightDecayCallback(config.wd_ratio),
    ]
    
    # Train
    history = model.fit(
        x=create_batch_generator(X_train, y_train, nef_train, config.batch_all_signs_n),
        steps_per_epoch=len(X_train) // (NUM_CLASSES * config.batch_all_signs_n),
        epochs=config.n_epochs,
        batch_size=config.batch_size,
        validation_data=val_data,
        callbacks=callbacks,
        verbose=config.verbose,
    )
    
    return history


# ======================== Evaluation Functions ========================
def evaluate_model(model: tf.keras.Model, X_val: np.ndarray, y_val: np.ndarray,
                   nef_val: np.ndarray, ORD2SIGN: Dict, SIGN2ORD: Dict) -> None:
    """Evaluate model and print classification report"""
    if not config.use_validation:
        return
    
    # Get predictions
    y_pred = model.predict(
        {'frames': X_val, 'non_empty_frame_idxs': nef_val}, 
        verbose=2
    ).argmax(axis=1)
    
    # Create classification report
    labels = [ORD2SIGN.get(i, '').replace(' ', '_') for i in range(NUM_CLASSES)]
    report = sklearn.metrics.classification_report(
        y_val, y_pred, target_names=labels, output_dict=True
    )
    
    # Format and display report
    df_report = pd.DataFrame(report).T.round(2)
    df_report = df_report.astype({'support': np.uint16})
    
    # Sort by F1-score (excluding summary rows)
    class_rows = df_report.head(NUM_CLASSES).sort_values('f1-score', ascending=False)
    summary_rows = df_report.tail(3)
    
    print("\nClassification Report:")
    print(pd.concat([class_rows, summary_rows]))


# ======================== Main Execution ========================
def main():
    """Main training pipeline"""
    # Load data
    print("Loading data...")
    X_train, y_train, nef_train, val_data, y_val, SIGN2ORD, ORD2SIGN = prepare_data()
    
    # Test batch generator
    batch_gen = create_batch_generator(X_train, y_train, nef_train)
    X_batch, y_batch = next(batch_gen)
    print_shape_dtype(
        [X_batch['frames'], X_batch['non_empty_frame_idxs'], y_batch],
        ['X_batch["frames"]', 'X_batch["non_empty_frame_idxs"]', 'y_batch']
    )
    
    if not config.train_model:
        print("Training skipped (TRAIN_MODEL=False)")
        return
    
    # Build and compile model
    print("\nBuilding model...")
    tf.keras.backend.clear_session()
    model = build_model()
    compile_model(model)
    model.summary()
    
    # Verify prediction speed
    print("\nTesting prediction speed...")
    start_time = time.time()
    for _ in range(100):
        model.predict_on_batch({
            'frames': X_train[:1], 
            'non_empty_frame_idxs': nef_train[:1]
        })
    avg_time = (time.time() - start_time) / 100
    print(f'Average prediction time: {avg_time*1000:.2f}ms')
    
    # Sanity check on validation data
    if config.use_validation and val_data:
        print(f'\nValidation set: {pd.Series(y_val).nunique()} unique signs')
        model.evaluate(*val_data, verbose=2)
    
    # Train model
    print("\nTraining model...")
    history = train_model(model, X_train, y_train, nef_train, val_data)
    
    # Save weights
    model.save_weights('model_conv.h5')
    print("\nModel weights saved to model_conv.h5")
    
    # Evaluate
    if config.use_validation and val_data:
        X_val, nef_val = val_data[0]['frames'], val_data[0]['non_empty_frame_idxs']
        evaluate_model(model, X_val, y_val, nef_val, ORD2SIGN, SIGN2ORD)
    
    print("\nConv1D model training complete!")


if __name__ == "__main__":
    main()