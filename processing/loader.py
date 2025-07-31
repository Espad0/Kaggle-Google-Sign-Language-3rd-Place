"""
Data loading and preparation functions.
"""
import os
import gc
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

from core import (
    DataConfig, 
    load_compressed, 
    save_compressed, 
    load_parquet_landmarks,
    load_metadata
)
from .preprocessing import PreprocessLayer


def get_data(file_path: str, preprocess_layer: PreprocessLayer) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess a single sample."""
    # Load Raw Data
    data = load_parquet_landmarks(file_path)
    # Process Data Using Tensorflow
    data, non_empty_frame_idxs = preprocess_layer(data)
    return data.numpy(), non_empty_frame_idxs.numpy()


def process_dataset(train_df: pd.DataFrame, 
                   config: DataConfig,
                   preprocess_layer: PreprocessLayer,
                   show_progress: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process entire dataset through preprocessing pipeline."""
    n_samples = len(train_df)
    landmarks = preprocess_layer.landmarks
    
    # Create arrays to save data
    X = np.zeros([n_samples, config.INPUT_SIZE, landmarks.n_cols, config.N_DIMS], dtype=np.float32)
    y = np.zeros([n_samples], dtype=np.int32)
    NON_EMPTY_FRAME_IDXS = np.full([n_samples, config.INPUT_SIZE], -1, dtype=np.float32)

    # Process data
    iterator = tqdm(train_df[['file_path', 'sign_ord']].values) if show_progress else train_df[['file_path', 'sign_ord']].values
    
    for row_idx, (file_path, sign_ord) in enumerate(iterator):
        # Log message every 5000 samples
        if row_idx % 5000 == 0 and show_progress:
            print(f'Generated {row_idx}/{n_samples}')

        data, non_empty_frame_idxs = get_data(file_path, preprocess_layer)
        X[row_idx] = data
        y[row_idx] = sign_ord
        NON_EMPTY_FRAME_IDXS[row_idx] = non_empty_frame_idxs
        
        # Sanity check, data should not contain NaN values
        if np.isnan(data).sum() > 0:
            print(f'Warning: NaN values found in sample {row_idx}')

    return X, NON_EMPTY_FRAME_IDXS, y


def create_nan_samples() -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic samples with masked values for robustness (Conv1D specific)."""
    config = DataConfig()
    from core import LandmarkIndices
    landmarks = LandmarkIndices()
    
    n_samples = 10
    samples = np.zeros((n_samples, config.INPUT_SIZE, landmarks.n_cols, config.N_DIMS), dtype=np.float32)
    frame_idxs = np.full((n_samples, config.INPUT_SIZE), -1, dtype=np.float32)
    
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
        samples[idx, pattern] = config.MASK_VAL
        valid_frames = np.arange(config.INPUT_SIZE)[pattern]
        frame_idxs[idx, :len(valid_frames)] = valid_frames
    
    return samples, frame_idxs


def split_train_val(X: np.ndarray, y: np.ndarray, 
                   NON_EMPTY_FRAME_IDXS: np.ndarray,
                   participant_ids: np.ndarray,
                   val_size: float = 0.10,
                   seed: int = 42,
                   add_nan_samples: bool = False) -> Dict:
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
    
    # Add synthetic samples to training if requested (Conv1D)
    if add_nan_samples:
        nan_samples, nan_frames = create_nan_samples()
        X_train = np.concatenate([X_train, nan_samples])
        y_train = np.concatenate([y_train, np.zeros(len(nan_samples), dtype=np.int32)])
        NON_EMPTY_FRAME_IDXS_TRAIN = np.concatenate([NON_EMPTY_FRAME_IDXS_TRAIN, nan_frames])
    
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


def prepare_data(config: Dict[str, bool] = None,
                model_type: str = 'transformer') -> Dict:
    """Main data preparation pipeline."""
    if config is None:
        config = {
            'preprocess': True,
            'use_validation': False,
            'show_plots': False,
            'analyze_stats': True
        }
    
    data_config = DataConfig()
    
    # Load metadata
    print("Loading metadata...")
    train_df, sign2ord, ord2sign = load_metadata()
    
    print(f"Dataset: {len(train_df)} samples, {train_df['sign'].nunique()} unique signs")
    
    # Process or load data
    if config['preprocess']:
        print("\nProcessing data from scratch...")
        preprocess_layer = PreprocessLayer(model_type=model_type)
        X, NON_EMPTY_FRAME_IDXS, y = process_dataset(train_df, data_config, preprocess_layer)
        
        # Save processed data
        print("Saving processed data...")
        save_compressed(X, 'outputs/X.zip')
        save_compressed(y, 'outputs/y.zip')
        save_compressed(NON_EMPTY_FRAME_IDXS, 'outputs/NON_EMPTY_FRAME_IDXS.zip')
        
        # Also save in original format for backward compatibility
        np.save('outputs/X.npy', X)
        np.save('outputs/y.npy', y)
        np.save('outputs/NON_EMPTY_FRAME_IDXS.npy', NON_EMPTY_FRAME_IDXS)
    else:
        print("\nLoading preprocessed data...")
        # Try compressed format first, fall back to original
        try:
            X = load_compressed('outputs/X.zip')
            y = load_compressed('outputs/y.zip')
            NON_EMPTY_FRAME_IDXS = load_compressed('outputs/NON_EMPTY_FRAME_IDXS.zip')
        except FileNotFoundError:
            X = np.load('outputs/X.npy')
            y = np.load('outputs/y.npy')
            NON_EMPTY_FRAME_IDXS = np.load('outputs/NON_EMPTY_FRAME_IDXS.npy')
    
    print(f"Data shape: {X.shape}, Memory: {(X.nbytes + NON_EMPTY_FRAME_IDXS.nbytes + y.nbytes) / 1e9:.2f} GB")
    
    # Split data
    if config['use_validation']:
        print("\nSplitting data with validation set...")
        split_data = split_train_val(
            X, y, NON_EMPTY_FRAME_IDXS, 
            train_df['participant_id'].values,
            seed=data_config.SEED,
            add_nan_samples=(model_type == 'conv1d')
        )
        
        # Save splits
        print("Saving train/validation splits...")
        for key in ['X_train', 'y_train', 'NON_EMPTY_FRAME_IDXS_TRAIN',
                    'X_val', 'y_val', 'NON_EMPTY_FRAME_IDXS_VAL']:
            np.save(f'outputs/{key}.npy', split_data[key])
            save_compressed(split_data[key], f'outputs/{key}.zip')
        
        result = split_data
    else:
        print("\nUsing all data for training...")
        # Add synthetic samples for Conv1D
        if model_type == 'conv1d':
            nan_samples, nan_frames = create_nan_samples()
            X_all = np.concatenate([X, nan_samples])
            y_all = np.concatenate([y, np.zeros(len(nan_samples), dtype=np.int32)])
            frames_all = np.concatenate([NON_EMPTY_FRAME_IDXS, nan_frames])
        else:
            X_all, y_all, frames_all = X, y, NON_EMPTY_FRAME_IDXS
            
        result = {
            'X_train': X_all,
            'y_train': y_all,
            'NON_EMPTY_FRAME_IDXS_TRAIN': frames_all,
            'validation_data': None,
            'y_val': None
        }
    
    # Clean up
    del X, y, NON_EMPTY_FRAME_IDXS
    gc.collect()
    
    # Add metadata
    result.update({
        'train': train_df,
        'SIGN2ORD': sign2ord,
        'ORD2SIGN': ord2sign,
        'config': data_config,
    })
    
    print("\nData preparation complete!")
    return result


def load_preprocessed_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed data (backward compatibility)."""
    try:
        X = load_compressed('X.zip')
        frame_indices = load_compressed('NON_EMPTY_FRAME_IDXS.zip')
        y = load_compressed('y.zip')
    except FileNotFoundError:
        X = np.load('X.npy')
        frame_indices = np.load('NON_EMPTY_FRAME_IDXS.npy')
        y = np.load('y.npy')
    
    from core import print_shape_dtype
    print_shape_dtype([X, frame_indices, y], ['X', 'NON_EMPTY_FRAME_IDXS', 'y'])
    return X, frame_indices, y