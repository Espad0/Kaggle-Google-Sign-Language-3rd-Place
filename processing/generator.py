"""
Batch generators for training models.
"""
from typing import Generator, Dict
import numpy as np
from core import DataConfig, LandmarkIndices


def get_train_batch_all_signs(X: np.ndarray, y: np.ndarray, 
                            NON_EMPTY_FRAME_IDXS: np.ndarray,
                            config: DataConfig) -> Generator:
    """
    Custom sampler to get a batch containing N times all signs.
    Used by Transformer model.
    """
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


def create_batch_generator(
    X: np.ndarray, 
    y: np.ndarray, 
    non_empty_frames: np.ndarray,
    n: int = 2,
    num_classes: int = 250
) -> Generator:
    """
    Generate batches with n samples per class.
    Used by Conv1D model.
    """
    # Map classes to sample indices
    class_to_idxs = {
        i: np.where(y == i)[0].astype(np.int32)
        for i in range(num_classes)
    }
    
    landmarks = LandmarkIndices()
    config = DataConfig()
    
    batch_size = num_classes * n
    X_batch = np.zeros([batch_size, config.INPUT_SIZE, landmarks.n_cols, config.N_DIMS], dtype=np.float32)
    y_batch = np.repeat(np.arange(num_classes), n)
    nef_batch = np.zeros([batch_size, config.INPUT_SIZE], dtype=np.float32)
    
    while True:
        for i in range(num_classes):
            idxs = np.random.choice(class_to_idxs[i], n)
            start_idx = i * n
            end_idx = start_idx + n
            X_batch[start_idx:end_idx] = X[idxs]
            nef_batch[start_idx:end_idx] = non_empty_frames[idxs]
        
        yield {'frames': X_batch, 'non_empty_frame_idxs': nef_batch}, y_batch