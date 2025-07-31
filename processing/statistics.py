"""
Statistical analysis functions for sign language data.
"""
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from core import DataConfig, LandmarkIndices


def analyze_frame_statistics(train_df: pd.DataFrame, 
                           sample_size: int = 1000,
                           show_plots: bool = False):
    """Analyze frame statistics in dataset."""
    config = DataConfig()
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
    if show_plots:
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
                           config: DataConfig) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
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