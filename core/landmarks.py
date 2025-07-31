"""
Landmark indices management for face, hands, and pose.
"""
import numpy as np


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
        
        # Number of columns after preprocessing
        self.n_cols = len(self.left_dominant)
        
        # Calculate relative indices in processed data
        self._calculate_relative_indices()
    
    def _calculate_relative_indices(self):
        """Calculate relative indices for processed data."""
        # For transformer compatibility (using old naming)
        self.LIPS_IDXS0 = self.LIPS
        self.LEFT_HAND_IDXS0 = self.LEFT_HAND
        self.RIGHT_HAND_IDXS0 = self.RIGHT_HAND
        self.LEFT_POSE_IDXS0 = self.LEFT_POSE
        self.RIGHT_POSE_IDXS0 = self.RIGHT_POSE
        self.LANDMARK_IDXS_LEFT_DOMINANT0 = self.left_dominant
        self.LANDMARK_IDXS_RIGHT_DOMINANT0 = self.right_dominant
        self.HAND_IDXS0 = self.all_hands
        self.N_COLS = self.n_cols
        
        # Indices in processed array
        self.lips_idx = np.where(np.isin(self.left_dominant, self.LIPS))[0]
        self.left_hand_idx = np.where(np.isin(self.left_dominant, self.LEFT_HAND))[0]
        self.right_hand_idx = np.where(np.isin(self.left_dominant, self.RIGHT_HAND))[0]
        self.pose_idx = np.where(np.isin(self.left_dominant, self.LEFT_POSE))[0]
        
        # For transformer compatibility
        self.LIPS_IDXS = self.lips_idx
        self.LEFT_HAND_IDXS = self.left_hand_idx
        self.RIGHT_HAND_IDXS = self.right_hand_idx
        self.POSE_IDXS = self.pose_idx
        self.HAND_IDXS = np.concatenate([self.left_hand_idx, self.right_hand_idx])
        
        # Start positions
        self.lips_start = 0
        self.left_hand_start = len(self.lips_idx)
        self.right_hand_start = self.left_hand_start + len(self.left_hand_idx)
        self.pose_start = self.right_hand_start + len(self.right_hand_idx)
        
        # For transformer compatibility
        self.LIPS_START = self.lips_start
        self.LEFT_HAND_START = self.left_hand_start
        self.RIGHT_HAND_START = self.right_hand_start
        self.POSE_START = self.pose_start