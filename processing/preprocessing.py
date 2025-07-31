"""
TensorFlow preprocessing layer for sign language data.
Shared between Transformer and Conv1D models.
"""
import numpy as np
import tensorflow as tf
from core import DataConfig, LandmarkIndices


class PreprocessLayer(tf.keras.layers.Layer):
    """
    TensorFlow layer to process data in TFLite.
    Handles dominant hand detection, frame filtering, and normalization.
    """
    
    def __init__(self, model_type='transformer'):
        super(PreprocessLayer, self).__init__()
        self.landmarks = LandmarkIndices()
        self.config = DataConfig()
        self.model_type = model_type
        self._init_constants()
    
    def _init_constants(self):
        """Initialize TensorFlow constants."""
        if self.model_type == 'transformer':
            # Transformer-style normalization correction
            normalisation_correction = tf.constant([
                # Add 0.50 to left hand and pose
                [0] * len(self.landmarks.LIPS_IDXS) + 
                [0.50] * len(self.landmarks.LEFT_HAND_IDXS) + 
                [0.50] * len(self.landmarks.POSE_IDXS),
                # Y coordinates stay intact
                [0] * len(self.landmarks.LANDMARK_IDXS_LEFT_DOMINANT0),
                # Z coordinates stay intact
                [0] * len(self.landmarks.LANDMARK_IDXS_LEFT_DOMINANT0),
            ], dtype=tf.float32)
            self.normalisation_correction = tf.transpose(normalisation_correction, [1, 0])
        else:
            # Conv1D-style normalization correction
            correction = np.zeros((self.landmarks.n_cols, 3), dtype=np.float32)
            correction[self.landmarks.left_hand_idx, 0] = 0.50
            correction[self.landmarks.pose_idx, 0] = 0.50
            self.norm_correction = tf.constant(correction)
            
            # Landmark indices
            self.left_dominant_idx = tf.constant(self.landmarks.left_dominant, dtype=tf.int32)
            self.right_dominant_idx = tf.constant(self.landmarks.right_dominant, dtype=tf.int32)
            self.left_hand_idx = tf.constant(self.landmarks.LEFT_HAND, dtype=tf.int32)
            self.right_hand_idx = tf.constant(self.landmarks.RIGHT_HAND, dtype=tf.int32)
            
            # Constants
            self.n_hand_points = tf.constant(len(self.landmarks.LEFT_HAND), dtype=tf.int32)
            self.hand_threshold = tf.constant(self.config.HAND_THRESHOLD, dtype=tf.float32)
            self.flip_x = tf.constant([-1, 1, 1], dtype=tf.float32)
    
    def pad_edge(self, t, repeats, side):
        """Pad tensor by repeating edge values (transformer only)."""
        if side == 'LEFT':
            return tf.concat((tf.repeat(t[:1], repeats=repeats, axis=0), t), axis=0)
        elif side == 'RIGHT':
            return tf.concat((t, tf.repeat(t[-1:], repeats=repeats, axis=0)), axis=0)
    
    @tf.function(
        input_signature=(tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32),),
    )
    def call(self, data0):
        """Process input data: detect dominant hand, filter frames, normalize."""
        if self.model_type == 'transformer':
            return self._call_transformer(data0)
        else:
            return self._call_conv1d(data0)
    
    def _call_transformer(self, data0):
        """Transformer-specific preprocessing."""
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
    
    def _call_conv1d(self, data):
        """Conv1D-specific preprocessing."""
        # Detect dominant hand
        left_dominant = self._detect_dominant_hand(data)
        
        # Filter frames with sufficient hand data
        data, frame_indices = self._filter_valid_frames(data, left_dominant)
        
        # Extract relevant landmarks
        data = self._extract_landmarks(data, left_dominant)
        
        # Adjust coordinates for right-dominant hand
        if not left_dominant:
            data = self._adjust_right_dominant(data)
        
        # Resize to target length
        data, frame_indices = self._resize_sequence(data, frame_indices)
        
        # Handle NaN frames
        data, frame_indices = self._handle_nan_frames(data, frame_indices)
        
        # Normalize
        data = self._normalize_data(data)
        
        return data, frame_indices
    
    def _detect_dominant_hand(self, data):
        """Determine dominant hand based on non-NaN values."""
        left_sum = tf.reduce_sum(tf.where(tf.math.is_nan(
            tf.gather(data, self.left_hand_idx, axis=1)), 0, 1))
        right_sum = tf.reduce_sum(tf.where(tf.math.is_nan(
            tf.gather(data, self.right_hand_idx, axis=1)), 0, 1))
        return left_sum >= right_sum
    
    def _filter_valid_frames(self, data, left_dominant):
        """Keep frames with sufficient hand landmarks."""
        hand_idx = self.left_hand_idx if left_dominant else self.right_hand_idx
        hand_data = tf.gather(data, hand_idx, axis=1)
        
        # Count non-NaN values per frame
        non_nan_count = tf.reduce_sum(
            tf.where(tf.math.is_nan(hand_data), 0, 1), axis=[1, 2])
        
        # Keep frames above threshold
        min_points = tf.cast(self.n_hand_points, tf.float32) * self.hand_threshold
        valid_frames = tf.where(tf.cast(non_nan_count, tf.float32) >= min_points)
        valid_frames = tf.squeeze(valid_frames, axis=1)
        
        # Filter data
        data = tf.gather(data, valid_frames, axis=0)
        frame_indices = tf.cast(valid_frames, tf.float32)
        frame_indices = frame_indices - tf.reduce_min(frame_indices)
        
        return data, frame_indices
    
    def _extract_landmarks(self, data, left_dominant):
        """Extract relevant landmarks based on dominant hand."""
        indices = self.left_dominant_idx if left_dominant else self.right_dominant_idx
        return tf.gather(data, indices, axis=1)
    
    def _adjust_right_dominant(self, data):
        """Mirror and adjust coordinates for right-dominant hand."""
        data_clean = tf.where(tf.math.is_nan(data), 0.0, data)
        return self.norm_correction + (data_clean * self.flip_x)
    
    def _resize_sequence(self, data, frame_indices):
        """Resize sequence to target length."""
        n_frames = tf.shape(data)[0]
        
        if n_frames < self.config.INPUT_SIZE:
            # Pad shorter sequences
            pad_size = self.config.INPUT_SIZE - n_frames
            data = tf.pad(data, [[0, pad_size], [0, 0], [0, 0]])
            frame_indices = tf.pad(frame_indices, [[0, pad_size]], constant_values=-1)
        else:
            # Downsample longer sequences
            indices = self._downsample_indices(n_frames)
            data = tf.gather(data, indices, axis=0)
            frame_indices = tf.gather(frame_indices, indices, axis=0)
        
        return data, frame_indices
    
    def _downsample_indices(self, n_frames):
        """Generate downsampled indices with reduced edge probability."""
        # Lower probability for first/last frames
        probs = tf.concat([
            [0.05],
            tf.fill([n_frames - 2], 0.95),
            [0.05]
        ], axis=0)
        
        # Sample without replacement
        seed = (tf.reduce_sum(tf.cast(n_frames, tf.int32)), 
                tf.reduce_max(tf.cast(n_frames, tf.int32)))
        
        return tf.reshape(
            tf.random.stateless_categorical(
                tf.math.log([probs]), self.config.INPUT_SIZE, seed),
            [self.config.INPUT_SIZE])
    
    def _handle_nan_frames(self, data, frame_indices):
        """Replace frames containing NaN with zeros."""
        has_nan = tf.reduce_any(tf.math.is_nan(data), axis=[1, 2])
        data = tf.where(has_nan[..., tf.newaxis, tf.newaxis], 0.0, data)
        frame_indices = tf.where(has_nan, -1.0, frame_indices)
        return data, frame_indices
    
    def _normalize_data(self, data):
        """Normalize data with robust statistics."""
        # Calculate statistics
        mean = tf.reduce_mean(data, axis=[0, 1], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), 0.0, mean)
        
        std = tf.math.reduce_std(data, axis=[0, 1], keepdims=True)
        std = tf.where(tf.math.is_nan(std), 1.0, std)
        std = tf.where(std < self.config.MIN_STD, 1.0, std)
        
        # Normalize and clip
        data = (data - mean) / std
        data = tf.where(tf.math.is_nan(data), 0.0, data)
        data = tf.clip_by_value(data, *self.config.CLIP_RANGE)
        
        return data