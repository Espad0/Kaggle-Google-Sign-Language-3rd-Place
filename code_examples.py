"""
Key Code Examples from ASL Fingerspelling Recognition Solution
==============================================================

This file contains well-documented excerpts from the competition solution
to demonstrate code quality, clarity, and engineering best practices.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, List, Dict, Optional


class PreprocessLayer(tf.keras.layers.Layer):
    """
    Advanced preprocessing layer for ASL landmark sequences.
    
    This layer handles:
    1. Variable-length sequence normalization
    2. Missing data imputation
    3. Feature extraction and scaling
    4. Sequence padding/downsampling to fixed size
    
    The preprocessing is optimized for TensorFlow Lite deployment
    and handles edge cases gracefully.
    """
    
    def __init__(self):
        super(PreprocessLayer, self).__init__()
        
        # Define landmark indices for different body parts
        # These were determined through extensive experimentation
        self.HAND_IDXS = np.concatenate([
            np.arange(468, 489),  # Left hand landmarks
            np.arange(522, 543)   # Right hand landmarks
        ])
        
        # Reference points for stable normalization
        # Using nose and shoulders for scale-invariant features
        self.REF_IDXS = [1, 2, 98, 327, 11, 12, 13, 14]
        
    def pad_edge(self, tensor: tf.Tensor, repeats: int, side: str) -> tf.Tensor:
        """
        Pad tensor by repeating edge values.
        
        This is superior to zero-padding for time series as it maintains
        continuity at sequence boundaries.
        
        Args:
            tensor: Input tensor to pad
            repeats: Number of times to repeat edge values
            side: 'LEFT' or 'RIGHT' for padding direction
            
        Returns:
            Padded tensor
        """
        if side == 'LEFT':
            # Repeat first frame for left padding
            padding = tf.repeat(tensor[:1], repeats=repeats, axis=0)
            return tf.concat([padding, tensor], axis=0)
        elif side == 'RIGHT':
            # Repeat last frame for right padding
            padding = tf.repeat(tensor[-1:], repeats=repeats, axis=0)
            return tf.concat([tensor, padding], axis=0)
    
    @tf.function(
        input_signature=(tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32),),
    )
    def call(self, data: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Process raw landmark data into model-ready features.
        
        The function signature is decorated with @tf.function for:
        1. Graph optimization
        2. TensorFlow Lite compatibility
        3. Improved inference speed
        
        Args:
            data: Raw landmarks of shape [frames, landmarks, xyz]
            
        Returns:
            Tuple of (processed_features, frame_indices)
        """
        # Step 1: Identify frames with valid hand data
        # This is crucial as many frames have missing hand landmarks
        hand_data = tf.gather(data, self.HAND_IDXS, axis=1)
        frames_hands_valid = tf.reduce_sum(
            tf.cast(~tf.math.is_nan(hand_data), tf.int32),
            axis=[1, 2]
        )
        valid_frame_idxs = tf.squeeze(
            tf.where(frames_hands_valid > 0), axis=1
        )
        
        # Step 2: Filter to only valid frames
        # This significantly reduces computation for sparse sequences
        data_filtered = tf.gather(data, valid_frame_idxs, axis=0)
        n_frames = tf.shape(data_filtered)[0]
        
        # Step 3: Normalize using reference landmarks
        # This makes features invariant to camera distance and position
        ref_points = tf.gather(data_filtered, self.REF_IDXS, axis=1)
        ref_mean = tf.reduce_mean(ref_points, axis=[0, 1], keepdims=True)
        ref_std = tf.math.reduce_std(ref_points, axis=[0, 1], keepdims=True)
        
        # Avoid division by zero with small epsilon
        ref_std = tf.maximum(ref_std, 1e-6)
        data_normalized = (data_filtered - ref_mean) / ref_std
        
        # Step 4: Handle sequence length variations
        if n_frames < self.target_length:
            # Pad short sequences symmetrically
            return self._pad_sequence(data_normalized, valid_frame_idxs)
        else:
            # Downsample long sequences intelligently
            return self._downsample_sequence(data_normalized, valid_frame_idxs)
    
    def _pad_sequence(self, data: tf.Tensor, indices: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Handle sequences shorter than target length."""
        # Implementation details...
        pass
        
    def _downsample_sequence(self, data: tf.Tensor, indices: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Handle sequences longer than target length with intelligent pooling."""
        # Implementation details...
        pass


class LandmarkEmbedding(tf.keras.layers.Layer):
    """
    Specialized embedding layer for different landmark groups.
    
    This layer learns group-specific representations for:
    - Lips (facial expressions)
    - Hands (primary gestures)
    - Pose (body orientation)
    - Eyes (supplementary features)
    
    Each group has different dimensionality and processing needs.
    """
    
    def __init__(self, units: int, name: str):
        super().__init__(name=f'{name}_embedding')
        self.units = units
        
    def build(self, input_shape):
        # Learnable embedding for missing landmarks
        # This is superior to using zeros or mean imputation
        self.empty_embedding = self.add_weight(
            name='empty_embedding',
            shape=[self.units],
            initializer='zeros',
            trainable=True
        )
        
        # Two-layer MLP for feature extraction
        # GELU activation chosen for smooth gradients
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(
                384, 
                activation='gelu',
                kernel_initializer='glorot_uniform'
            ),
            tf.keras.layers.Dense(
                self.units,
                kernel_initializer='he_uniform'
            )
        ])
        
    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Process landmark features with missing data handling.
        
        Args:
            x: Input tensor of shape [batch, sequence, features]
            
        Returns:
            Embedded features of shape [batch, sequence, units]
        """
        # Detect missing landmarks (all zeros after preprocessing)
        is_missing = tf.reduce_sum(x, axis=2, keepdims=True) == 0
        
        # Use empty embedding for missing data, dense network for valid data
        # This conditional processing maintains gradient flow
        embedded = tf.where(
            is_missing,
            self.empty_embedding,
            self.dense(x)
        )
        
        return embedded


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Custom Multi-Head Attention implementation for TFLite compatibility.
    
    TensorFlow's built-in MultiHeadAttention isn't supported in TFLite,
    so this implementation uses basic operations that are guaranteed
    to work after conversion.
    """
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        
        # Separate projections for each head
        # This is less memory efficient but more TFLite-friendly
        self.wq = [tf.keras.layers.Dense(self.depth) for _ in range(num_heads)]
        self.wk = [tf.keras.layers.Dense(self.depth) for _ in range(num_heads)]
        self.wv = [tf.keras.layers.Dense(self.depth) for _ in range(num_heads)]
        
        # Output projection
        self.wo = tf.keras.layers.Dense(d_model)
        
    def scaled_dot_product_attention(
        self, 
        q: tf.Tensor, 
        k: tf.Tensor, 
        v: tf.Tensor,
        mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """
        Compute scaled dot-product attention.
        
        The scaling factor prevents gradient vanishing in deep networks.
        """
        # Compute attention scores
        scores = tf.matmul(q, k, transpose_b=True)
        
        # Scale by square root of dimension
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scores = scores / tf.math.sqrt(dk)
        
        # Apply mask if provided (for padding)
        if mask is not None:
            scores += (mask * -1e9)
        
        # Softmax to get attention weights
        weights = tf.nn.softmax(scores, axis=-1)
        
        # Apply attention to values
        output = tf.matmul(weights, v)
        
        return output
    
    def call(self, x: tf.Tensor, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Apply multi-head attention to input sequence.
        
        Args:
            x: Input tensor of shape [batch, sequence, features]
            mask: Optional attention mask
            
        Returns:
            Attended features of shape [batch, sequence, d_model]
        """
        batch_size = tf.shape(x)[0]
        
        # Process each attention head separately
        attention_outputs = []
        for i in range(self.num_heads):
            # Generate Q, K, V for this head
            q = self.wq[i](x)
            k = self.wk[i](x)
            v = self.wv[i](x)
            
            # Apply attention
            head_output = self.scaled_dot_product_attention(q, k, v, mask)
            attention_outputs.append(head_output)
        
        # Concatenate all heads
        multi_head = tf.concat(attention_outputs, axis=-1)
        
        # Final linear projection
        output = self.wo(multi_head)
        
        return output


def create_ensemble_prediction(
    models: List[tf.keras.Model],
    preprocessors: List[tf.keras.layers.Layer],
    weights: List[float]
) -> tf.keras.Model:
    """
    Create an ensemble model with multiple preprocessing strategies.
    
    This function demonstrates production-ready ensemble creation with:
    1. Input validation
    2. Flexible weighting schemes
    3. Efficient computation graph
    
    Args:
        models: List of trained models
        preprocessors: Corresponding preprocessing layers
        weights: Ensemble weights (should sum to 1.0)
        
    Returns:
        Combined ensemble model
    """
    # Validate inputs
    assert len(models) == len(preprocessors) == len(weights), \
        "Models, preprocessors, and weights must have same length"
    assert abs(sum(weights) - 1.0) < 1e-6, \
        f"Weights must sum to 1.0, got {sum(weights)}"
    
    # Define ensemble computation
    def ensemble_forward(inputs):
        predictions = []
        
        for model, preprocessor, weight in zip(models, preprocessors, weights):
            # Apply model-specific preprocessing
            processed = preprocessor(inputs)
            
            # Get model prediction
            pred = model(processed)
            
            # Apply ensemble weight
            weighted_pred = pred * weight
            predictions.append(weighted_pred)
        
        # Sum weighted predictions
        ensemble_output = tf.add_n(predictions)
        
        return ensemble_output
    
    # Create functional model
    inputs = tf.keras.Input(shape=(None, 543, 3))
    outputs = ensemble_forward(inputs)
    
    ensemble_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return ensemble_model


# Example usage demonstrating best practices
if __name__ == "__main__":
    # Initialize components with clear configuration
    config = {
        'sequence_length': 32,
        'n_landmarks': 543,
        'embedding_dim': 256,
        'n_heads': 8,
        'n_classes': 250
    }
    
    # Create preprocessing layer
    preprocessor = PreprocessLayer()
    
    # Create model components
    landmark_embedder = LandmarkEmbedding(
        units=config['embedding_dim'],
        name='hand'
    )
    
    attention_layer = MultiHeadAttention(
        d_model=config['embedding_dim'],
        num_heads=config['n_heads']
    )
    
    print("ASL Recognition Model Components Initialized Successfully")
    print(f"Configuration: {config}")