"""
Conv1D model architecture for sign language recognition.
"""
import tensorflow as tf
from typing import Dict, Optional, List

from core import DataConfig, Conv1DConfig, LandmarkIndices
from .base import BaseModel


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


class Conv1DModel(BaseModel):
    """Conv1D model for sign language recognition."""
    
    def __init__(self, data_config: DataConfig = None, model_config: Conv1DConfig = None):
        if data_config is None:
            data_config = DataConfig()
        if model_config is None:
            model_config = Conv1DConfig()
        super().__init__(data_config, model_config)
        self.landmarks = LandmarkIndices()
    
    def build_model(self, stats: Optional[Dict] = None) -> tf.keras.Model:
        """Build the Conv1D model architecture."""
        # Input layers
        frames = tf.keras.layers.Input(
            [self.data_config.INPUT_SIZE, self.landmarks.n_cols, self.data_config.N_DIMS], 
            dtype=tf.float32, name='frames'
        )
        non_empty_frame_idxs = tf.keras.layers.Input(
            [self.data_config.INPUT_SIZE], 
            dtype=tf.float32, name='non_empty_frame_idxs'
        )
        
        # Extract x,y coordinates only (drop z)
        x = tf.slice(frames, [0, 0, 0, 0], [-1, self.data_config.INPUT_SIZE, self.landmarks.n_cols, 2])
        x = tf.reshape(x, [-1, self.data_config.INPUT_SIZE, self.landmarks.n_cols * 2])
        
        # Convolutional blocks
        x = create_conv_block(x, 64, 3, depth_multiplier=1)
        x = create_conv_block(x, 64, 5, strides=2, depth_multiplier=4)
        x = tf.keras.layers.MaxPool1D(2, 2)(x)
        
        x = create_conv_block(x, 256, 3, depth_multiplier=1)
        x = create_conv_block(x, 256, 3, strides=2, depth_multiplier=4)
        
        # Global pooling and dense layers
        x = tf.keras.layers.GlobalAvgPool1D()(x)
        x = tf.keras.layers.Dropout(rate=self.model_config.dropout_rate)(x)
        
        for _ in range(2):
            x = tf.keras.layers.Dense(1024, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(rate=self.model_config.dropout_rate)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(self.data_config.NUM_CLASSES, activation='softmax')(x)
        
        # Create model
        model = tf.keras.models.Model(inputs=[frames, non_empty_frame_idxs], outputs=outputs)
        
        self._model = model
        return model