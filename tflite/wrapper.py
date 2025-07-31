"""
TFLite model wrapper that includes preprocessing.
"""
import tensorflow as tf
from typing import Optional

from core import DataConfig, LandmarkIndices
from processing import PreprocessLayer


class TFLiteModel(tf.Module):
    """TFLite model wrapper that includes preprocessing."""
    
    def __init__(self, model, preprocess_layer: Optional[PreprocessLayer] = None, model_type: str = 'transformer'):
        super(TFLiteModel, self).__init__()
        
        # Load the feature generation and main models
        if preprocess_layer is None:
            print(f"PREPARING PREPROCESS LAYER FOR TFLITE MODEL ({model_type})") 
            preprocess_layer = PreprocessLayer(model_type=model_type)
        
        self.preprocess_layer = preprocess_layer
        self.model = model
        self.config = DataConfig()
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name='inputs')
    ])
    def __call__(self, inputs):
        # Preprocess Data
        x, non_empty_frame_idxs = self.preprocess_layer(inputs)
        # Add Batch Dimension
        x = tf.expand_dims(x, axis=0)
        non_empty_frame_idxs = tf.expand_dims(non_empty_frame_idxs, axis=0)
        # Make Prediction
        outputs = self.model({'frames': x, 'non_empty_frame_idxs': non_empty_frame_idxs})
        # Squeeze Output 1x250 -> 250
        outputs = tf.squeeze(outputs, axis=0)

        # Return a dictionary with the output tensor
        return {'outputs': outputs}