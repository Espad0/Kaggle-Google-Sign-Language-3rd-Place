"""
Loss functions for training sign language models.
"""
import tensorflow as tf
from core import DataConfig


@tf.function
def sparse_categorical_crossentropy_with_label_smoothing(y_true, y_pred, label_smoothing: float = 0.25):
    """
    Sparse categorical crossentropy with label smoothing.
    Works with both Transformer and Conv1D models.
    """
    config = DataConfig()
    
    # One Hot Encode Sparsely Encoded Target Sign
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, config.NUM_CLASSES, axis=1)
    y_true = tf.squeeze(y_true, axis=2)
    
    # Categorical Crossentropy with native label smoothing support
    return tf.keras.losses.categorical_crossentropy(
        y_true, y_pred, label_smoothing=label_smoothing
    )


def get_loss_function(label_smoothing: float = 0.25):
    """Get the configured loss function."""
    def loss_fn(y_true, y_pred):
        return sparse_categorical_crossentropy_with_label_smoothing(
            y_true, y_pred, label_smoothing=label_smoothing
        )
    return loss_fn