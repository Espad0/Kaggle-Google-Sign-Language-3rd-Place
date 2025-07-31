"""
Base model interface for sign language recognition models.
"""
from abc import ABC, abstractmethod
import tensorflow as tf
from typing import Dict, Optional, Tuple
import numpy as np


class BaseModel(ABC):
    """Abstract base class for all sign language models."""
    
    def __init__(self, data_config, model_config):
        self.data_config = data_config
        self.model_config = model_config
        self._model = None
    
    @abstractmethod
    def build_model(self, stats: Optional[Dict] = None) -> tf.keras.Model:
        """Build and return the model architecture."""
        pass
    
    @property
    def model(self) -> tf.keras.Model:
        """Get the underlying Keras model."""
        if self._model is None:
            self._model = self.build_model()
        return self._model
    
    def compile(self, optimizer=None, loss=None, metrics=None):
        """Compile the model."""
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def fit(self, *args, **kwargs):
        """Train the model."""
        return self.model.fit(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        """Make predictions."""
        return self.model.predict(*args, **kwargs)
    
    def evaluate(self, *args, **kwargs):
        """Evaluate the model."""
        return self.model.evaluate(*args, **kwargs)
    
    def save_weights(self, filepath):
        """Save model weights."""
        self.model.save_weights(filepath)
    
    def load_weights(self, filepath):
        """Load model weights."""
        self.model.load_weights(filepath)
    
    def summary(self, *args, **kwargs):
        """Print model summary."""
        self.model.summary(*args, **kwargs)