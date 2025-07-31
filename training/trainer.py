"""
Main trainer class for sign language models.
"""
import time
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf

from core import DataConfig, TrainingConfig
from processing import get_train_batch_all_signs, create_batch_generator
from .losses import get_loss_function
from .callbacks import get_callbacks


class Trainer:
    """Trainer class that works with any model."""
    
    def __init__(self, config: TrainingConfig = None):
        if config is None:
            config = TrainingConfig()
        self.config = config
        self.data_config = DataConfig()
    
    def compile_model(self, model, label_smoothing: Optional[float] = None):
        """Compile model with optimizer, loss, and metrics."""
        if label_smoothing is None:
            label_smoothing = self.config.label_smoothing
            
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=1e-3, 
                weight_decay=1e-5, 
                clipnorm=1.0
            ),
            loss=get_loss_function(label_smoothing),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_acc'),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top_10_acc'),
            ]
        )
    
    def train(self, 
              model: Union[tf.keras.Model, 'BaseModel'],
              train_data: Dict,
              val_data: Optional[Dict] = None,
              weights_path: str = None) -> tf.keras.callbacks.History:
        """Train the model."""
        # Extract data
        if hasattr(model, 'model'):
            # It's a BaseModel instance
            keras_model = model.model
        else:
            # It's already a Keras model
            keras_model = model
        
        # Get data arrays
        X_train = train_data.get('X_train')
        y_train = train_data.get('y_train')
        frames_train = train_data.get('NON_EMPTY_FRAME_IDXS_TRAIN', train_data.get('frames_train'))
        
        # Compile if not already compiled
        if not keras_model.optimizer:
            self.compile_model(keras_model)
        
        # Create batch generator based on model type
        model_name = keras_model.name if hasattr(keras_model, 'name') else 'unknown'
        
        if 'conv' in model_name.lower():
            # Conv1D model
            batch_generator = create_batch_generator(
                X_train, y_train, frames_train, 
                n=self.data_config.BATCH_ALL_SIGNS_N,
                num_classes=self.data_config.NUM_CLASSES
            )
        else:
            # Transformer model
            batch_generator = get_train_batch_all_signs(
                X_train, y_train, frames_train, self.data_config
            )
        
        # Create callbacks
        callbacks = get_callbacks(self.config)
        
        # Prepare validation data
        if val_data is not None:
            if isinstance(val_data, dict):
                validation_data = val_data.get('validation_data')
            else:
                validation_data = val_data
        else:
            validation_data = None
        
        # Test prediction speed
        print("\nTesting prediction speed...")
        start_time = time.time()
        for _ in range(100):
            keras_model.predict_on_batch({
                'frames': X_train[:1], 
                'non_empty_frame_idxs': frames_train[:1]
            })
        avg_time = (time.time() - start_time) / 100
        print(f'Average prediction time: {avg_time*1000:.2f}ms')
        
        # Train
        print("\nStarting training...")
        history = keras_model.fit(
            x=batch_generator,
            steps_per_epoch=len(X_train) // (self.data_config.NUM_CLASSES * self.data_config.BATCH_ALL_SIGNS_N),
            epochs=self.config.n_epochs,
            batch_size=self.config.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=self.config.verbose,
        )
        
        # Save weights if path provided
        if weights_path:
            keras_model.save_weights(weights_path)
            print(f"\nModel weights saved to {weights_path}")
        
        return history