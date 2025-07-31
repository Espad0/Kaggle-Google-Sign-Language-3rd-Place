"""
Training callbacks for sign language models.
"""
import math
import numpy as np
import tensorflow as tf
from typing import List


class WeightDecayCallback(tf.keras.callbacks.Callback):
    """Update weight decay with learning rate."""
    def __init__(self, wd_ratio: float = 0.05):
        self.wd_ratio = wd_ratio
    
    def on_epoch_begin(self, epoch, logs=None):
        self.model.optimizer.weight_decay = self.model.optimizer.learning_rate * self.wd_ratio
        current_lr = self.model.optimizer.learning_rate.numpy()
        current_wd = self.model.optimizer.weight_decay.numpy()
        print(f'LR: {current_lr:.2e}, WD: {current_wd:.2e}')


def create_lr_schedule(n_epochs: int, 
                      lr_max: float = 1e-3,
                      n_warmup_epochs: int = 0,
                      warmup_method: str = 'log',
                      num_cycles: float = 0.5) -> List[float]:
    """
    Create learning rate schedule with warmup and cosine decay.
    Used by both Transformer and Conv1D models.
    """
    schedule = []
    
    for step in range(n_epochs):
        if step < n_warmup_epochs:
            # Warmup phase
            if warmup_method == 'log':
                lr = lr_max * 0.10 ** (n_warmup_epochs - step)
            else:
                lr = lr_max * 2 ** -(n_warmup_epochs - step)
        else:
            # Cosine decay phase
            progress = float(step - n_warmup_epochs) / float(max(1, n_epochs - n_warmup_epochs))
            lr = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr_max
        
        schedule.append(lr)
    
    return schedule


def get_callbacks(training_config) -> List[tf.keras.callbacks.Callback]:
    """Get standard callbacks for training."""
    # Create learning rate schedule
    lr_schedule = create_lr_schedule(
        n_epochs=training_config.n_epochs,
        lr_max=training_config.lr_max,
        n_warmup_epochs=training_config.n_warmup_epochs,
        warmup_method=training_config.warmup_method
    )
    
    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(
            lambda step: lr_schedule[step], 
            verbose=1
        ),
        WeightDecayCallback(training_config.wd_ratio),
    ]
    
    return callbacks