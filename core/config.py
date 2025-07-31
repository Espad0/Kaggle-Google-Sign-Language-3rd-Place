"""
Shared configuration classes for all models.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class DataConfig:
    """Central configuration for data preprocessing."""
    # Data dimensions
    N_ROWS: int = 543  # landmarks per frame
    N_DIMS: int = 3
    DIM_NAMES: List[str] = None
    
    # Model parameters
    NUM_CLASSES: int = 250
    INPUT_SIZE: int = 32
    MASK_VAL: int = 4237
    SEED: int = 42
    
    # Conv-specific
    HAND_THRESHOLD: float = 0.60
    CLIP_RANGE: Tuple[float, float] = (-10.0, 10.0)
    MIN_STD: float = 0.01
    
    # Training settings
    BATCH_ALL_SIGNS_N: int = 2
    BATCH_SIZE: int = 128
    
    # Visualization
    SHOW_PLOTS: bool = False
    IS_INTERACTIVE: bool = True
    
    def __post_init__(self):
        if self.DIM_NAMES is None:
            self.DIM_NAMES = ['x', 'y', 'z']
    
    @property
    def verbose(self) -> int:
        return 1 if self.IS_INTERACTIVE else 2


@dataclass
class TransformerConfig:
    """Configuration specific to Transformer model."""
    # Model architecture
    layer_norm_eps: float = 1e-6
    lips_units: int = 384
    hands_units: int = 384
    pose_units: int = 384
    units: int = 512
    num_blocks: int = 2
    mlp_ratio: int = 2
    num_heads: int = 8
    
    # Dropout rates
    embedding_dropout: float = 0.00
    mlp_dropout_ratio: float = 0.30
    classifier_dropout_ratio: float = 0.10


@dataclass  
class Conv1DConfig:
    """Configuration specific to Conv1D model."""
    # Model architecture
    filters: List[int] = None
    kernel_sizes: List[int] = None
    dropout_rate: float = 0.5
    
    def __post_init__(self):
        if self.filters is None:
            self.filters = [64, 64, 256, 256]
        if self.kernel_sizes is None:
            self.kernel_sizes = [3, 5, 3, 3]


@dataclass
class TrainingConfig:
    """Common training configuration."""
    # Training parameters
    train_model: bool = True
    use_validation: bool = False
    show_plots: bool = False
    
    # Optimization
    n_epochs: int = 100
    lr_max: float = 1e-3
    n_warmup_epochs: int = 0
    wd_ratio: float = 0.05
    warmup_method: str = 'log'
    label_smoothing: float = 0.25
    
    # Data
    generate_data: bool = False
    batch_size: int = 64
    
    @property
    def verbose(self) -> int:
        return 1 if self.show_plots else 2