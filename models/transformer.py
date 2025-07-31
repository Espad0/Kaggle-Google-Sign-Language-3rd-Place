"""
Transformer model architecture for sign language recognition.
"""
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, Optional

from core import DataConfig, TransformerConfig, LandmarkIndices
from .base import BaseModel


def scaled_dot_product(q, k, v, softmax, attention_mask):
    """Scaled dot product attention."""
    # Calculate Q . K(transpose)
    qkt = tf.matmul(q, k, transpose_b=True)
    # Calculate scaling factor
    dk = tf.math.sqrt(tf.cast(q.shape[-1], dtype=tf.float32))
    scaled_qkt = qkt / dk
    softmax = softmax(scaled_qkt, mask=attention_mask)
    
    z = tf.matmul(softmax, v)
    return z


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention layer."""
    
    def __init__(self, d_model, num_of_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_of_heads = num_of_heads
        self.depth = d_model // num_of_heads
        self.wq = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wk = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wv = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wo = tf.keras.layers.Dense(d_model)
        self.softmax = tf.keras.layers.Softmax()
        
    def call(self, x, attention_mask):
        multi_attn = []
        for i in range(self.num_of_heads):
            Q = self.wq[i](x)
            K = self.wk[i](x)
            V = self.wv[i](x)
            multi_attn.append(scaled_dot_product(Q, K, V, self.softmax, attention_mask))
            
        multi_head = tf.concat(multi_attn, axis=-1)
        multi_head_attention = self.wo(multi_head)
        return multi_head_attention


class TransformerBlock(tf.keras.Model):
    """Full Transformer model with multiple blocks."""
    
    def __init__(self, num_blocks, config: TransformerConfig):
        super(TransformerBlock, self).__init__(name='transformer')
        self.num_blocks = num_blocks
        self.config = config
    
    def build(self, input_shape):
        self.ln_1s = []
        self.mhas = []
        self.ln_2s = []
        self.mlps = []
        
        # Make Transformer Blocks
        for i in range(self.num_blocks):
            # Multi Head Attention
            self.mhas.append(MultiHeadAttention(self.config.units, self.config.num_heads))
            # Multi Layer Perceptron
            self.mlps.append(tf.keras.Sequential([
                tf.keras.layers.Dense(
                    self.config.units * self.config.mlp_ratio, 
                    activation=tf.keras.activations.gelu,
                    kernel_initializer=tf.keras.initializers.glorot_uniform()
                ),
                tf.keras.layers.Dropout(self.config.mlp_dropout_ratio),
                tf.keras.layers.Dense(
                    self.config.units,
                    kernel_initializer=tf.keras.initializers.he_uniform()
                ),
            ]))
    
    def call(self, x, attention_mask):
        # Iterate input over transformer blocks
        for mha, mlp in zip(self.mhas, self.mlps):
            x = x + mha(x, attention_mask)
            x = x + mlp(x)
        return x


class LandmarkEmbedding(tf.keras.Model):
    """Embedding layer for individual landmark types."""
    
    def __init__(self, units, name):
        super(LandmarkEmbedding, self).__init__(name=f'{name}_embedding')
        self.units = units
        
    def build(self, input_shape):
        # Embedding for missing landmark in frame, initialized with zeros
        self.empty_embedding = self.add_weight(
            name=f'{self.name}_empty_embedding',
            shape=[self.units],
            initializer=tf.keras.initializers.constant(0.0),
        )
        # Embedding
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.units, 
                name=f'{self.name}_dense_1', 
                use_bias=False,
                kernel_initializer=tf.keras.initializers.glorot_uniform()
            ),
            tf.keras.layers.Activation(tf.keras.activations.gelu),
            tf.keras.layers.Dense(
                self.units, 
                name=f'{self.name}_dense_2', 
                use_bias=False,
                kernel_initializer=tf.keras.initializers.he_uniform()
            ),
        ], name=f'{self.name}_dense')

    def call(self, x):
        return tf.where(
            # Checks whether landmark is missing in frame
            tf.reduce_sum(x, axis=2, keepdims=True) == 0,
            # If so, the empty embedding is used
            self.empty_embedding,
            # Otherwise the landmark data is embedded
            self.dense(x),
        )


class Embedding(tf.keras.Model):
    """Main embedding layer combining all landmarks."""
    
    def __init__(self, config: TransformerConfig, data_config: DataConfig):
        super(Embedding, self).__init__()
        self.config = config
        self.data_config = data_config
        
    def build(self, input_shape):
        # Positional Embedding, initialized with zeros
        self.positional_embedding = tf.keras.layers.Embedding(
            self.data_config.INPUT_SIZE + 1, 
            self.config.units, 
            embeddings_initializer=tf.keras.initializers.constant(0.0)
        )
        # Embedding layer for Landmarks
        self.lips_embedding = LandmarkEmbedding(self.config.lips_units, 'lips')
        self.left_hand_embedding = LandmarkEmbedding(self.config.hands_units, 'left_hand')
        self.pose_embedding = LandmarkEmbedding(self.config.pose_units, 'pose')
        # Landmark Weights
        self.landmark_weights = tf.Variable(
            tf.zeros([3], dtype=tf.float32), 
            name='landmark_weights'
        )
        # Fully Connected Layers for combined landmarks
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.config.units, 
                name='fully_connected_1', 
                use_bias=False,
                kernel_initializer=tf.keras.initializers.glorot_uniform()
            ),
            tf.keras.layers.Activation(tf.keras.activations.gelu),
            tf.keras.layers.Dense(
                self.config.units, 
                name='fully_connected_2', 
                use_bias=False,
                kernel_initializer=tf.keras.initializers.he_uniform()
            ),
        ], name='fc')

    def call(self, lips0, left_hand0, pose0, non_empty_frame_idxs, training=False):
        # Lips
        lips_embedding = self.lips_embedding(lips0)
        # Left Hand
        left_hand_embedding = self.left_hand_embedding(left_hand0)
        # Pose
        pose_embedding = self.pose_embedding(pose0)
        
        # Merge Embeddings of all landmarks with mean pooling
        x = tf.stack((
            lips_embedding, left_hand_embedding, pose_embedding,
        ), axis=3)
        x = x * tf.nn.softmax(self.landmark_weights)
        x = tf.reduce_sum(x, axis=3)
        
        # Fully Connected Layers
        x = self.fc(x)
        
        # Add Positional Embedding
        max_frame_idxs = tf.clip_by_value(
            tf.reduce_max(non_empty_frame_idxs, axis=1, keepdims=True),
            1,
            np.PINF,
        )
        normalised_non_empty_frame_idxs = tf.where(
            tf.math.equal(non_empty_frame_idxs, -1.0),
            self.data_config.INPUT_SIZE,
            tf.cast(
                non_empty_frame_idxs / max_frame_idxs * self.data_config.INPUT_SIZE,
                tf.int32,
            ),
        )
        x = x + self.positional_embedding(normalised_non_empty_frame_idxs)
        
        return x


class TransformerModel(BaseModel):
    """Transformer model for sign language recognition."""
    
    def __init__(self, data_config: DataConfig = None, model_config: TransformerConfig = None):
        if data_config is None:
            data_config = DataConfig()
        if model_config is None:
            model_config = TransformerConfig()
        super().__init__(data_config, model_config)
        self.landmarks = LandmarkIndices()
        
    def build_model(self, stats: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None) -> tf.keras.Model:
        """Build the transformer model architecture."""
        if stats is None:
            # Default statistics if not provided
            stats = {
                'lips': (np.zeros((40, 2)), np.ones((40, 2))),
                'left_hand': (np.zeros((21, 2)), np.ones((21, 2))),
                'pose': (np.zeros((5, 2)), np.ones((5, 2)))
            }
        
        # Extract statistics
        LIPS_MEAN, LIPS_STD = stats['lips']
        LEFT_HANDS_MEAN, LEFT_HANDS_STD = stats['left_hand']
        POSE_MEAN, POSE_STD = stats['pose']
        
        # Inputs
        frames = tf.keras.layers.Input(
            [self.data_config.INPUT_SIZE, self.landmarks.N_COLS, self.data_config.N_DIMS], 
            dtype=tf.float32, name='frames'
        )
        non_empty_frame_idxs = tf.keras.layers.Input(
            [self.data_config.INPUT_SIZE], 
            dtype=tf.float32, name='non_empty_frame_idxs'
        )
        
        # Padding Mask
        mask0 = tf.cast(tf.math.not_equal(non_empty_frame_idxs, -1), tf.float32)
        mask0 = tf.expand_dims(mask0, axis=2)
        
        # Random Frame Masking (for training robustness)
        mask = tf.where(
            (tf.random.uniform(tf.shape(mask0)) > 0.25) & tf.math.not_equal(mask0, 0.0),
            1.0,
            0.0,
        )
        # Correct Samples Which are all masked now...
        mask = tf.where(
            tf.math.equal(tf.reduce_sum(mask, axis=[1, 2], keepdims=True), 0.0),
            mask0,
            mask,
        )
        
        # Extract x,y coordinates only (drop z)
        x = frames
        x = tf.slice(x, [0, 0, 0, 0], [-1, self.data_config.INPUT_SIZE, self.landmarks.N_COLS, 2])
        
        # LIPS
        lips = tf.slice(x, [0, 0, self.landmarks.LIPS_START, 0], [-1, self.data_config.INPUT_SIZE, 40, 2])
        lips = tf.where(
            tf.math.equal(lips, 0.0),
            0.0,
            (lips - LIPS_MEAN) / LIPS_STD,
        )
        
        # LEFT HAND
        left_hand = tf.slice(x, [0, 0, 40, 0], [-1, self.data_config.INPUT_SIZE, 21, 2])
        left_hand = tf.where(
            tf.math.equal(left_hand, 0.0),
            0.0,
            (left_hand - LEFT_HANDS_MEAN) / LEFT_HANDS_STD,
        )
        
        # POSE
        pose = tf.slice(x, [0, 0, 61, 0], [-1, self.data_config.INPUT_SIZE, 5, 2])
        pose = tf.where(
            tf.math.equal(pose, 0.0),
            0.0,
            (pose - POSE_MEAN) / POSE_STD,
        )
        
        # Flatten
        lips = tf.reshape(lips, [-1, self.data_config.INPUT_SIZE, 40 * 2])
        left_hand = tf.reshape(left_hand, [-1, self.data_config.INPUT_SIZE, 21 * 2])
        pose = tf.reshape(pose, [-1, self.data_config.INPUT_SIZE, 5 * 2])
        
        # Embedding
        x = Embedding(self.model_config, self.data_config)(lips, left_hand, pose, non_empty_frame_idxs)
        
        # Encoder Transformer Blocks
        x = TransformerBlock(self.model_config.num_blocks, self.model_config)(x, mask)
        
        # Pooling
        x = tf.reduce_sum(x * mask, axis=1) / tf.reduce_sum(mask, axis=1)
        
        # Classifier Dropout
        x = tf.keras.layers.Dropout(self.model_config.classifier_dropout_ratio)(x)
        
        # Classification Layer
        x = tf.keras.layers.Dense(
            self.data_config.NUM_CLASSES, 
            activation=tf.keras.activations.softmax,
            kernel_initializer=tf.keras.initializers.glorot_uniform()
        )(x)
        
        outputs = x
        
        # Create Tensorflow Model
        model = tf.keras.models.Model(inputs=[frames, non_empty_frame_idxs], outputs=outputs)
        
        self._model = model
        return model