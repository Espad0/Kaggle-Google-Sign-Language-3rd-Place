# 🥉 3rd Place Solution - Google ASL Fingerspelling Recognition Competition

## Overview

This repository contains the training pipeline based on our 3rd place solution for the [Google - Isolated Sign Language](https://www.kaggle.com/competitions/asl-signs/overview) competition on Kaggle. The challenge involved building a model to recognize Isolated Sign Language (ISL) gestures from video sequences captured via MediaPipe landmarks.

## Problem Statement

The competition challenged participants to build a computer vision system capable of recognizing isolated sign language gestures in real-world conditions. 

![Sign Language Example](tv_sign.gif)
![YouTube Sign Language Demo](tv_sign_youtube.gif)

**Technical challenges:**

- Different durations for each gesture
- Missing/incomplete data, frequent landmark occlusions
- Tensorflow Lite models for mobile devices with 40MB size limit
- 250+ gesture classes including letters, numbers, and common signs

## Solution Architecture

We ensemble Transformers and Conv1D architectures, with custom preprocessing for sign language landmarks. This approach captures both temporal patterns (Transformers) and local features (Conv1D), improving robustness and accuracy across diverse signing styles and conditions.

### 📐 Model Architectures

#### Conv1D Model Implementation
```python
# Conv1D Model Architecture - Functional API
inputs = Input(shape=(32, 66, 2), name='frames')
x = Reshape((32, 66*2))(inputs)

# First Conv Block
x = Conv1D(64, 1, strides=1, padding='valid', activation='relu')(x)
x = BatchNormalization()(x)
x = DepthwiseConv1D(3, strides=1, padding='valid', depth_multiplier=1, activation='relu')(x)
x = BatchNormalization()(x)

# Second Conv Block with downsampling
x = Conv1D(64, 1, strides=1, padding='valid', activation='relu')(x)
x = BatchNormalization()(x)
x = DepthwiseConv1D(5, strides=2, padding='valid', depth_multiplier=4, activation='relu')(x)
x = BatchNormalization()(x)

x = MaxPool1D(2, 2)(x)

# Third Conv Block
x = Conv1D(256, 1, strides=1, padding='valid', activation='relu')(x)
x = BatchNormalization()(x)
x = DepthwiseConv1D(3, strides=1, padding='valid', depth_multiplier=1, activation='relu')(x)
x = BatchNormalization()(x)

# Fourth Conv Block with downsampling
x = Conv1D(256, 1, strides=1, padding='valid', activation='relu')(x)
x = BatchNormalization()(x)
x = DepthwiseConv1D(3, strides=2, padding='valid', depth_multiplier=4, activation='relu')(x)
x = BatchNormalization()(x)

# Global pooling and classification
x = GlobalAvgPool1D()(x)
x = Dropout(rate=0.4)(x)

x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(rate=0.4)(x)

x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(rate=0.4)(x)

outputs = Dense(250, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
```

#### Transformer Model Implementation
```python
# Transformer Model Architecture (Functional API)
def build_transformer():
    # Inputs
    frames = Input(shape=(32, 66, 3), name='frames')
    non_empty_frame_idxs = Input(shape=(32,), name='non_empty_frame_idxs')
    
    # Extract x,y coordinates only (drop z)
    x = Lambda(lambda x: x[:, :, :, :2])(frames)
    
    # Split landmarks
    lips = Lambda(lambda x: x[:, :, 0:40, :])(x)      # 40 lip points
    left_hand = Lambda(lambda x: x[:, :, 40:61, :])(x) # 21 hand points
    pose = Lambda(lambda x: x[:, :, 61:66, :])(x)      # 5 pose points
    
    # Landmark-specific embeddings
    lips_embedding = LandmarkEmbedding(256, 'lips')(lips)
    left_hand_embedding = LandmarkEmbedding(256, 'left_hand')(left_hand)
    pose_embedding = LandmarkEmbedding(256, 'pose')(pose)
    
    # Combine embeddings with learnable weights
    x = WeightedAverage()([lips_embedding, left_hand_embedding, pose_embedding])
    
    # Add positional encoding
    x = PositionalEmbedding()(x, non_empty_frame_idxs)
    
    # Transformer blocks (3 blocks, 8 heads each)
    for i in range(3):
        # Multi-head attention
        attn = MultiHeadAttention(
            d_model=256, 
            num_heads=8
        )(x, attention_mask=mask)
        x = Add()([x, attn])  # Residual connection
        
        # Feed-forward network
        ffn = Sequential([
            Dense(256 * 4, activation='gelu'),
            Dropout(0.1),
            Dense(256)
        ])(x)
        x = Add()([x, ffn])  # Residual connection
    
    # Pooling with mask
    x = MaskedGlobalAveragePooling1D()(x, mask)
    
    # Classification head
    x = Dropout(0.5)(x)
    outputs = Dense(250, activation='softmax')(x)
    
    return Model(inputs=[frames, non_empty_frame_idxs], outputs=outputs)
```

### 🏗️ Model Ensemble Strategy

Our 3rd place solution leverages a sophisticated multi-model ensemble combining custom Transformer and Conv1D architectures:

1. **Transformer Models** (3 variants with 5-fold cross-validation)
   - Custom MultiHeadAttention implementation specifically designed for TFLite compatibility
   - Separate embeddings for lips (256 units), hands (256 units), pose (256 units), and motion (128 units)
   - Late dropout with adaptive scheduling (starts at epoch 80)
   - Positional encoding normalized by frame indices

2. **Conv1D Models** (7 specialized variants)
   - Depthwise separable convolutions with depth multipliers 1-4x
   - Multiple preprocessing strategies:
     - **V0**: Base model with lips (40 points) + hands (21 points each)
     - **V0Pose**: Adds 12 pose landmarks for body context
     - **V0Eyes**: Incorporates 32 eye landmarks for facial expressions
     - **V0EyesSparse**: Uses every 2nd eye landmark for efficiency
   - Strategic batch normalization after each conv layer

### 🔧 Key Technical Implementations

#### 1. Advanced Preprocessing Pipeline
```python
# Landmark extraction strategy (543 total landmarks -> 106 selected)
LIPS: 40 carefully selected mouth landmarks
LEFT_HAND: 21 MediaPipe hand keypoints
RIGHT_HAND: 21 MediaPipe hand keypoints  
POSE: 8 upper body joints (shoulders, elbows, wrists, hips)
EYES: 32 eye contour points (16 per eye)

# Hand dominance detection and mirroring
- Automatic detection of dominant hand per video
- Mirror transformation for left-handed signers
- Ensures consistent feature representation
```

#### 2. Intelligent Frame Sampling
- **Fixed 32-frame sequences** for consistent input size
- **Adaptive downsampling** for videos > 32 frames:
  - Random frame selection weighted by hand visibility
  - Preserves frames with clear hand gestures
- **Smart padding** for videos < 32 frames:
  - Edge padding with random offset to prevent overfitting
  - NaN masking for missing landmarks

#### 3. Motion Feature Engineering
```python
# Compute frame-to-frame differences
motion = frames[t] - frames[t-1]
# Add motion magnitude as additional channel
motion_dist = sqrt(mean(motion^2))
# 106 landmarks × 3 channels (dx, dy, magnitude)
```

#### 4. Landmark-Specific Embeddings with Attention
- Each landmark group has dedicated embedding network
- **Learnable soft attention weights** combine features:
  ```python
  weights = softmax([w_lips, w_hands, w_pose])
  combined = weights[0]*lips + weights[1]*hands + weights[2]*pose
  ```
- Empty landmark handling with trainable embeddings

#### 5. Test-Time Augmentation (TTA)
```python
# Each Conv1D model uses dual preprocessing at inference:
x = preprocess_layer[0](inputs)  # Standard preprocessing
x1 = preprocess_layer[1](inputs) # With random frame sampling (r_long=True)
outputs = 0.5*model(x) + 0.5*model(x1)  # Average predictions

# TTA strategies:
- Random frame sampling for long videos
- Random padding offsets for short videos
- Ensures robust predictions across video variations
```

#### 6. Optimized Model Ensemble
```python
# Final ensemble with carefully tuned weights
outputs = 0.2*model_96frames +      # Long sequence model
          0.2*model_32frames +       # Base Conv1D
          0.3*model_32pose +         # With pose landmarks
          0.3*model_32eyes +         # With eye tracking
          0.3*model_frankv2 +        # Enhanced preprocessing
          0.3*model_mu0 +            # Zero-mean normalization
          1.5*transformer_ensemble   # Main transformer models
```

### 📊 Data Augmentation Strategies

#### Training-Time Augmentations
1. **Hand Mirroring**
   - Automatic left/right hand detection per video
   - Horizontal flip transformation matrix for left-handed signers
   - Ensures model learns both hand orientations

2. **Frame-Level Augmentations**
   - **Random padding offsets**: For videos < 32 frames, random positioning within padded sequence
   - **Dynamic frame sampling**: Weighted selection based on hand visibility for videos > 32 frames
   - **Late dropout scheduling**: Starts at epoch 80 to prevent early overfitting

3. **Normalization Strategies**
   - **Per-video normalization**: Using reference landmarks (shoulders, nose, eyes)
   - **Zero-mean variants**: Some models trained with μ=0 normalization
   - **Local statistics**: Separate normalization for each landmark group

4. **CutMix Augmentation** (Custom Implementation)
   - Mixed samples from different sign language classes
   - **Weighted label assignment**:
     - Hand landmarks: 0.7 weight (primary signal)
     - Other landmarks (lips, pose): 0.3 weight (context)
   - Helps model learn robust features across different signers

#### Test-Time Augmentations (TTA)
- **Dual preprocessing paths** with 50/50 averaging
- **Random frame sampling** (r_long=True) vs deterministic sampling
- **Multiple landmark configurations** (with/without eyes, sparse landmarks)

## Repository Structure

```
ISL/
├── transformer-conv1d-isl-submission-n-fold-and-conv1.ipynb  # Main solution notebook
├── README.md                                                   # This file
└── model.tflite                                               # Deployed model (generated)
```

## How to Run

1. Clone the repository
2. Install dependencies: `pip install tensorflow tensorflow-addons pandas numpy`
3. Open the Jupyter notebook
4. Run all cells to reproduce the solution

## Future Improvements

- Experiment with vision transformers (ViT)
- Implement knowledge distillation for smaller models
- Add temporal convolutions for better motion modeling
- Explore self-supervised pretraining on unlabeled videos

## Contact

Feel free to reach out if you have questions about the implementation or would like to discuss the approach in more detail.

---

*This solution demonstrates practical deep learning skills applied to a challenging real-world problem, achieving top-tier results in a competitive environment.*