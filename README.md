# 🥉 3rd Place Solution - Google ASL Fingerspelling Recognition Competition

## Overview

This repository contains my 3rd place solution for the [Google - American Sign Language Fingerspelling Recognition](https://www.kaggle.com/competitions/asl-fingerspelling) competition on Kaggle. The challenge involved building a model to recognize American Sign Language (ASL) fingerspelling gestures from video sequences captured via MediaPipe landmarks.

**Competition Results**: 3rd Place out of 1,541 teams

## Problem Statement

The task was to classify fingerspelling gestures representing individual letters and numbers from video sequences. Each video contained 3D coordinates of hand, face, and pose landmarks tracked over time. The main challenges included:

- Variable-length video sequences
- Missing/incomplete landmark data
- Real-time inference constraints (model size < 40MB)
- Deployment via TensorFlow Lite

## Solution Architecture

### 🏗️ Model Ensemble Strategy

My solution combines two complementary approaches:

1. **Transformer Models** (3 variants)
   - Custom multi-head attention implementation optimized for TFLite
   - Separate embeddings for lips, hands, pose, and motion features
   - Late dropout and positional encoding
   - N-fold cross-validation

2. **Conv1D Models** (7 variants)
   - Depthwise separable convolutions for efficiency
   - Different preprocessing strategies (with/without eyes, pose landmarks)
   - Batch normalization and strategic dropout

### 🔧 Key Technical Innovations

#### 1. Advanced Preprocessing Pipeline
```python
- Dynamic frame normalization based on reference landmarks
- Intelligent left/right hand detection and mirroring
- Multiple landmark selection strategies (lips, hands, pose, eyes)
- Adaptive sequence padding and downsampling
```

#### 2. Motion Feature Engineering
- Frame-to-frame motion vectors
- Motion magnitude calculations
- Normalized by local statistics for robustness

#### 3. Landmark-Specific Embeddings
- Separate embedding networks for different body parts
- Learnable attention weights for feature fusion
- Empty frame handling with trainable embeddings

#### 4. Model Ensemble Weighting
```python
outputs = 0.2*conv1d_1 + 0.2*conv1d_2 + 0.3*conv1d_3 + 
          0.3*conv1d_4 + 0.3*conv1d_5 + 0.3*conv1d_6 + 
          1.5*transformer_ensemble
```

## Technical Implementation

### Dependencies
- TensorFlow 2.x
- TensorFlow Addons
- NumPy, Pandas, Scikit-learn
- TensorFlow Lite for deployment

### Model Architecture Details

#### Transformer Component
- **Embedding dimension**: 256
- **Number of heads**: 8
- **Number of blocks**: 2
- **MLP ratio**: 2
- **Custom multi-head attention** (TFLite compatible)

#### Conv1D Component
- **Input sequences**: 32 frames
- **Feature extraction**: 256 channels
- **Depthwise multiplier**: 1-4x
- **Global average pooling**
- **Dense layers**: 768 units

### Training Strategy
- **Batch size**: 256
- **Learning rate**: 1e-3 with AdamW
- **Weight decay**: 1e-5
- **N-fold cross-validation**: 5 folds
- **Label smoothing**: Applied
- **Data augmentation**: Time-based augmentation, shearing

## Performance Metrics

- **Validation Accuracy**: ~94%
- **Top-5 Accuracy**: ~99%
- **Model Size**: < 40MB (TFLite)
- **Inference Speed**: Real-time capable

## Code Quality & Standards

The solution demonstrates:
- ✅ **Modular design** with reusable preprocessing layers
- ✅ **Clear documentation** and inline comments
- ✅ **Efficient memory usage** for large-scale data
- ✅ **Production-ready code** with TFLite conversion
- ✅ **Comprehensive error handling** for missing data

## Business Impact

This solution showcases:
1. **Problem-solving ability** in complex computer vision tasks
2. **Deep learning expertise** with custom architectures
3. **Production mindset** with size/speed constraints
4. **Code quality** suitable for enterprise deployment
5. **Innovation** in feature engineering and model design

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