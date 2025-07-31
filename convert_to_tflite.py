"""
TFLite Conversion Module for Sign Language Recognition

This module handles the conversion of the trained transformer model to TFLite format
for deployment and submission.
"""

import os
import zipfile
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import tflite_runtime.interpreter as tflite

from preprocess_data import (
    Config,
    LandmarkIndices,
    PreprocessLayer,
    load_relevant_data_subset
)
from train_transformer import get_model


# Create config instances
config = Config()
landmarks = LandmarkIndices()

# Map constants for compatibility
N_ROWS = config.N_ROWS
N_DIMS = config.N_DIMS
N_COLS = landmarks.N_COLS
NUM_CLASSES = config.NUM_CLASSES
INPUT_SIZE = config.INPUT_SIZE


# ======================== TFLite Model Wrapper ========================
class TFLiteModel(tf.Module):
    """TFLite model wrapper that includes preprocessing."""
    
    def __init__(self, model, preprocess_layer: Optional[PreprocessLayer] = None):
        super(TFLiteModel, self).__init__()
        
        # Load the feature generation and main models
        if preprocess_layer is None:
            print("PREPARING PREPROCESS LAYER FOR TFLITE MODEL") 
            preprocess_layer = PreprocessLayer()
        self.preprocess_layer = preprocess_layer
        self.model = model
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, N_ROWS, N_DIMS], dtype=tf.float32, name='inputs')
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


# ======================== Conversion Functions ========================
def convert_to_tflite(model: tf.keras.Model, 
                     output_path: str = 'model.tflite',
                     preprocess_layer: Optional[PreprocessLayer] = None) -> None:
    """Convert Keras model to TFLite format."""
    # Wrap model with preprocessing
    tflite_keras_model = TFLiteModel(model, preprocess_layer)
    
    # Create Model Converter
    keras_model_converter = tf.lite.TFLiteConverter.from_keras_model(tflite_keras_model)
    
    # Convert Model
    tflite_model = keras_model_converter.convert()
    
    # Write Model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {output_path}")
    print(f"Model size: {len(tflite_model) / (1024 * 1024):.2f} MB")


def create_submission_zip(tflite_path: str = 'model.tflite', 
                         zip_path: str = 'submission.zip') -> None:
    """Create submission zip file with TFLite model."""
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(tflite_path, 'model.tflite')
    
    print(f"Model zipped successfully as {zip_path}")


def verify_tflite_model(tflite_path: str = 'model.tflite',
                       train_csv_path: str = 'data/train.csv',
                       sample_idx: int = 5) -> None:
    """Verify TFLite model can be loaded and used for prediction."""
    # Load train data to get a sample
    train_df = pd.read_csv(train_csv_path)
    train_df['file_path'] = train_df['path'].apply(lambda x: f'./{x}')
    train_df['sign_ord'] = train_df['sign'].astype('category').cat.codes
    
    # Create translation dictionaries
    ORD2SIGN = train_df[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()
    
    # Load sample data
    demo_raw_data = load_relevant_data_subset(train_df['file_path'].values[sample_idx])
    print(f'demo_raw_data shape: {demo_raw_data.shape}, dtype: {demo_raw_data.dtype}')
    
    # Test with TFLiteModel first (before TFLite conversion)
    print("\nTesting with TFLiteModel (pre-conversion)...")
    preprocess_layer = PreprocessLayer()
    
    # Load model weights
    print("Loading model weights...")
    stats = {
        'lips': (np.zeros((40, 2)), np.ones((40, 2))),
        'left_hand': (np.zeros((21, 2)), np.ones((21, 2))),
        'pose': (np.zeros((5, 2)), np.ones((5, 2)))
    }
    
    # Try to load actual stats if available
    try:
        import pickle
        with open('landmark_stats.pkl', 'rb') as f:
            stats = pickle.load(f)
        print("Loaded saved landmark statistics")
    except:
        print("Using default landmark statistics")
    
    model = get_model(stats)
    model.load_weights('model.h5')
    
    tflite_keras_model = TFLiteModel(model, preprocess_layer)
    demo_output = tflite_keras_model(demo_raw_data)["outputs"]
    print(f'demo_output shape: {demo_output.shape}, dtype: {demo_output.dtype}')
    demo_prediction = demo_output.numpy().argmax()
    print(f'demo_prediction: {demo_prediction}, label: {ORD2SIGN.get(demo_prediction)}')
    print(f'ground truth: {train_df.iloc[sample_idx]["sign_ord"]}, '
          f'label: {train_df.iloc[sample_idx]["sign"]}')
    
    # Test with actual TFLite model
    print("\nTesting with TFLite interpreter...")
    interpreter = tflite.Interpreter(tflite_path)
    found_signatures = list(interpreter.get_signature_list().keys())
    print(f"Available signatures: {found_signatures}")
    
    if found_signatures:
        prediction_fn = interpreter.get_signature_runner("serving_default")
        output = prediction_fn(inputs=demo_raw_data)
        sign = output['outputs'].argmax()
    else:
        # Fallback to direct invoke
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], demo_raw_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        sign = output.argmax()
    
    print(f"PRED : {ORD2SIGN.get(sign)}, [{sign}]")
    print(f"TRUE : {train_df.iloc[sample_idx]['sign']}, "
          f"[{train_df.iloc[sample_idx]['sign_ord']}]")


# ======================== Main Conversion Pipeline ========================
def main(model_weights_path: str = 'model.h5',
         output_tflite_path: str = 'model.tflite',
         output_zip_path: str = 'submission.zip',
         verify: bool = True):
    """Main conversion pipeline."""
    # Load landmark statistics
    print("Loading landmark statistics...")
    stats = {
        'lips': (np.zeros((40, 2)), np.ones((40, 2))),
        'left_hand': (np.zeros((21, 2)), np.ones((21, 2))),
        'pose': (np.zeros((5, 2)), np.ones((5, 2)))
    }
    
    # Try to load actual stats if available
    try:
        # First try compressed format
        from preprocess_data import load_compressed
        X_train = load_compressed('X_train.zip')
        
        # Calculate stats
        from preprocess_data import calculate_mean_std_stats
        stats = calculate_mean_std_stats(X_train, landmarks, config)
        print("Calculated landmark statistics from training data")
    except:
        try:
            # Try loading saved stats
            import pickle
            with open('landmark_stats.pkl', 'rb') as f:
                stats = pickle.load(f)
            print("Loaded saved landmark statistics")
        except:
            print("Warning: Using default landmark statistics")
    
    # Build model
    print("\nBuilding model architecture...")
    model = get_model(stats)
    
    # Load weights
    print(f"Loading model weights from {model_weights_path}...")
    model.load_weights(model_weights_path)
    
    # Convert to TFLite
    print("\nConverting to TFLite...")
    preprocess_layer = PreprocessLayer()
    convert_to_tflite(model, output_tflite_path, preprocess_layer)
    
    # Create submission zip
    create_submission_zip(output_tflite_path, output_zip_path)
    
    # Verify if requested
    if verify:
        print("\nVerifying TFLite model...")
        verify_tflite_model(output_tflite_path)
    
    print("\nConversion complete!")


# ======================== Utility Functions ========================
def save_landmark_stats(stats: dict, output_path: str = 'landmark_stats.pkl'):
    """Save landmark statistics for later use."""
    import pickle
    with open(output_path, 'wb') as f:
        pickle.dump(stats, f)
    print(f"Landmark statistics saved to {output_path}")


def test_tflite_size_optimization(model_weights_path: str = 'model.h5'):
    """Test different TFLite optimization strategies."""
    # Load model
    stats = {
        'lips': (np.zeros((40, 2)), np.ones((40, 2))),
        'left_hand': (np.zeros((21, 2)), np.ones((21, 2))),
        'pose': (np.zeros((5, 2)), np.ones((5, 2)))
    }
    model = get_model(stats)
    model.load_weights(model_weights_path)
    
    preprocess_layer = PreprocessLayer()
    tflite_keras_model = TFLiteModel(model, preprocess_layer)
    
    # Test different optimization strategies
    strategies = [
        ("No optimization", [], None),
        ("Default optimization", [tf.lite.Optimize.DEFAULT], None),
        ("Float16 quantization", [tf.lite.Optimize.DEFAULT], [tf.float16]),
        ("Dynamic range quantization", [tf.lite.Optimize.DEFAULT], None),
    ]
    
    for name, optimizations, supported_types in strategies:
        print(f"\n{name}:")
        converter = tf.lite.TFLiteConverter.from_keras_model(tflite_keras_model)
        converter.optimizations = optimizations
        if supported_types:
            converter.target_spec.supported_types = supported_types
        
        try:
            tflite_model = converter.convert()
            size_mb = len(tflite_model) / (1024 * 1024)
            print(f"  Size: {size_mb:.2f} MB")
            
            # Save for testing
            with open(f'model_{name.lower().replace(" ", "_")}.tflite', 'wb') as f:
                f.write(tflite_model)
        except Exception as e:
            print(f"  Failed: {str(e)}")


if __name__ == "__main__":
    # Run main conversion
    main(
        model_weights_path='model.h5',
        output_tflite_path='model.tflite',
        output_zip_path='submission.zip',
        verify=True
    )