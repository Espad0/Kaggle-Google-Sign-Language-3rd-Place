"""
TFLite conversion functions.
"""
import os
import zipfile
from typing import Union, Optional
import numpy as np
import pandas as pd
import tensorflow as tf

from core import DataConfig, load_parquet_landmarks, load_metadata
from processing import PreprocessLayer
from .wrapper import TFLiteModel


def convert_model_to_tflite(model: Union[tf.keras.Model, 'BaseModel'], 
                           output_path: str = 'outputs/model.tflite',
                           model_type: str = 'transformer',
                           preprocess_layer: Optional[PreprocessLayer] = None,
                           use_optimization: bool = True) -> None:
    """Convert Keras model to TFLite format."""
    # Extract Keras model if needed
    if hasattr(model, 'model'):
        keras_model = model.model
    else:
        keras_model = model
    
    # Wrap model with preprocessing
    tflite_keras_model = TFLiteModel(keras_model, preprocess_layer, model_type=model_type)
    
    # Create Model Converter
    if model_type == 'conv1d' and use_optimization:
        # Conv1D model uses concrete functions and optimization
        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [tflite_keras_model.__call__.get_concrete_function()]
        )
        
        # Add optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, 
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.allow_custom_ops = True
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
    else:
        # Transformer model uses standard conversion
        converter = tf.lite.TFLiteConverter.from_keras_model(tflite_keras_model)
        
        if use_optimization:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert Model
    print("Converting model to TFLite...")
    tflite_model = converter.convert()
    
    # Write Model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {output_path}")
    print(f"Model size: {len(tflite_model) / (1024 * 1024):.2f} MB")


def create_submission_zip(tflite_path: str = 'outputs/model.tflite', 
                         zip_path: str = 'outputs/submission.zip') -> None:
    """Create submission zip file with TFLite model."""
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(tflite_path, 'model.tflite')
    
    print(f"Model zipped successfully as {zip_path}")


def load_calibration_data(n_samples: int = 1000) -> np.ndarray:
    """Load raw landmark data for Conv1D calibration."""
    config = DataConfig()
    
    # Load metadata
    train_df, _, _ = load_metadata()
    
    # Take a sample of files
    sample_df = train_df.sample(n=min(n_samples, len(train_df)), random_state=config.SEED)
    
    raw_data = []
    for _, row in sample_df.iterrows():
        file_path = row['file_path']
        if os.path.exists(file_path):
            # Load parquet file
            data = load_parquet_landmarks(file_path)
            # Take first frame as sample (TFLite expects single frame input)
            if len(data) > 0:
                raw_data.append(data[0])
    
    return np.array(raw_data, dtype=np.float32)


def verify_tflite_model(tflite_path: str = 'outputs/model.tflite',
                       sample_idx: int = 5) -> None:
    """Verify TFLite model can be loaded and used for prediction."""
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        print("Warning: tflite_runtime not available, skipping verification")
        return
    
    # Load metadata
    train_df, _, ord2sign = load_metadata()
    
    # Load sample data
    demo_raw_data = load_parquet_landmarks(train_df['file_path'].values[sample_idx])
    print(f'demo_raw_data shape: {demo_raw_data.shape}, dtype: {demo_raw_data.dtype}')
    
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
    
    print(f"PRED : {ord2sign.get(sign)}, [{sign}]")
    print(f"TRUE : {train_df.iloc[sample_idx]['sign']}, "
          f"[{train_df.iloc[sample_idx]['sign_ord']}]")