import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tflite_runtime.interpreter as tflite

# Import necessary variables and classes
from preprocess_data_conv import (
    Config, LandmarkIndices, PreprocessLayer, load_compressed
)
from train_conv import build_model as get_model

# Create instances to access constants
config = Config()
landmarks = LandmarkIndices()

# Map old names to new values for compatibility
N_ROWS = config.N_ROWS
N_DIMS = config.N_DIMS
N_COLS = landmarks.n_cols
NUM_CLASSES = config.NUM_CLASSES
INPUT_SIZE = config.INPUT_SIZE


# TFLite model for submission
class TFLiteModel(tf.Module):
    def __init__(self, model):
        super(TFLiteModel, self).__init__()

        # Load the feature generation and main models
        self.preprocess_layer = PreprocessLayer()
        self.model = model
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, N_ROWS, N_DIMS], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs):
        # Preprocess Data
        x, non_empty_frame_idxs = self.preprocess_layer(inputs)
        # Add Batch Dimension
        x = tf.expand_dims(x, axis=0)
        non_empty_frame_idxs = tf.expand_dims(non_empty_frame_idxs, axis=0)
        # Make Prediction
        outputs = self.model({ 'frames': x, 'non_empty_frame_idxs': non_empty_frame_idxs })
        # Squeeze Output 1x250 -> 250
        outputs = tf.squeeze(outputs, axis=0)

        # Return a dictionary with the output tensor
        return {'outputs': outputs}


def convert_tflite(tflite_model, tflite_path, X_train):
    # Concatenate all data for calibration dataset
    # X_train has shape (n_samples, 543, 3) for raw data
    calibration_data = np.concatenate([
        X_train[:1000],
        np.zeros([10, N_ROWS, N_DIMS], dtype=np.float32),
        np.ones([10, N_ROWS, N_DIMS], dtype=np.float32),
    ])

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_concrete_functions([tflite_model.__call__.get_concrete_function()])

    # TFLite requires a batch axis: (543,3) -> (1,543,3). Model outputs shape (250,), it must output shape (1, 250) for inference
    def representative_data_gen():
        for frame_data in calibration_data:
            yield [frame_data]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Try float16 quantization instead of INT8 for better compatibility
    converter.target_spec.supported_types = [tf.float16]
    # Add TF Select operations to support StatelessMultinomial
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    tflite_model_ser = converter.convert()

    # Check size
    model_size_mb = len(tflite_model_ser) / (1024 * 1024)
    print(f'Model size: {model_size_mb:.2f} MB')

    # Save the model
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model_ser)


def load_raw_data(train_csv_path='data/train.csv', train_landmark_dir='data/train_landmark_files', n_samples=1000):
    """Load raw landmark data for calibration"""
    # Read train.csv to get file paths
    train_df = pd.read_csv(train_csv_path)
    
    # Take a sample of files
    sample_df = train_df.sample(n=min(n_samples, len(train_df)), random_state=42)
    
    raw_data = []
    for _, row in sample_df.iterrows():
        file_path = os.path.join(train_landmark_dir, str(row['participant_id']), f"{row['sequence_id']}.parquet")
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            # Convert to numpy array with shape (n_frames, 543, 3)
            frames = df[['x', 'y', 'z']].values.reshape(len(df)//543, 543, 3)
            # Take first frame as sample (TFLite expects single frame input)
            raw_data.append(frames[0])
    
    return np.array(raw_data, dtype=np.float32)


def main():
    # Load raw landmark data for calibration
    print("Loading raw landmark data for calibration...")
    X_train_raw = load_raw_data(n_samples=1000)  # Load 1000 samples for calibration
    print(f"Loaded {len(X_train_raw)} raw samples with shape {X_train_raw.shape}")
    
    # Initialize model
    keras_model = get_model()
    keras_model.load_weights('model_conv.h5')
    # Wrap the keras model with preprocessing using TFLiteModel
    tflite_model = TFLiteModel(keras_model)

    # Convert
    convert_tflite(tflite_model, 'model_conv.tflite', X_train_raw)

    # Verify TFLite model was exported correctly
    # Load the model
    interpreter = tflite.Interpreter('model_conv.tflite')
    interpreter.allocate_tensors()

    # List of found signatures
    found_signatures = list(interpreter.get_signature_list().keys())
    print(f"Available signatures: {found_signatures}")

    if found_signatures:
        # Use the first available signature
        prediction_fn = interpreter.get_signature_runner(found_signatures[0])
        
        # Example prediction with raw data
        output = prediction_fn(inputs=X_train_raw[0])
        print('outputs' in output)
        
        # Verify output matches (there will be some difference due to quantization)
        demo_data = tf.constant(X_train_raw[0:1].astype(np.float32))
        keras_output = tflite_model(demo_data)['outputs'].numpy()
        tflite_output = prediction_fn(inputs=demo_data[0])['outputs']
    else:
        # Use the default invoke method if no signatures
        print("No signatures found, using default invoke method")
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input details: {input_details}")
        print(f"Output details: {output_details}")
        
        # Example prediction with raw data
        interpreter.set_tensor(input_details[0]['index'], X_train_raw[0:1].astype(np.float32))
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details[0]['index'])
        print(f"Raw TFLite output shape: {tflite_output.shape}")
        
        # If output has batch dimension, remove it
        if len(tflite_output.shape) > 1:
            tflite_output = tflite_output[0]
        
        # Verify output matches
        demo_data = tf.constant(X_train_raw[0:1].astype(np.float32))
        keras_output = tflite_model(demo_data)['outputs'].numpy()

    print(f'Keras: {keras_output.shape}, {keras_output.argmax()}, {keras_output.max():.3f}')
    print(f'TFLite: {tflite_output.shape}, {tflite_output.argmax()}, {tflite_output.max():.3f}')
    print(f'DIFFERENCE: {np.abs(keras_output - tflite_output).max():.3f}')

    plt.figure(figsize=(20,4))
    plt.plot(keras_output, alpha=0.50, label='keras', linewidth=4)
    plt.plot(tflite_output, alpha=0.75, label='tflite', linewidth=2)
    plt.vlines(x=[keras_output.argmax(), tflite_output.argmax()], ymin=0, ymax=1, color='red', alpha=0.25, linewidth=5)
    plt.legend()
    plt.show()

    print("Conv1D model TFLite conversion complete!")


if __name__ == "__main__":
    main()