import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tflite_runtime.interpreter as tflite

# Import necessary variables and classes
from preprocess_data_conv import (
    N_ROWS, N_DIMS, N_COLS, NUM_CLASSES, INPUT_SIZE,
    LANDMARK_IDXS_LEFT_DOMINANT0, LANDMARK_IDXS_RIGHT_DOMINANT0,
    LIPS_IDXS0, LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0, LEFT_POSE_IDXS0,
    HAND_IDXS0, LIPS_IDXS, LEFT_HAND_IDXS, RIGHT_HAND_IDXS,
    HAND_IDXS, POSE_IDXS, PreprocessLayer, load_preprocessed_data
)
from train_conv import get_model


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
    calibration_data = np.concatenate([
        X_train[:1000],
        np.zeros([10, INPUT_SIZE, N_COLS, 3], dtype=np.float32),
        np.ones([10, INPUT_SIZE, N_COLS, 3], dtype=np.float32),
    ])

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_concrete_functions([tflite_model.__call__.get_concrete_function()])

    # TFLite requires a batch axis: (543,3) -> (1,543,3). Model outputs shape (250,), it must output shape (1, 250) for inference
    def representative_data_gen():
        for frame_data in calibration_data:
            yield [frame_data]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # TFLite Model requires dynamic range quantization with int8/uint8 activations and int8 weights
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    # The inference is broken if we don't restrict all intermediate operations to int8
    converter._experimental_disable_per_channel = True
    tflite_model_ser = converter.convert()

    # Check size
    model_size_mb = len(tflite_model_ser) / (1024 * 1024)
    print(f'Model size: {model_size_mb:.2f} MB')

    # Save the model
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model_ser)


def main():
    # Load preprocessed data to get X_train for calibration and testing
    X, _, _ = load_preprocessed_data()
    X_train = X  # Use all data for calibration
    
    # Initialize model
    keras_model = get_model()
    keras_model.load_weights('model_conv.h5')
    # Wrap the keras model with preprocessing using TFLiteModel
    tflite_model = TFLiteModel(keras_model)

    # Convert
    convert_tflite(tflite_model, 'model_conv.tflite', X_train)

    # Verify TFLite model was exported correctly
    # Load the model
    interpreter = tflite.Interpreter('model_conv.tflite')

    # List of found signatures
    found_signatures = list(interpreter.get_signature_list().keys())

    # Serving signature
    prediction_fn = interpreter.get_signature_runner("serving_default")

    # Example prediction
    output = prediction_fn(inputs=X_train[0])
    print('outputs' in output)

    # Verify output matches (there will be some difference due to quantization)
    demo_data = tf.constant(X_train[0:1].astype(np.float32))
    keras_output = tflite_model(demo_data)['outputs'].numpy()
    tflite_output = prediction_fn(inputs=demo_data[0])['outputs']

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