"""
Model evaluation functions.
"""
from typing import Dict, Union
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.metrics
import scipy.special

from core import DataConfig


def evaluate_model(model: Union[tf.keras.Model, 'BaseModel'], 
                  X_val: np.ndarray, 
                  y_val: np.ndarray,
                  frames_val: np.ndarray) -> Dict:
    """Evaluate model and return metrics."""
    # Extract Keras model if needed
    if hasattr(model, 'model'):
        keras_model = model.model
    else:
        keras_model = model
    
    # Get predictions
    y_pred = keras_model.predict(
        {'frames': X_val, 'non_empty_frame_idxs': frames_val}, 
        verbose=2
    )
    y_pred_classes = y_pred.argmax(axis=1)
    
    # Calculate metrics
    metrics = {
        'accuracy': sklearn.metrics.accuracy_score(y_val, y_pred_classes),
        'top_5_accuracy': sklearn.metrics.top_k_accuracy_score(y_val, y_pred, k=5),
        'top_10_accuracy': sklearn.metrics.top_k_accuracy_score(y_val, y_pred, k=10),
    }
    
    return metrics, y_pred_classes


def print_classification_report(model: Union[tf.keras.Model, 'BaseModel'], 
                              X_val: np.ndarray, 
                              y_val: np.ndarray,
                              frames_val: np.ndarray,
                              ORD2SIGN: Dict,
                              SIGN2ORD: Dict = None) -> pd.DataFrame:
    """Print classification report for validation data."""
    config = DataConfig()
    
    # Get predictions
    metrics, y_pred = evaluate_model(model, X_val, y_val, frames_val)
    
    # Create labels
    labels = [ORD2SIGN.get(i, '').replace(' ', '_') for i in range(config.NUM_CLASSES)]
    
    # Classification report for all signs
    classification_report = sklearn.metrics.classification_report(
        y_val,
        y_pred,
        target_names=labels,
        output_dict=True,
    )
    
    # Round Data for better readability
    classification_report = pd.DataFrame(classification_report).T
    classification_report = classification_report.round(2)
    classification_report = classification_report.astype({
        'support': np.uint16,
    })
    
    # Add signs
    if SIGN2ORD is None:
        SIGN2ORD = {v: k for k, v in ORD2SIGN.items()}
    
    classification_report['sign'] = [e if e in SIGN2ORD else -1 for e in classification_report.index]
    classification_report['sign_ord'] = classification_report['sign'].apply(SIGN2ORD.get).fillna(-1).astype(np.int16)
    
    # Sort on F1-score (excluding summary rows)
    classification_report = pd.concat((
        classification_report.head(config.NUM_CLASSES).sort_values('f1-score', ascending=False),
        classification_report.tail(3),
    ))

    print("\nClassification Report:")
    print(classification_report)
    
    # Print overall metrics
    print(f"\nOverall Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Top-5 Accuracy: {metrics['top_5_accuracy']:.4f}")
    print(f"Top-10 Accuracy: {metrics['top_10_accuracy']:.4f}")
    
    return classification_report


def print_landmark_weights(model: Union[tf.keras.Model, 'BaseModel']) -> None:
    """Print the learned landmark weights (Transformer only)."""
    # Extract Keras model if needed
    if hasattr(model, 'model'):
        keras_model = model.model
    else:
        keras_model = model
    
    try:
        embedding_layer = keras_model.get_layer('embedding')
        for w in embedding_layer.weights:
            if 'landmark_weights' in w.name:
                weights = scipy.special.softmax(w.numpy())
                
        landmarks = ['lips', 'left_hand', 'pose']
        
        print("\nLandmark weights:")
        for w, lm in zip(weights, landmarks):
            print(f'{lm}: {(w*100):.1f}%')
    except:
        print("Landmark weights not available for this model")