"""
Main entry point for training, evaluating, and converting sign language recognition models.

This script provides a unified interface for:
- Training transformer or Conv1D models on sign language data
- Converting trained models to TensorFlow Lite format
- Evaluating model performance on validation sets
- Preprocessing data from scratch when needed

Usage Examples:
    # Train a transformer model
    python main.py --model transformer --train
    
    # Train a Conv1D model with validation split
    python main.py --model conv1d --train --validation
    
    # Train with custom hyperparameters
    python main.py --model transformer --train --epochs 150 --batch-size 32 --lr 0.001
    
    # Preprocess data from scratch before training
    python main.py --model transformer --train --preprocess
    
    # Convert a trained model to TFLite format
    python main.py --model transformer --convert-tflite
    
    # Evaluate model performance (requires validation data)
    python main.py --model transformer --evaluate --validation

Arguments:
    --model: Model type to use (transformer or conv1d)
    --train: Train the model
    --convert-tflite: Convert trained model to TFLite format
    --validation: Use validation split for training/evaluation
    --preprocess: Preprocess data from scratch
    --epochs: Number of training epochs (default: 100)
    --batch-size: Batch size for training (default: 64)
    --lr: Maximum learning rate (default: 0.001)
    --evaluate: Evaluate model on validation set

Output Files:
    - Transformer model weights: outputs/model.h5
    - Conv1D model weights: outputs/model_conv.h5
    - TFLite models: outputs/model.tflite or outputs/model_conv.tflite
"""
import argparse
import sys
import gc
import numpy as np

from core import (
    DataConfig, TrainingConfig, TransformerConfig, Conv1DConfig,
    LandmarkIndices, load_compressed, load_metadata
)
from models import TransformerModel, Conv1DModel
from processing import prepare_data, calculate_mean_std_stats
from training import Trainer, print_classification_report
from tflite import convert_model_to_tflite, verify_tflite_model


def main():
    parser = argparse.ArgumentParser(description='Sign Language Recognition')
    parser.add_argument('--model', type=str, choices=['transformer', 'conv1d'], 
                        required=True, help='Model type to use')
    parser.add_argument('--train', action='store_true', 
                        help='Train the model')
    parser.add_argument('--convert-tflite', action='store_true',
                        help='Convert trained model to TFLite')
    parser.add_argument('--validation', action='store_true',
                        help='Use validation split')
    parser.add_argument('--preprocess', action='store_true',
                        help='Preprocess data from scratch')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Maximum learning rate')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate model on validation set')
    
    args = parser.parse_args()
    
    # Select the appropriate model based on model type
    if args.model == 'transformer':
        model_config = TransformerConfig()
        model_class = TransformerModel
        weights_file = 'outputs/model.h5'
        tflite_file = 'outputs/model.tflite'
    else:
        model_config = Conv1DConfig()
        model_class = Conv1DModel
        weights_file = 'outputs/model_conv.h5'
        tflite_file = 'outputs/model_conv.tflite'
    
    # Create configurations
    data_config = DataConfig()
    training_config = TrainingConfig(
        train_model=args.train,
        use_validation=args.validation,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr_max=args.lr,
        generate_data=args.preprocess
    )
    
    if args.train:
        print(f"Training {args.model} model...")
        
        # Prepare data configuration
        data_prep_config = {
            'preprocess': args.preprocess,
            'use_validation': args.validation,
            'show_plots': False,
            'analyze_stats': True
        }
        
        # Load data
        data = prepare_data(data_prep_config, model_type=args.model)
        
        # Calculate statistics for transformer model
        stats = None
        if args.model == 'transformer':
            print("Calculating landmark statistics for transformer...")
            landmarks = LandmarkIndices()
            stats = calculate_mean_std_stats(data['X_train'], landmarks, data_config)
        
        # Create and train model
        model = model_class(data_config, model_config)
        if stats:
            model.build_model(stats)
        
        trainer = Trainer(training_config)
        
        # Prepare training data
        train_data = {
            'X_train': data['X_train'],
            'y_train': data['y_train'],
            'NON_EMPTY_FRAME_IDXS_TRAIN': data['NON_EMPTY_FRAME_IDXS_TRAIN']
        }
        
        history = trainer.train(
            model=model,
            train_data=train_data,
            val_data=data.get('validation_data'),
            weights_path=weights_file
        )
        
        print(f"Training complete! Weights saved to {weights_file}")
        
        # Cleanup
        del data
        gc.collect()
    
    if args.convert_tflite:
        print(f"Converting {args.model} model to TFLite...")
        
        # Load statistics for transformer
        stats = None
        if args.model == 'transformer':
            try:
                # Try to load from training data
                X_train = load_compressed('X_train.zip')
                landmarks = LandmarkIndices()
                stats = calculate_mean_std_stats(X_train, landmarks, data_config)
                print("Calculated landmark statistics from training data")
                del X_train
                gc.collect()
            except:
                print("Warning: Using default landmark statistics")
                stats = {
                    'lips': (np.zeros((40, 2)), np.ones((40, 2))),
                    'left_hand': (np.zeros((21, 2)), np.ones((21, 2))),
                    'pose': (np.zeros((5, 2)), np.ones((5, 2)))
                }
        
        # Create model and load weights
        model = model_class(data_config, model_config)
        if stats:
            model.build_model(stats)
        model.load_weights(weights_file)
        
        # Convert to TFLite
        convert_model_to_tflite(
            model=model,
            output_path=tflite_file,
            model_type=args.model
        )
        
        # Verify if requested
        verify_tflite_model(tflite_file)
        
        print(f"Conversion complete! TFLite model saved to {tflite_file}")
    
    if args.evaluate:
        if not args.validation:
            print("Error: --evaluate requires --validation flag")
            sys.exit(1)
        
        print(f"Evaluating {args.model} model...")
        
        # Load validation data
        try:
            X_val = load_compressed('X_val.zip')
            y_val = load_compressed('y_val.zip')
            frames_val = load_compressed('NON_EMPTY_FRAME_IDXS_VAL.zip')
        except:
            print("Error: Validation data not found. Run with --train --validation first")
            sys.exit(1)
        
        # Load metadata
        _, sign2ord, ord2sign = load_metadata()
        
        # Load statistics for transformer
        stats = None
        if args.model == 'transformer':
            try:
                X_train = load_compressed('X_train.zip')
                landmarks = LandmarkIndices()
                stats = calculate_mean_std_stats(X_train, landmarks, data_config)
                del X_train
                gc.collect()
            except:
                stats = {
                    'lips': (np.zeros((40, 2)), np.ones((40, 2))),
                    'left_hand': (np.zeros((21, 2)), np.ones((21, 2))),
                    'pose': (np.zeros((5, 2)), np.ones((5, 2)))
                }
        
        # Create model and load weights
        model = model_class(data_config, model_config)
        if stats:
            model.build_model(stats)
        model.load_weights(weights_file)
        
        # Evaluate
        print_classification_report(model, X_val, y_val, frames_val, ord2sign, sign2ord)
    
    if not args.train and not args.convert_tflite and not args.evaluate:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()