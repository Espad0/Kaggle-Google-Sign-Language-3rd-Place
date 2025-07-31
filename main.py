"""
Main entry point for training sign language recognition models.

Usage:
    python main.py --model transformer --train
    python main.py --model conv1d --train --validation
    python main.py --model transformer --convert-tflite
"""
import argparse
import sys


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
    
    args = parser.parse_args()
    
    # Import the appropriate modules based on model type
    if args.model == 'transformer':
        from models.transformer import TransformerModel
        from core.config import TransformerConfig as ModelConfig
        model_class = TransformerModel
        weights_file = 'model.h5'
        tflite_file = 'model.tflite'
    else:
        from models.conv1d import Conv1DModel  
        from core.config import Conv1DConfig as ModelConfig
        model_class = Conv1DModel
        weights_file = 'model_conv.h5'
        tflite_file = 'model_conv.tflite'
    
    # Import common modules
    from core.config import DataConfig, TrainingConfig
    from data.loader import prepare_data
    from training.trainer import Trainer
    from tflite.converter import convert_model_to_tflite
    
    # Create configurations
    data_config = DataConfig()
    model_config = ModelConfig()
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
        
        # Load data
        data = prepare_data(data_config, training_config)
        
        # Create and train model
        model = model_class(data_config, model_config)
        trainer = Trainer(training_config)
        
        history = trainer.train(
            model=model,
            train_data=data['train'],
            val_data=data.get('val'),
            weights_path=weights_file
        )
        
        print(f"Training complete! Weights saved to {weights_file}")
    
    if args.convert_tflite:
        print(f"Converting {args.model} model to TFLite...")
        
        # Create model and load weights
        model = model_class(data_config, model_config)
        model.load_weights(weights_file)
        
        # Convert to TFLite
        convert_model_to_tflite(
            model=model,
            output_path=tflite_file,
            model_type=args.model
        )
        
        print(f"Conversion complete! TFLite model saved to {tflite_file}")
    
    if not args.train and not args.convert_tflite:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()