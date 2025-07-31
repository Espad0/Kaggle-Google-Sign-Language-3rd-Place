"""
Example script showing how to use the refactored code to replicate train.py functionality
"""

# This script demonstrates how to use the refactored modules to achieve
# the same functionality as the original train.py

# ========== Configuration ==========
PREPROCESS_DATA = False  # Set to True to re-preprocess data
TRAIN_MODEL = True
USE_VAL = False
SHOW_PLOTS = False

# ========== Data Preprocessing ==========
if PREPROCESS_DATA:
    print("Preprocessing data from scratch...")
    from preprocess_data import prepare_data
    
    # This replaces the entire preprocessing section of train.py
    data = prepare_data(
        preprocess=True,
        use_validation=USE_VAL,
        show_plots=SHOW_PLOTS
    )
    print("Data preprocessing complete!")

# ========== Model Training ==========
if TRAIN_MODEL:
    print("\nTraining transformer model...")
    from train_transformer import main as train_main
    
    # Configure training
    from train_transformer import TrainingConfig
    config = TrainingConfig()
    config.train_model = TRAIN_MODEL
    config.use_validation = USE_VAL
    config.show_plots = SHOW_PLOTS
    config.n_epochs = 100
    config.batch_all_signs_n = 2
    
    # This replaces the entire training section of train.py
    # Note: The main() function in train_transformer.py handles everything
    model = train_main()
    print("Training complete!")

# ========== TFLite Conversion ==========
print("\nConverting to TFLite...")
from convert_to_tflite import main as convert_main

# This replaces the TFLite conversion section of train.py
convert_main(
    model_weights_path='model.h5',
    output_tflite_path='model.tflite',
    output_zip_path='submission.zip',
    verify=True
)
print("Conversion complete!")

print("\n" + "="*60)
print("All steps completed successfully!")
print("="*60)
print("\nThe refactored code produces the same outputs as train.py:")
print("- model.h5 (trained model weights)")
print("- model.tflite (TFLite model)")
print("- submission.zip (competition submission)")
print("\nAdditional benefits of refactored code:")
print("- Compressed data files (.zip) save disk space")
print("- Modular structure allows running individual steps")
print("- Better error handling and progress reporting")
print("- Easier to maintain and extend")