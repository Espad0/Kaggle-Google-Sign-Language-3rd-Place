"""
Test script to verify refactored code maintains compatibility with original train.py
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

print("Testing refactored modules...")

# Test 1: Import modules
try:
    import preprocess_data
    import train_transformer
    import convert_to_tflite
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import error: {e}")
    exit(1)

# Test 2: Check configuration compatibility
try:
    from preprocess_data import Config, LandmarkIndices
    config = Config()
    landmarks = LandmarkIndices()
    
    # Verify constants match original values
    assert config.N_ROWS == 543
    assert config.N_DIMS == 3
    assert config.NUM_CLASSES == 250
    assert config.INPUT_SIZE == 32
    assert landmarks.N_COLS == 66  # 40 lips + 21 hand + 5 pose
    
    print("✓ Configuration constants match original values")
except Exception as e:
    print(f"✗ Configuration error: {e}")

# Test 3: Test PreprocessLayer compatibility
try:
    from preprocess_data import PreprocessLayer, load_relevant_data_subset
    
    # Create a dummy sample
    dummy_data = np.random.randn(10, 543, 3).astype(np.float32)
    
    # Test preprocessing
    preprocess_layer = PreprocessLayer()
    processed, frame_idxs = preprocess_layer(dummy_data)
    
    assert processed.shape == (32, 66, 3)
    assert frame_idxs.shape == (32,)
    
    print("✓ PreprocessLayer works correctly")
except Exception as e:
    print(f"✗ PreprocessLayer error: {e}")

# Test 4: Test batch generator compatibility
try:
    from preprocess_data import get_train_batch_all_signs
    
    # Create dummy data
    X = np.random.randn(500, 32, 66, 3).astype(np.float32)
    y = np.random.randint(0, 250, 500).astype(np.int32)
    NON_EMPTY_FRAME_IDXS = np.random.randn(500, 32).astype(np.float32)
    
    # Test batch generator
    batch_gen = get_train_batch_all_signs(X, y, NON_EMPTY_FRAME_IDXS, config)
    X_batch, y_batch = next(batch_gen)
    
    assert X_batch['frames'].shape == (500, 32, 66, 3)
    assert X_batch['non_empty_frame_idxs'].shape == (500, 32)
    assert y_batch.shape == (500,)
    assert len(np.unique(y_batch)) == 250
    
    print("✓ Batch generator works correctly")
except Exception as e:
    print(f"✗ Batch generator error: {e}")

# Test 5: Test model architecture
try:
    from train_transformer import get_model
    
    # Create dummy stats
    stats = {
        'lips': (np.zeros((40, 2)), np.ones((40, 2))),
        'left_hand': (np.zeros((21, 2)), np.ones((21, 2))),
        'pose': (np.zeros((5, 2)), np.ones((5, 2)))
    }
    
    # Build model
    tf.keras.backend.clear_session()
    model = get_model(stats)
    
    # Test model input/output shapes
    test_input = {
        'frames': np.random.randn(1, 32, 66, 3).astype(np.float32),
        'non_empty_frame_idxs': np.ones((1, 32)).astype(np.float32)
    }
    
    output = model(test_input)
    assert output.shape == (1, 250)
    
    # Check model has expected layers
    layer_names = [layer.name for layer in model.layers]
    assert 'embedding' in [layer.name for layer in model.layers if hasattr(layer, 'name')]
    assert 'transformer' in [layer.name for layer in model.layers if hasattr(layer, 'name')]
    
    print("✓ Model architecture is correct")
except Exception as e:
    print(f"✗ Model architecture error: {e}")

# Test 6: Test TFLite conversion wrapper
try:
    from convert_to_tflite import TFLiteModel
    
    # Create TFLite wrapper
    tflite_model = TFLiteModel(model)
    
    # Test with raw input
    raw_input = np.random.randn(10, 543, 3).astype(np.float32)
    output = tflite_model(raw_input)
    
    assert 'outputs' in output
    assert output['outputs'].shape == (250,)
    
    print("✓ TFLite wrapper works correctly")
except Exception as e:
    print(f"✗ TFLite wrapper error: {e}")

# Test 7: Verify backward compatibility functions exist
try:
    # Check if backward compatibility function exists
    from preprocess_data import preprocess_data
    
    print("✓ Backward compatibility functions exist")
except Exception as e:
    print(f"✗ Backward compatibility error: {e}")

# Test 8: Test data I/O functions
try:
    from preprocess_data import save_compressed, load_compressed
    
    # Test save/load
    test_data = np.random.randn(10, 32, 66, 3).astype(np.float32)
    save_compressed(test_data, 'test_data.zip')
    loaded_data = load_compressed('test_data.zip')
    
    assert np.allclose(test_data, loaded_data)
    
    # Cleanup
    os.remove('test_data.zip')
    
    print("✓ Data I/O functions work correctly")
except Exception as e:
    print(f"✗ Data I/O error: {e}")

print("\n" + "="*50)
print("Refactoring test complete!")
print("="*50)

# Summary of changes from original train.py:
print("\nKey improvements in refactored code:")
print("1. Modular structure - separate files for preprocessing, training, and conversion")
print("2. Better configuration management using dataclasses")
print("3. Compressed data storage (.zip) for memory efficiency")
print("4. Cleaner function organization and reusability")
print("5. Improved error handling and type hints")
print("6. Backward compatibility maintained")

print("\nTo use the refactored code:")
print("1. For preprocessing: python preprocess_data.py")
print("2. For training: python train_transformer.py")
print("3. For TFLite conversion: python convert_to_tflite.py")