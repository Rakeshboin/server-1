"""
Script to convert .keras model to TensorFlow Lite (.tflite) format.

This script:
1. Loads the .keras model
2. Saves it as SavedModel format
3. Converts to TFLite format
4. Saves to assets/models/model.tflite

Usage:
    python server/convert_model.py

Environment variables:
    INPUT_MODEL: Path to input .keras model (default: deep_learning_crop_recommendation_model.keras)
    OUTPUT_PATH: Path to output .tflite file (default: assets/models/model.tflite)
"""

import os
import sys
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

# Configuration
INPUT_MODEL = os.environ.get("INPUT_MODEL", "deep_learning_crop_recommendation_model.keras")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "assets/models/model.tflite")
SAVED_MODEL_DIR = "saved_model_temp"

def convert_keras_to_tflite(input_model_path, output_path):
    """
    Convert a Keras model to TFLite format.
    
    Args:
        input_model_path: Path to input .keras model file
        output_path: Path to output .tflite file
    """
    print(f"Converting model from {input_model_path} to {output_path}...")
    
    # Step 1: Load the Keras model
    if not os.path.exists(input_model_path):
        print(f"ERROR: Model file not found at {input_model_path}")
        print(f"Current working directory: {os.getcwd()}")
        return False
    
    print(f"Loading Keras model from {input_model_path}...")
    try:
        model = keras.models.load_model(input_model_path)
        print(f"Model loaded successfully!")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return False
    
    # Step 2: Save as SavedModel format (optional intermediate step)
    print(f"\nSaving model as SavedModel format to {SAVED_MODEL_DIR}...")
    try:
        # Clean up old SavedModel directory if it exists
        if os.path.exists(SAVED_MODEL_DIR):
            import shutil
            shutil.rmtree(SAVED_MODEL_DIR)
        
        model.save(SAVED_MODEL_DIR)
        print("SavedModel created successfully!")
    except Exception as e:
        print(f"ERROR: Failed to save SavedModel: {e}")
        return False
    
    # Step 3: Convert to TFLite
    print(f"\nConverting SavedModel to TFLite format...")
    try:
        # Create TFLite converter from SavedModel
        converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
        
        # Optional: Set optimization flags
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert model
        tflite_model = converter.convert()
        print("TFLite conversion successful!")
        
        # Get model size
        model_size_mb = len(tflite_model) / (1024 * 1024)
        print(f"Model size: {model_size_mb:.2f} MB")
    except Exception as e:
        print(f"ERROR: Failed to convert to TFLite: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Save TFLite model
    print(f"\nSaving TFLite model to {output_path}...")
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created directory: {output_dir}")
        
        # Write TFLite model to file
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite model saved successfully to {output_path}!")
    except Exception as e:
        print(f"ERROR: Failed to save TFLite model: {e}")
        return False
    
    # Step 5: Clean up temporary SavedModel directory
    try:
        if os.path.exists(SAVED_MODEL_DIR):
            import shutil
            shutil.rmtree(SAVED_MODEL_DIR)
            print(f"\nCleaned up temporary directory: {SAVED_MODEL_DIR}")
    except Exception as e:
        print(f"WARNING: Failed to clean up {SAVED_MODEL_DIR}: {e}")
    
    print("\n" + "="*60)
    print("Conversion completed successfully!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Update pubspec.yaml to include the model asset:")
    print(f"   flutter:")
    print(f"     assets:")
    print(f"       - {output_path}")
    print(f"2. Run: flutter pub get")
    print(f"3. The model will be available in the app for on-device predictions")
    print()
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("FarmSense Model Converter")
    print("Converting .keras model to .tflite format")
    print("="*60)
    print()
    
    success = convert_keras_to_tflite(INPUT_MODEL, OUTPUT_PATH)
    
    if not success:
        print("\nERROR: Conversion failed. Check the error messages above.")
        sys.exit(1)
    
    sys.exit(0)

