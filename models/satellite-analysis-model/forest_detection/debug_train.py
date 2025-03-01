import os
import sys

# Add debug print statements at the start
print("Script starting...")

try:
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from datetime import datetime
    from model import build_forest_detection_model
    import ssl

    print("All imports successful")
except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Print directory contents to check paths
print(f"Current directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir('.')}")

# Check if the model.py file exists in the correct location
if os.path.exists("model.py"):
    print("model.py found")
else:
    print("model.py not found!")
    sys.exit(1)

# Disable SSL verification temporarily
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    print("SSL context modified")
except Exception as e:
    print(f"SSL context error: {e}")


# Add call to main function explicitly
def main():
    """Main function with debug prints"""
    print("Entering main function")

    # Debug directories
    DATA_DIR = "../../../data/processed/processed_ndvi_rgb/image_datasets"
    print(f"DATA_DIR path: {DATA_DIR}")
    print(f"DATA_DIR exists: {os.path.exists(DATA_DIR)}")

    if os.path.exists(DATA_DIR):
        print(f"DATA_DIR contents: {os.listdir(DATA_DIR)}")

    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    print(f"TRAIN_DIR path: {TRAIN_DIR}")
    print(f"TRAIN_DIR exists: {os.path.exists(TRAIN_DIR)}")

    VAL_DIR = os.path.join(DATA_DIR, "val")
    print(f"VAL_DIR path: {VAL_DIR}")
    print(f"VAL_DIR exists: {os.path.exists(VAL_DIR)}")

    # Try to load a small portion of your code to see where it fails
    print("Trying to build model...")
    try:
        model = build_forest_detection_model(
            input_shape=(64, 64, 4), use_pretrained=True
        )
        print("Model built successfully")
    except Exception as e:
        print(f"Model building error: {e}")
        import traceback

        traceback.print_exc()

    print("Main function complete")


# Make sure the main function is called
if __name__ == "__main__":
    print("Calling main function")
    main()
    print("Script completed")
