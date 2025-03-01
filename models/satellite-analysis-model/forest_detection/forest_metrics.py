#!/usr/bin/env python3
"""
forest_metrics.py

This script extracts forest metrics from a satellite image using the forest detection model.
It can process either a single image or extract one from a dataset.

Usage:
  python forest_metrics.py --model_path path/to/model.h5 --image_path path/to/image.npy --output_path results.json
  or
  python forest_metrics.py --model_path path/to/model.h5 --dataset_path path/to/X_features.npy --image_index 42 --output_path results.json

Arguments:
  --model_path    Path to the trained forest detection model (.h5)
  --image_path    Path to a single image file (.npy) [optional if using dataset_path]
  --dataset_path  Path to the dataset file (.npy) [optional if using image_path]
  --image_index   Index of the image to extract from dataset [required if using dataset_path]
  --output_path   Path to save the output JSON metrics
  --visualize     Flag to save visualization of the prediction (default: False)
  --viz_path      Path to save visualization (default: prediction.png)
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


# Define custom loss functions
def dice_loss(y_true, y_pred):
    """Dice loss function for segmentation"""
    smooth = 1.0
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def combined_loss(y_true, y_pred):
    """Combination of binary crossentropy and dice loss"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice


# Custom metrics for model loading
def custom_iou(y_true, y_pred, threshold=0.5):
    """Custom IoU metric with adjustable threshold"""
    # Apply threshold to predictions
    y_pred = tf.cast(y_pred > threshold, tf.float32)

    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = (
        tf.reduce_sum(y_true, axis=[1, 2, 3])
        + tf.reduce_sum(y_pred, axis=[1, 2, 3])
        - intersection
    )

    # Add small epsilon to avoid division by zero
    iou = tf.reduce_mean((intersection + 1e-7) / (union + 1e-7))
    return iou


def make_iou_threshold(threshold):
    """Create an IoU metric with a specific threshold"""

    def iou_threshold(y_true, y_pred):
        return custom_iou(y_true, y_pred, threshold)

    iou_threshold.__name__ = f"iou_threshold_{threshold}"
    return iou_threshold


def load_forest_detection_model(model_path):
    """Load the forest detection model with custom metrics and loss functions"""
    custom_objects = {
        "iou_threshold_0.1": make_iou_threshold(0.1),
        "iou_threshold_0.3": make_iou_threshold(0.3),
        "iou_threshold_0.5": make_iou_threshold(0.5),
        "custom_iou": custom_iou,
        "make_iou_threshold": make_iou_threshold,
        "dice_loss": dice_loss,
        "combined_loss": combined_loss,
    }

    print(f"Loading model from {model_path}")
    try:
        model = load_model(model_path, custom_objects=custom_objects)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model with custom objects: {e}")
        print("Attempting to load with compile=False...")

        # Try loading without compilation
        model = load_model(model_path, compile=False)

        # Recompile with standard loss and metrics
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        print("Model loaded without custom loss")
        return model


def load_image(image_path):
    """Load a single image from .npy file"""
    print(f"Loading image from {image_path}")

    try:
        image = np.load(image_path)

        # If the image doesn't have a batch dimension, add one
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        print(f"Image loaded successfully. Shape: {image.shape}")
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        raise


def extract_from_dataset(dataset_path, image_index):
    """Extract a single image from a dataset"""
    print(f"Loading dataset from {dataset_path}")

    try:
        dataset = np.load(dataset_path)
        print(f"Dataset shape: {dataset.shape}")

        if image_index >= len(dataset):
            raise ValueError(
                f"Image index {image_index} is out of range. Dataset has {len(dataset)} images."
            )

        # Extract image
        image = dataset[image_index]

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        print(f"Extracted image with shape: {image.shape}")
        return image
    except Exception as e:
        print(f"Error extracting from dataset: {e}")
        raise


def analyze_forest_coverage(model, image, thresholds=[0.1, 0.3, 0.5, 0.7, 0.9]):
    """
    Analyze forest coverage in a satellite image

    Args:
        model: Trained forest detection model
        image: Preprocessed satellite image (with batch dimension)
        thresholds: List of thresholds to apply to predictions

    Returns:
        Dictionary with forest coverage metrics and prediction
    """
    print("Analyzing forest coverage...")

    # Get model prediction
    prediction = model.predict(image)

    # Calculate forest coverage percentage at different thresholds
    coverage_metrics = {}
    binary_masks = {}

    for threshold in thresholds:
        # Apply threshold to get binary mask
        binary_mask = prediction[0, :, :, 0] > threshold

        # Calculate percentage of forest coverage
        forest_percentage = np.mean(binary_mask) * 100

        coverage_metrics[f"forest_coverage_threshold_{threshold}"] = float(
            f"{forest_percentage:.2f}"
        )
        binary_masks[f"binary_mask_{threshold}"] = binary_mask

    # Calculate average forest coverage across all thresholds
    coverage_metrics["average_forest_coverage"] = float(
        f"{np.mean(list(coverage_metrics.values())):.2f}"
    )

    # Calculate other useful statistics
    pixel_stats = {
        "min_probability": float(f"{np.min(prediction[0, :, :, 0]):.4f}"),
        "max_probability": float(f"{np.max(prediction[0, :, :, 0]):.4f}"),
        "mean_probability": float(f"{np.mean(prediction[0, :, :, 0]):.4f}"),
        "median_probability": float(f"{np.median(prediction[0, :, :, 0]):.4f}"),
        "std_probability": float(f"{np.std(prediction[0, :, :, 0]):.4f}"),
    }

    # Try to calculate connectivity metrics (for threshold 0.5)
    connectivity_metrics = {}
    edge_metrics = {}

    try:
        from scipy import ndimage

        binary_mask_05 = binary_masks["binary_mask_0.5"]

        # Label connected components
        labeled_mask, num_components = ndimage.label(binary_mask_05)

        # Calculate component sizes
        component_sizes = (
            np.bincount(labeled_mask.flatten())[1:] if num_components > 0 else []
        )

        connectivity_metrics = {
            "forest_patches": int(num_components),
            "largest_patch_percentage": (
                float(f"{np.max(component_sizes) / binary_mask_05.size * 100:.2f}")
                if component_sizes.size > 0
                else 0
            ),
            "average_patch_size_percentage": (
                float(f"{np.mean(component_sizes) / binary_mask_05.size * 100:.2f}")
                if component_sizes.size > 0
                else 0
            ),
        }

        # Calculate forest edge metrics
        if np.any(binary_mask_05):
            eroded = ndimage.binary_erosion(binary_mask_05)
            edge_mask = binary_mask_05 & ~eroded
            edge_percentage = np.mean(edge_mask) * 100
        else:
            edge_percentage = 0

        edge_metrics = {"forest_edge_percentage": float(f"{edge_percentage:.2f}")}
    except ImportError:
        print("SciPy not found. Skipping connectivity and edge metrics.")
    except Exception as e:
        print(f"Error calculating connectivity metrics: {e}")

    # Combine all metrics
    metrics = {"forest_coverage": coverage_metrics, "pixel_statistics": pixel_stats}

    if connectivity_metrics:
        metrics["connectivity"] = connectivity_metrics

    if edge_metrics:
        metrics["edge_metrics"] = edge_metrics

    return metrics, prediction


def save_visualization(image, prediction, output_path):
    """Save visualization of model prediction"""
    plt.figure(figsize=(15, 5))

    # Display RGB image
    plt.subplot(1, 3, 1)
    # Check if image has 4 channels (RGB + NDVI)
    if image.shape[2] >= 3:
        plt.imshow(np.clip(image[:, :, :3], 0, 1))  # Show RGB channels
    else:
        plt.imshow(image[:, :, 0], cmap="gray")
    plt.title("RGB Image")
    plt.axis("off")

    # Display NDVI if available
    plt.subplot(1, 3, 2)
    if image.shape[2] >= 4:
        plt.imshow(image[:, :, 3], cmap="viridis")
        plt.title("NDVI")
    else:
        plt.imshow(image[:, :, 0], cmap="gray")
        plt.title("Channel 1")
    plt.axis("off")

    # Display prediction
    plt.subplot(1, 3, 3)
    plt.imshow(prediction[0, :, :, 0], cmap="viridis")
    plt.title("Forest Prediction")
    plt.colorbar(label="Probability")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Visualization saved to {output_path}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Forest Coverage Metrics Extraction")
    parser.add_argument(
        "--model_path", required=True, help="Path to the trained forest detection model"
    )
    parser.add_argument("--image_path", help="Path to a single image file (.npy)")
    parser.add_argument("--dataset_path", help="Path to the dataset file (.npy)")
    parser.add_argument(
        "--image_index", type=int, help="Index of the image to extract from dataset"
    )
    parser.add_argument(
        "--output_path", required=True, help="Path to save the output JSON metrics"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Save visualization of the prediction"
    )
    parser.add_argument(
        "--viz_path", default="prediction.png", help="Path to save visualization"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.image_path is None and args.dataset_path is None:
        parser.error("Either --image_path or --dataset_path is required")

    if args.dataset_path is not None and args.image_index is None:
        parser.error("--image_index is required when using --dataset_path")

    # Create output directory if needed
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = load_forest_detection_model(args.model_path)

    # Load image
    if args.image_path:
        image = load_image(args.image_path)
    else:
        image = extract_from_dataset(args.dataset_path, args.image_index)

    # Analyze forest coverage
    metrics, prediction = analyze_forest_coverage(model, image)

    # Save visualization if requested
    if args.visualize:
        save_visualization(image[0], prediction, args.viz_path)

    # Save metrics to JSON
    with open(args.output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {args.output_path}")
    print(
        f"Average forest coverage: {metrics['forest_coverage']['average_forest_coverage']}%"
    )
    if "connectivity" in metrics:
        print(f"Forest patches: {metrics['connectivity']['forest_patches']}")
    if "edge_metrics" in metrics:
        print(
            f"Forest edge percentage: {metrics['edge_metrics']['forest_edge_percentage']}%"
        )


if __name__ == "__main__":
    main()
