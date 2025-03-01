#!/usr/bin/env python3
"""
change_metrics.py

This script extracts forest change metrics from two satellite images using both 
the forest detection model and change detection model.

Usage:
  python change_metrics.py --forest_model path/to/forest_model.h5 --change_model path/to/change_model.h5 
                          --image_t1 path/to/image_t1.npy --image_t2 path/to/image_t2.npy 
                          --output_path results.json

  or with dataset extraction:
  
  python change_metrics.py --forest_model path/to/forest_model.h5 --change_model path/to/change_model.h5 
                          --dataset_path path/to/X_features.npy --index_t1 5 --index_t2 10 
                          --output_path results.json
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


# Custom loss functions
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


def load_model_with_custom_metrics(model_path):
    """Load model with custom metrics and preprocessing functions"""

    # Define a dummy preprocess_input function
    def preprocess_input(x):
        """Dummy preprocessing function that returns the input unchanged"""
        return x

    # Custom objects dictionary with all necessary functions
    custom_objects = {
        "iou_threshold_0.1": make_iou_threshold(0.1),
        "iou_threshold_0.3": make_iou_threshold(0.3),
        "iou_threshold_0.5": make_iou_threshold(0.5),
        "custom_iou": custom_iou,
        "make_iou_threshold": make_iou_threshold,
        "dice_loss": dice_loss,
        "combined_loss": combined_loss,
        "preprocess_input": preprocess_input,  # Add the dummy preprocessing function
    }

    print(f"Loading model from {model_path}")
    try:
        # First try with custom objects
        model = load_model(model_path, custom_objects=custom_objects)
        print("Model loaded successfully with custom objects")
    except Exception as e:
        print(f"Error loading model with custom objects: {e}")
        print("Attempting to load with compile=False...")

        try:
            # Try loading without compilation
            model = load_model(model_path, compile=False, custom_objects=custom_objects)
            print("Model loaded with compile=False")

            # Recompile with standard loss and metrics
            model.compile(
                optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
            )
            print("Model recompiled with standard metrics")
        except Exception as e:
            print(f"Error loading with compile=False: {e}")
            print("Trying one more approach...")

            # Last resort: try to load without custom objects
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                print("Model loaded without custom objects")

                # Recompile with standard metrics
                model.compile(
                    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
                )
                print("Model recompiled with standard metrics")
            except Exception as e:
                print(f"All loading attempts failed: {e}")
                raise ValueError(f"Could not load model from {model_path}")

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


def analyze_forest_coverage(model, image):
    """Get forest coverage from a single image using forest detection model"""
    # Get model prediction
    prediction = model.predict(image)

    # Apply threshold (0.5) to get binary mask
    binary_mask = prediction[0, :, :, 0] > 0.5

    # Calculate forest percentage
    forest_percentage = np.mean(binary_mask) * 100

    return float(f"{forest_percentage:.2f}"), binary_mask, prediction[0, :, :, 0]


def analyze_forest_change(forest_model, change_model, image_t1, image_t2):
    """
    Analyze forest cover change between two time periods

    Args:
        forest_model: Trained forest detection model
        change_model: Trained change detection model
        image_t1: Preprocessed satellite image from time period 1 (earlier)
        image_t2: Preprocessed satellite image from time period 2 (later)

    Returns:
        Dictionary with forest change metrics and predictions
    """
    print("Analyzing forest change...")

    # Get forest coverage for each time period using forest detection model
    forest_percentage_t1, binary_mask_t1, prediction_t1 = analyze_forest_coverage(
        forest_model, image_t1
    )
    forest_percentage_t2, binary_mask_t2, prediction_t2 = analyze_forest_coverage(
        forest_model, image_t2
    )

    # Process with change detection model
    # Try different concatenation approaches based on your model
    change_prediction = None
    try:
        # Try concatenating along batch dimension
        change_input = np.concatenate([image_t1, image_t2], axis=0)
        change_prediction = change_model.predict(change_input)
        print("Change model prediction successful (batch concat)")
    except:
        try:
            # Try concatenating along channel dimension
            # First, remove batch dimension if needed
            if image_t1.shape[0] == 1 and image_t2.shape[0] == 1:
                img1 = image_t1[0]
                img2 = image_t2[0]
                combined = np.concatenate([img1, img2], axis=-1)
                combined = np.expand_dims(combined, axis=0)
                change_prediction = change_model.predict(combined)
                print("Change model prediction successful (channel concat)")
        except Exception as e:
            print(f"Error during change model prediction: {e}")
            print("Calculating change metrics from forest model outputs only")

    # Calculate forest change
    forest_change = forest_percentage_t2 - forest_percentage_t1
    forest_change_relative = (forest_change / (forest_percentage_t1 + 1e-8)) * 100

    # Calculate deforested and reforested areas
    deforested_mask = np.logical_and(binary_mask_t1, np.logical_not(binary_mask_t2))
    deforested_percentage = np.mean(deforested_mask) * 100

    reforested_mask = np.logical_and(np.logical_not(binary_mask_t1), binary_mask_t2)
    reforested_percentage = np.mean(reforested_mask) * 100

    # Calculate forest fragmentation metrics
    fragmentation_metrics = {}
    edge_metrics = {}

    try:
        from scipy import ndimage

        # Label connected components for each time period
        labeled_mask_t1, num_components_t1 = ndimage.label(binary_mask_t1)
        labeled_mask_t2, num_components_t2 = ndimage.label(binary_mask_t2)

        # Calculate component sizes
        component_sizes_t1 = (
            np.bincount(labeled_mask_t1.flatten())[1:] if num_components_t1 > 0 else []
        )
        component_sizes_t2 = (
            np.bincount(labeled_mask_t2.flatten())[1:] if num_components_t2 > 0 else []
        )

        # Calculate forest fragmentation metrics
        fragmentation_metrics = {
            "forest_patches_t1": int(num_components_t1),
            "forest_patches_t2": int(num_components_t2),
            "change_in_patches": int(num_components_t2 - num_components_t1),
            "largest_patch_percentage_t1": (
                float(f"{np.max(component_sizes_t1) / binary_mask_t1.size * 100:.2f}")
                if component_sizes_t1.size > 0
                else 0
            ),
            "largest_patch_percentage_t2": (
                float(f"{np.max(component_sizes_t2) / binary_mask_t2.size * 100:.2f}")
                if component_sizes_t2.size > 0
                else 0
            ),
            "change_in_largest_patch": float(
                f"{(np.max(component_sizes_t2) if component_sizes_t2.size > 0 else 0) / binary_mask_t2.size * 100 - (np.max(component_sizes_t1) if component_sizes_t1.size > 0 else 0) / binary_mask_t1.size * 100:.2f}"
            ),
        }

        # Calculate forest edge metrics
        # Create edge masks using morphological operations
        if np.any(binary_mask_t1):
            eroded_t1 = ndimage.binary_erosion(binary_mask_t1)
            edge_mask_t1 = binary_mask_t1 & ~eroded_t1
            edge_percentage_t1 = np.mean(edge_mask_t1) * 100
        else:
            edge_percentage_t1 = 0

        if np.any(binary_mask_t2):
            eroded_t2 = ndimage.binary_erosion(binary_mask_t2)
            edge_mask_t2 = binary_mask_t2 & ~eroded_t2
            edge_percentage_t2 = np.mean(edge_mask_t2) * 100
        else:
            edge_percentage_t2 = 0

        edge_metrics = {
            "forest_edge_percentage_t1": float(f"{edge_percentage_t1:.2f}"),
            "forest_edge_percentage_t2": float(f"{edge_percentage_t2:.2f}"),
            "change_in_edge_percentage": float(
                f"{edge_percentage_t2 - edge_percentage_t1:.2f}"
            ),
        }
    except ImportError:
        print("SciPy not found. Skipping fragmentation and edge metrics.")
    except Exception as e:
        print(f"Error calculating fragmentation metrics: {e}")

    # Create basic change metrics
    basic_change_metrics = {
        "forest_coverage_t1": forest_percentage_t1,
        "forest_coverage_t2": forest_percentage_t2,
        "absolute_change": float(f"{forest_change:.2f}"),
        "relative_change_percentage": float(f"{forest_change_relative:.2f}"),
        "deforested_percentage": float(f"{deforested_percentage:.2f}"),
        "reforested_percentage": float(f"{reforested_percentage:.2f}"),
    }

    # Combine all metrics
    metrics = {"basic_change_metrics": basic_change_metrics}

    if fragmentation_metrics:
        metrics["fragmentation_metrics"] = fragmentation_metrics

    if edge_metrics:
        metrics["edge_metrics"] = edge_metrics

    # Add change model metrics if available
    if change_prediction is not None:
        # This format will depend on your change detection model's output format
        # Adjust according to your model's actual output
        try:
            change_metrics = {
                "model_detected_changes": float(
                    f"{np.mean(change_prediction > 0.5) * 100:.2f}"
                ),
                "mean_change_magnitude": float(
                    f"{np.mean(np.abs(change_prediction)):.4f}"
                ),
                "max_change_magnitude": float(
                    f"{np.max(np.abs(change_prediction)):.4f}"
                ),
            }
            metrics["change_model_metrics"] = change_metrics
        except:
            print(
                "Unable to extract change model metrics due to incompatible output format"
            )

    return metrics, {
        "binary_mask_t1": binary_mask_t1,
        "binary_mask_t2": binary_mask_t2,
        "prediction_t1": prediction_t1,
        "prediction_t2": prediction_t2,
        "change_prediction": change_prediction,
    }


def save_change_visualization(image_t1, image_t2, predictions, output_path):
    """Save visualization of forest change"""
    plt.figure(figsize=(15, 10))

    # Extract predictions
    binary_mask_t1 = predictions["binary_mask_t1"]
    binary_mask_t2 = predictions["binary_mask_t2"]
    prediction_t1 = predictions["prediction_t1"]
    prediction_t2 = predictions["prediction_t2"]

    # Time 1 (Earlier)
    plt.subplot(2, 3, 1)
    if image_t1.shape[2] >= 3:
        plt.imshow(np.clip(image_t1[:, :, :3], 0, 1))  # Show RGB channels
    else:
        plt.imshow(image_t1[:, :, 0], cmap="gray")
    plt.title("RGB Image (Earlier)")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    if image_t1.shape[2] >= 4:
        plt.imshow(image_t1[:, :, 3], cmap="viridis")
        plt.title("NDVI (Earlier)")
    else:
        plt.imshow(image_t1[:, :, 0], cmap="gray")
        plt.title("Channel 1 (Earlier)")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(prediction_t1, cmap="viridis")
    plt.title("Forest Prediction (Earlier)")
    plt.colorbar(label="Probability")
    plt.axis("off")

    # Time 2 (Later)
    plt.subplot(2, 3, 4)
    if image_t2.shape[2] >= 3:
        plt.imshow(np.clip(image_t2[:, :, :3], 0, 1))  # Show RGB channels
    else:
        plt.imshow(image_t2[:, :, 0], cmap="gray")
    plt.title("RGB Image (Later)")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    if image_t2.shape[2] >= 4:
        plt.imshow(image_t2[:, :, 3], cmap="viridis")
        plt.title("NDVI (Later)")
    else:
        plt.imshow(image_t2[:, :, 0], cmap="gray")
        plt.title("Channel 1 (Later)")
    plt.axis("off")

    # Create change map
    plt.subplot(2, 3, 6)
    change_map = np.zeros_like(binary_mask_t1, dtype=np.uint8)
    change_map[np.logical_and(~binary_mask_t1, ~binary_mask_t2)] = (
        0  # No change (non-forest)
    )
    change_map[np.logical_and(binary_mask_t1, binary_mask_t2)] = 1  # No change (forest)
    change_map[np.logical_and(binary_mask_t1, ~binary_mask_t2)] = 2  # Deforestation
    change_map[np.logical_and(~binary_mask_t1, binary_mask_t2)] = 3  # Reforestation

    cmap = plt.cm.get_cmap("viridis", 4)
    plt.imshow(change_map, cmap=cmap, vmin=0, vmax=3)
    plt.title("Forest Change Map")
    cbar = plt.colorbar(ticks=[0.5, 1.5, 2.5, 3.5])
    cbar.set_ticklabels(
        ["No Forest", "Stable Forest", "Deforestation", "Reforestation"]
    )
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Change visualization saved to {output_path}")


def analyze_forest_change(forest_model, image_t1, image_t2):
    """
    Analyze forest cover change between two time periods using only the forest detection model

    Args:
        forest_model: Trained forest detection model
        image_t1: Preprocessed satellite image from time period 1 (earlier)
        image_t2: Preprocessed satellite image from time period 2 (later)

    Returns:
        Dictionary with forest change metrics and predictions
    """
    print("Analyzing forest change using forest detection model only...")

    # Get forest coverage for each time period using forest detection model
    forest_percentage_t1, binary_mask_t1, prediction_t1 = analyze_forest_coverage(
        forest_model, image_t1
    )
    forest_percentage_t2, binary_mask_t2, prediction_t2 = analyze_forest_coverage(
        forest_model, image_t2
    )

    # Calculate forest change
    forest_change = forest_percentage_t2 - forest_percentage_t1
    forest_change_relative = (forest_change / (forest_percentage_t1 + 1e-8)) * 100

    # Calculate deforested and reforested areas
    deforested_mask = np.logical_and(binary_mask_t1, np.logical_not(binary_mask_t2))
    deforested_percentage = np.mean(deforested_mask) * 100

    reforested_mask = np.logical_and(np.logical_not(binary_mask_t1), binary_mask_t2)
    reforested_percentage = np.mean(reforested_mask) * 100

    # Calculate forest fragmentation metrics
    fragmentation_metrics = {}
    edge_metrics = {}

    try:
        from scipy import ndimage

        # Label connected components for each time period
        labeled_mask_t1, num_components_t1 = ndimage.label(binary_mask_t1)
        labeled_mask_t2, num_components_t2 = ndimage.label(binary_mask_t2)

        # Calculate component sizes
        component_sizes_t1 = (
            np.bincount(labeled_mask_t1.flatten())[1:] if num_components_t1 > 0 else []
        )
        component_sizes_t2 = (
            np.bincount(labeled_mask_t2.flatten())[1:] if num_components_t2 > 0 else []
        )

        # Calculate forest fragmentation metrics
        fragmentation_metrics = {
            "forest_patches_t1": int(num_components_t1),
            "forest_patches_t2": int(num_components_t2),
            "change_in_patches": int(num_components_t2 - num_components_t1),
            "largest_patch_percentage_t1": (
                float(f"{np.max(component_sizes_t1) / binary_mask_t1.size * 100:.2f}")
                if component_sizes_t1.size > 0
                else 0
            ),
            "largest_patch_percentage_t2": (
                float(f"{np.max(component_sizes_t2) / binary_mask_t2.size * 100:.2f}")
                if component_sizes_t2.size > 0
                else 0
            ),
            "change_in_largest_patch": float(
                f"{(np.max(component_sizes_t2) if component_sizes_t2.size > 0 else 0) / binary_mask_t2.size * 100 - (np.max(component_sizes_t1) if component_sizes_t1.size > 0 else 0) / binary_mask_t1.size * 100:.2f}"
            ),
        }

        # Calculate forest edge metrics
        # Create edge masks using morphological operations
        if np.any(binary_mask_t1):
            eroded_t1 = ndimage.binary_erosion(binary_mask_t1)
            edge_mask_t1 = binary_mask_t1 & ~eroded_t1
            edge_percentage_t1 = np.mean(edge_mask_t1) * 100
        else:
            edge_percentage_t1 = 0

        if np.any(binary_mask_t2):
            eroded_t2 = ndimage.binary_erosion(binary_mask_t2)
            edge_mask_t2 = binary_mask_t2 & ~eroded_t2
            edge_percentage_t2 = np.mean(edge_mask_t2) * 100
        else:
            edge_percentage_t2 = 0

        edge_metrics = {
            "forest_edge_percentage_t1": float(f"{edge_percentage_t1:.2f}"),
            "forest_edge_percentage_t2": float(f"{edge_percentage_t2:.2f}"),
            "change_in_edge_percentage": float(
                f"{edge_percentage_t2 - edge_percentage_t1:.2f}"
            ),
        }
    except ImportError:
        print("SciPy not found. Skipping fragmentation and edge metrics.")
    except Exception as e:
        print(f"Error calculating fragmentation metrics: {e}")

    # Create basic change metrics
    basic_change_metrics = {
        "forest_coverage_t1": forest_percentage_t1,
        "forest_coverage_t2": forest_percentage_t2,
        "absolute_change": float(f"{forest_change:.2f}"),
        "relative_change_percentage": float(f"{forest_change_relative:.2f}"),
        "deforested_percentage": float(f"{deforested_percentage:.2f}"),
        "reforested_percentage": float(f"{reforested_percentage:.2f}"),
    }

    # Create difference image as an approximation for change detection
    difference_image = np.abs(prediction_t2 - prediction_t1)

    # Calculate change metrics from difference image
    change_magnitude = np.mean(difference_image) * 100
    significant_change = (
        np.mean(difference_image > 0.2) * 100
    )  # 20% threshold for significant change

    change_metrics = {
        "mean_change_magnitude": float(f"{change_magnitude:.2f}"),
        "significant_change_percentage": float(f"{significant_change:.2f}"),
    }

    # Combine all metrics
    metrics = {
        "basic_change_metrics": basic_change_metrics,
        "change_metrics": change_metrics,
    }

    if fragmentation_metrics:
        metrics["fragmentation_metrics"] = fragmentation_metrics

    if edge_metrics:
        metrics["edge_metrics"] = edge_metrics

    return metrics, {
        "binary_mask_t1": binary_mask_t1,
        "binary_mask_t2": binary_mask_t2,
        "prediction_t1": prediction_t1,
        "prediction_t2": prediction_t2,
        "change_prediction": difference_image,
    }


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Forest Change Metrics Extraction")
    parser.add_argument(
        "--forest_model", required=True, help="Path to the forest detection model"
    )
    parser.add_argument("--image_t1", help="Path to the first image (earlier time)")
    parser.add_argument("--image_t2", help="Path to the second image (later time)")
    parser.add_argument("--dataset_path", help="Path to the dataset file (.npy)")
    parser.add_argument(
        "--index_t1", type=int, help="Index of the first image in dataset"
    )
    parser.add_argument(
        "--index_t2", type=int, help="Index of the second image in dataset"
    )
    parser.add_argument(
        "--output_path", required=True, help="Path to save the output JSON metrics"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Save visualization of the change"
    )
    parser.add_argument(
        "--viz_path", default="change_prediction.png", help="Path to save visualization"
    )

    args = parser.parse_args()

    # Validate arguments
    if (args.image_t1 is None or args.image_t2 is None) and args.dataset_path is None:
        parser.error("Either --image_t1 and --image_t2 OR --dataset_path is required")

    if args.dataset_path is not None and (
        args.index_t1 is None or args.index_t2 is None
    ):
        parser.error("--index_t1 and --index_t2 are required when using --dataset_path")

    # Create output directory if needed
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Load forest model only
    forest_model = load_model_with_custom_metrics(args.forest_model)

    # Load images
    if args.image_t1 and args.image_t2:
        image_t1 = load_image(args.image_t1)  # Earlier time
        image_t2 = load_image(args.image_t2)  # Later time
    else:
        image_t1 = extract_from_dataset(
            args.dataset_path, args.index_t1
        )  # Earlier time
        image_t2 = extract_from_dataset(args.dataset_path, args.index_t2)  # Later time

    # Analyze forest change using forest model only
    metrics, predictions = analyze_forest_change(forest_model, image_t1, image_t2)

    # Save visualization if requested
    if args.visualize:
        save_change_visualization(image_t1[0], image_t2[0], predictions, args.viz_path)

    # Save metrics to JSON
    with open(args.output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Change metrics saved to {args.output_path}")
    print("\n=== Forest Change Summary ===")
    print(
        f"Time 1 Forest Coverage: {metrics['basic_change_metrics']['forest_coverage_t1']:.2f}%"
    )
    print(
        f"Time 2 Forest Coverage: {metrics['basic_change_metrics']['forest_coverage_t2']:.2f}%"
    )
    print(f"Absolute Change: {metrics['basic_change_metrics']['absolute_change']:.2f}%")
    print(
        f"Deforested Area: {metrics['basic_change_metrics']['deforested_percentage']:.2f}%"
    )
    print(
        f"Reforested Area: {metrics['basic_change_metrics']['reforested_percentage']:.2f}%"
    )

    if "fragmentation_metrics" in metrics:
        print(
            f"\nForest Patches (T1): {metrics['fragmentation_metrics']['forest_patches_t1']}"
        )
        print(
            f"Forest Patches (T2): {metrics['fragmentation_metrics']['forest_patches_t2']}"
        )

    print(
        f"\nMean Change Magnitude: {metrics['change_metrics']['mean_change_magnitude']:.2f}%"
    )
    print(
        f"Significant Change Area: {metrics['change_metrics']['significant_change_percentage']:.2f}%"
    )

    if args.visualize:
        print(f"\nVisualization saved to {args.viz_path}")

if __name__ == "__main__":
    main()
