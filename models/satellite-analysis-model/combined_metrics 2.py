"""
verify_carbon_claim.py

This script combines both forest detection and change detection models to verify carbon credit claims
and produce a comprehensive verification report.

Usage:
  python verify_carbon_claim.py --forest_model path/to/forest_model.h5 
                              --change_model path/to/change_model.h5
                              --image_current path/to/current_image.npy
                              --image_previous path/to/previous_image.npy
                              --claimed_credits 1000
                              --output_dir path/to/output
                              --area 100
                              --coords "37.7749,-122.4194"

Arguments:
  --forest_model      Path to the trained forest detection model (.h5)
  --change_model      Path to the trained change detection model (.h5)
  --image_current     Path to the current satellite image (.npy)
  --image_previous    Path to the previous satellite image (.npy, optional)
  --claimed_credits   Number of carbon credits claimed (tCO2)
  --output_dir        Directory to save the verification results
  --area              Area in hectares (default: 1.0)
  --coords            Comma-separated lat,lon coordinates (optional)
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.models import load_model


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
    """Load model with custom metrics"""
    custom_objects = {
        'iou_threshold_0.1': make_iou_threshold(0.1),
        'iou_threshold_0.3': make_iou_threshold(0.3),
        'iou_threshold_0.5': make_iou_threshold(0.5),
        'custom_iou': custom_iou,
        'make_iou_threshold': make_iou_threshold
    }
    
    print(f"Loading model from {model_path}")
    model = load_model(model_path, custom_objects=custom_objects)
    print("Model loaded successfully")
    return model


def load_satellite_image(image_path):
    """Load satellite image from .npy file"""
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


def analyze_forest_coverage(model, image, thresholds=[0.1, 0.3, 0.5, 0.7, 0.9]):
    """Analyze forest coverage in a satellite image"""
    print("Analyzing forest coverage...")
    
    # Get model prediction
    prediction = model.predict(image)
    
    # Calculate forest coverage percentage at different thresholds
    coverage_metrics = {}
    
    for threshold in thresholds:
        # Apply threshold to get binary mask
        binary_mask = prediction[0, :, :, 0] > threshold
        
        # Calculate percentage of forest coverage
        forest_percentage = np.mean(binary_mask) * 100
        
        coverage_metrics[f"forest_coverage_threshold_{threshold}"] = float(f"{forest_percentage:.2f}")
    
    # Calculate average forest coverage across all thresholds
    coverage_metrics["average_forest_coverage"] = float(f"{np.mean(list(coverage_metrics.values())):.2f}")
    
    return coverage_metrics, prediction


def analyze_forest_change(change_model, forest_model, image_t1, image_t2):
    """Analyze forest cover change between two time periods"""
    print("Analyzing forest change...")
    
    # Get forest coverage for each time period using forest detection model
    coverage_metrics_t1, prediction_t1 = analyze_forest_coverage(forest_model, image_t1)
    coverage_metrics_t2, prediction_t2 = analyze_forest_coverage(forest_model, image_t2)
    
    # Apply threshold (0.5) to get binary masks
    binary_mask_t1 = prediction_t1[0, :, :, 0] > 0.5
    binary_mask_t2 = prediction_t2[0, :, :, 0] > 0.5
    
    # Calculate forest percentage for each time period
    forest_percentage_t1 = coverage_metrics_t1["average_forest_coverage"]
    forest_percentage_t2 = coverage_metrics_t2["average_forest_coverage"]
    
    # Calculate change in forest coverage
    forest_change = forest_percentage_t2 - forest_percentage_t1
    forest_change_relative = (forest_change / (forest_percentage_t1 + 1e-8)) * 100
    
    # Use change detection model if available
    change_prediction = None
    try:
        # This depends on your change detection model's input format
        # Adjust as needed for your specific model
        try:
            # Try stacking images along batch dimension
            change_input = np.concatenate([image_t1, image_t2], axis=0)
            change_prediction = change_model.predict(change_input)
        except:
            # Try concatenating along channel dimension
            combined = np.concatenate([image_t1[0], image_t2[0]], axis=-1)
            combined = np.expand_dims(combined, axis=0)
            change_prediction = change_model.predict(combined)
        
        print("Change detection model prediction successful")
    except Exception as e:
        print(f"Error with change detection model: {e}")
        print("Using forest detection results to calculate change metrics")
    
    # Calculate deforested and reforested areas
    deforested_mask = np.logical_and(binary_mask_t1, np.logical_not(binary_mask_t2))
    deforested_percentage = np.mean(deforested_mask) * 100
    
    reforested_mask = np.logical_and(np.logical_not(binary_mask_t1), binary_mask_t2)
    reforested_percentage = np.mean(reforested_mask) * 100
    
    # Create change metrics dictionary
    change_metrics = {
        "forest_coverage_t1": forest_percentage_t1,
        "forest_coverage_t2": forest_percentage_t2,
        "absolute_change": float(f"{forest_change:.2f}"),
        "relative_change_percentage": float(f"{forest_change_relative:.2f}"),
        "deforested_percentage": float(f"{deforested_percentage:.2f}"),
        "reforested_percentage": float(f"{reforested_percentage:.2f}")
    }
    
    return change_metrics, {
        "binary_mask_t1": binary_mask_t1,
        "binary_mask_t2": binary_mask_t2,
        "prediction_t1": prediction_t1[0, :, :, 0],
        "prediction_t2": prediction_t2[0, :, :, 0],
        "change_prediction": change_prediction
    }


def calculate_carbon_sequestration(forest_coverage, area_hectares=1.0, carbon_density=250):
    """Calculate estimated carbon sequestration based on forest coverage"""
    # Forest area in hectares
    forest_area = (forest_coverage / 100) * area_hectares
    
    # Carbon sequestration calculation
    carbon_sequestration = forest_area * carbon_density
    
    return float(f"{carbon_sequestration:.2f}")


def evaluate_carbon_claim(claimed_credits, calculated_credits):
    """Evaluate the legitimacy of a carbon credit claim"""
    # Calculate absolute difference
    absolute_difference = abs(claimed_credits - calculated_credits)
    
    # Calculate relative difference
    relative_difference = (absolute_difference / (calculated_credits + 1e-8)) * 100
    
    # Legitimacy score (0-100)
    # Higher score means more legitimate claim
    # Using exponential decay to penalize larger differences
    legitimacy_score = 100 * np.exp(-0.05 * relative_difference)
    
    # Classification based on legitimacy score
    if legitimacy_score >= 90:
        classification = "Highly Legitimate"
    elif legitimacy_score >= 70:
        classification = "Legitimate"
    elif legitimacy_score >= 50:
        classification = "Questionable"
    elif legitimacy_score >= 30:
        classification = "Suspicious"
    else:
        classification = "Fraudulent"
    
    # Create evaluation metrics dictionary
    evaluation_metrics = {
        "claimed_carbon_credits": float(f"{claimed_credits:.2f}"),
        "calculated_carbon_credits": float(f"{calculated_credits:.2f}"),
        "absolute_difference": float(f"{absolute_difference:.2f}"),
        "relative_difference_percentage": float(f"{relative_difference:.2f}"),
        "legitimacy_score": float(f"{legitimacy_score:.2f}"),
        "classification": classification
    }
    
    return evaluation_metrics


def save_verification_visualization(image_current, image_previous, predictions, metrics, output_path):
    """Save comprehensive verification visualization"""
    fig = plt.figure(figsize=(15, 12))
    
    # Title with overall result
    classification = metrics["carbon_credit_evaluation"]["classification"]
    legitimacy_score = metrics["carbon_credit_evaluation"]["legitimacy_score"]
    fig.suptitle(f"Carbon Credit Verification Result: {classification} (Score: {legitimacy_score:.1f}/100)", 
                fontsize=16, fontweight='bold')
    
    # Current forest coverage (top left)
    plt.subplot(2, 3, 1)
    if image_current.shape[2] >= 3:
        plt.imshow(np.clip(image_current[:, :, :3], 0, 1))
    else:
        plt.imshow(image_current[:, :, 0], cmap="gray")
    plt.title("Current Satellite Image")
    plt.axis("off")
    
    # Current forest mask (top middle)
    plt.subplot(2, 3, 2)
    plt.imshow(predictions["prediction_t2"], cmap="viridis")
    plt.title(f"Current Forest Coverage: {metrics['forest_change_metrics']['forest_coverage_t2']:.1f}%")
    plt.colorbar(label="Probability")
    plt.axis("off")
    
    # Change map (top right)
    plt.subplot(2, 3, 3)
    # Create change map
    change_map = np.zeros_like(predictions["binary_mask_t1"], dtype=np.uint8)
    change_map[np.logical_and(~predictions["binary_mask_t1"], ~predictions["binary_mask_t2"])] = 0  # No change (non-forest)
    change_map[np.logical_and(predictions["binary_mask_t1"], predictions["binary_mask_t2"])] = 1    # No change (forest)
    change_map[np.logical_and(predictions["binary_mask_t1"], ~predictions["binary_mask_t2"])] = 2   # Deforestation
    change_map[np.logical_and(~predictions["binary_mask_t1"], predictions["binary_mask_t2"])] = 3   # Reforestation
    
    cmap = plt.cm.get_cmap('viridis', 4)
    plt.imshow(change_map, cmap=cmap, vmin=0, vmax=3)
    plt.title("Forest Change Map")
    cbar = plt.colorbar(ticks=[0.5, 1.5, 2.5, 3.5])
    cbar.set_ticklabels(['No Forest', 'Stable Forest', 'Deforestation', 'Reforestation'])
    plt.axis("off")
    
    # Previous forest coverage (bottom left)
    plt.subplot(2, 3, 4)
    if image_previous is not None:
        if image_previous.shape[2] >= 3:
            plt.imshow(np.clip(image_previous[:, :, :3], 0, 1))
        else:
            plt.imshow(image_previous[:, :, 0], cmap="gray")
        plt.title("Previous Satellite Image")
    else:
        plt.text(0.5, 0.5, "No previous image available", 
                 horizontalalignment='center', verticalalignment='center')
        plt.title("Previous Image")
    plt.axis("off")
    
    # Previous forest mask (bottom middle)
    plt.subplot(2, 3, 5)
    if image_previous is not None:
        plt.imshow(predictions["prediction_t1"], cmap="viridis")
        plt.title(f"Previous Forest Coverage: {metrics['forest_change_metrics']['forest_coverage_t1']:.1f}%")
        plt.colorbar(label="Probability")
    else:
        plt.text(0.5, 0.5, "No previous mask available", 
                 horizontalalignment='center', verticalalignment='center')
        plt.title("Previous Forest Mask")
    plt.axis("off")
    
    # Metrics summary (bottom right)
    plt.subplot(2, 3, 6)
    plt.axis("off")
    
    # Create text box with key metrics
    metrics_text = (
        f"CARBON CREDIT VERIFICATION\n"
        f"-------------------------\n"
        f"Claimed Credits: {metrics['carbon_credit_evaluation']['claimed_carbon_credits']:.1f} tCO2\n"
        f"Calculated Credits: {metrics['carbon_credit_evaluation']['calculated_carbon_credits']:.1f} tCO2\n"
        f"Difference: {metrics['carbon_credit_evaluation']['absolute_difference']:.1f} tCO2\n"
        f"({metrics['carbon_credit_evaluation']['relative_difference_percentage']:.1f}%)\n\n"
    )
    
    if 'forest_change_metrics' in metrics:
        metrics_text += (
            f"FOREST CHANGE METRICS\n"
            f"--------------------\n"
            f"Absolute Change: {metrics['forest_change_metrics']['absolute_change']:.1f}%\n"
            f"Relative Change: {metrics['forest_change_metrics']['relative_change_percentage']:.1f}%\n"
            f"Deforestation: {metrics['forest_change_metrics']['deforested_percentage']:.1f}%\n"
            f"Reforestation: {metrics['forest_change_metrics']['reforested_percentage']:.1f}%\n"
        )
    
    plt.text(0.05, 0.95, metrics_text, horizontalalignment='left', 
             verticalalignment='top', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Verification visualization saved to {output_path}")


def save_metrics_json(metrics, output_path):
    """Save metrics to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {output_path}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Carbon Credit Verification')
    parser.add_argument('--forest_model', required=True, help='Path to the forest detection model')
    parser.add_argument('--change_model', required=True, help='Path to the change detection model')
    parser.add_argument('--image_current', required=True, help='Path to the current satellite image')
    parser.add_argument('--image_previous', help='Path to the previous satellite image (optional)')
    parser.add_argument('--claimed_credits', type=float, required=True, help='Number of carbon credits claimed')
    parser.add_argument('--output_dir', required=True, help='Directory to save results')
    parser.add_argument('--area', type=float, default=1.0, help='Area in hectares')
    parser.add_argument('--coords', help='Comma-separated lat,lon coordinates')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    forest_model = load_model_with_custom_metrics(args.forest_model)
    change_model = load_model_with_custom_metrics(args.change_model)
    
    # Load current image
    image_current = load_satellite_image(args.image_current)
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Analyze current forest coverage
    coverage_metrics, prediction = analyze_forest_coverage(forest_model, image_current)
    
    # Calculate carbon sequestration based on current forest coverage
    calculated_credits = calculate_carbon_sequestration(
        coverage_metrics["average_forest_coverage"], 
        area_hectares=args.area
    )
    
    # Evaluate carbon credit claim
    evaluation_metrics = evaluate_carbon_claim(args.claimed_credits, calculated_credits)
    
    # Initialize metrics dictionary
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "coordinates": None,
        "area_hectares": args.area,
        "forest_coverage_metrics": coverage_metrics,
        "carbon_credit_evaluation": evaluation_metrics
    }
    
    # Add coordinates if provided
    if args.coords:
        try:
            lat, lon = map(float, args.coords.split(','))
            metrics["coordinates"] = [lat, lon]
        except:
            print("Invalid coordinates format. Expected lat,lon")
    
    # Change detection if previous image is provided
    image_previous = None
    predictions = {
        "prediction_t2": prediction[0, :, :, 0],
        "binary_mask_t2": prediction[0, :, :, 0] > 0.5,
        "prediction_t1": None,
        "binary_mask_t1": None,
        "change_prediction": None
    }
    
    if args.image_previous:
        print("Analyzing historical change...")
        image_previous = load_satellite_image(args.image_previous)
        
        # Analyze forest change
        change_metrics, predictions = analyze_forest_change(
            change_model, 
            forest_model, 
            image_previous,  # Earlier time
            image_current    # Current time
        )
        
        # Add change metrics to results
        metrics["forest_change_metrics"] = change_metrics
    
    # Save comprehensive visualization
    viz_path = os.path.join(args.output_dir, f"carbon_verification_{timestamp}.png")
    save_verification_visualization(
        image_current[0], 
        image_previous[0] if image_previous is not None else None,
        predictions,
        metrics,
        viz_path
    )
    
    # Save metrics to JSON
    json_path = os.path.join(args.output_dir, f"carbon_verification_{timestamp}.json")
    save_metrics_json(metrics, json_path)
    
    # Print summary
    print("\n======= Carbon Credit Verification Summary =======")
    print(f"Claimed carbon credits: {args.claimed_credits:.2f} tCO2")
    print(f"Calculated carbon credits: {calculated_credits:.2f} tCO2")
    print(f"Legitimacy score: {evaluation_metrics['legitimacy_score']:.2f}/100")
    print(f"Classification: {evaluation_metrics['classification']}")
    
    if args.image_previous:
        forest_change = metrics["forest_change_metrics"]["absolute_change"]
        print(f"\nForest coverage change: {forest_change:.2f}%")
        
        if forest_change > 0:
            print("Forest coverage has increased ✓")
        else:
            print("Forest coverage has decreased ✗")
    
    print("\nVerification report saved to:")
    print(f" - JSON: {json_path}")
    print(f" - Image: {viz_path}")
    print("================================================\n")


if __name__ == "__main__":
    main()