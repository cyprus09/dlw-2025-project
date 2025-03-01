"""
This script uses Google Earth Engine to fetch satellite imagery for specific coordinates
and years, then analyzes forest changes using the trained forest detection model.

Usage:
  python gee_forest_change.py --forest_model path/to/model.h5 
                             --latitude 37.7749 --longitude -122.4194 
                             --year1 2019 --year2 2022
                             --output_dir results/
"""

import os
import json
import argparse
import tempfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.models import load_model
import ee
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

try:
    ee.Initialize(project="dlw-hackathon-452319")
    print("Earth Engine initialized successfully")
except Exception as e:
    print(f"Error initializing Earth Engine: {e}")
    print("You may need to authenticate first with 'earthengine authenticate'")
    exit(1)


# Custom metrics for model loading
def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice


def custom_iou(y_true, y_pred, threshold=0.5):
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


def preprocess_input(x):
    """Dummy preprocessing function"""
    return x


def load_forest_detection_model(model_path):
    """Load the forest detection model with custom metrics"""
    custom_objects = {
        "iou_threshold_0.1": make_iou_threshold(0.1),
        "iou_threshold_0.3": make_iou_threshold(0.3),
        "iou_threshold_0.5": make_iou_threshold(0.5),
        "custom_iou": custom_iou,
        "make_iou_threshold": make_iou_threshold,
        "dice_loss": dice_loss,
        "combined_loss": combined_loss,
        "preprocess_input": preprocess_input,
    }

    print(f"Loading model from {model_path}")
    try:
        model = load_model(model_path, custom_objects=custom_objects)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model with custom objects: {e}")
        print("Attempting to load with compile=False...")

        try:
            model = load_model(model_path, compile=False, custom_objects=custom_objects)
            print("Model loaded without compilation")

            # Recompile with standard metrics
            model.compile(
                optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
            )
            print("Model recompiled")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise


def get_sentinel_image(
    latitude, longitude, year, buffer_size=1000, max_cloud_percent=20, patch_size=64
):
    # Define point and region
    point = ee.Geometry.Point([longitude, latitude])
    region = point.buffer(buffer_size)

    # Define date range for the specified year
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    # Get Sentinel-2 surface reflectance collection
    s2_collection = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_percent))
    )

    # Check if we have images
    image_count = s2_collection.size().getInfo()
    if image_count == 0:
        print(
            f"No images found for {latitude}, {longitude} in {year} with cloud percentage < {max_cloud_percent}%"
        )
        print("Trying with higher cloud percentage...")

        # Try with higher cloud percentage
        s2_collection = (
            ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(region)
            .filterDate(start_date, end_date)
        )

        image_count = s2_collection.size().getInfo()
        if image_count == 0:
            raise ValueError(f"No images found for {latitude}, {longitude} in {year}")

    # Get the least cloudy image
    image = s2_collection.sort("CLOUDY_PIXEL_PERCENTAGE").first()

    # Select bands: RGB + NIR
    selected_bands = image.select(["B4", "B3", "B2", "B8"])

    # Get image dimensions - This is to handle non-square regions
    bounds = region.bounds().getInfo()["coordinates"][0]
    x_coords = [p[0] for p in bounds]
    y_coords = [p[1] for p in bounds]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Define a rectangular region
    rectangle = ee.Geometry.Rectangle([min_x, min_y, max_x, max_y])

    # Export the image to a fixed-size square
    proj = ee.Projection("EPSG:4326").atScale(10)  # Use WGS84 at 10m resolution

    # Create a square export region centered on the point
    side_length = buffer_size * 2  # in meters
    export_region = point.buffer(side_length / 2).bounds()

    try:
        print("Fetching image data...")
        array_data = selected_bands.sampleRectangle(region=export_region)
        array_data = array_data.getInfo()

        # Extract band arrays
        red_array = np.array(array_data["properties"]["B4"])
        green_array = np.array(array_data["properties"]["B3"])
        blue_array = np.array(array_data["properties"]["B2"])
        nir_array = np.array(array_data["properties"]["B8"])

        # Get the shape
        if red_array.ndim == 1:
            # Handle 1D arrays by reshaping them to square
            array_size = len(red_array)
            width = int(np.sqrt(array_size))
            height = width

            red_array = red_array.reshape(height, width)
            green_array = green_array.reshape(height, width)
            blue_array = blue_array.reshape(height, width)
            nir_array = nir_array.reshape(height, width)
        else:
            height, width = red_array.shape
    except Exception as e:
        print(f"Error with sampleRectangle: {e}")
        print("Trying alternative approach with getDownloadURL...")

        # Alternative: Export the image and download it
        try:
            # Define export parameters
            export_params = {
                "image": selected_bands,
                "description": f"sentinel2_{year}",
                "dimensions": f"{patch_size}x{patch_size}",
                "region": export_region,
                "format": "GEO_TIFF",
            }

            # Get download URL
            url = selected_bands.getDownloadURL(export_params)

            # Download the image
            import requests
            import tempfile
            import rasterio

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as temp_file:
                temp_path = temp_file.name

            print(f"Downloading image to {temp_path}...")
            response = requests.get(url)
            with open(temp_path, "wb") as f:
                f.write(response.content)

            # Read with rasterio
            with rasterio.open(temp_path) as src:
                # Read all bands
                bands = src.read()
                red_array = bands[0]
                green_array = bands[1]
                blue_array = bands[2]
                nir_array = bands[3]

                height, width = red_array.shape

            # Clean up temp file
            import os

            os.remove(temp_path)
        except Exception as e:
            print(f"Error with alternative approach: {e}")
            print("Trying with direct pixel sampling...")

            # Try one more approach - get a sample of pixels
            scale = 10  # meters per pixel
            num_pixels = patch_size

            # Create a grid of points covering the region
            grid = ee.ImageCollection.fromImages(
                [ee.Image.pixelLonLat().clip(export_region)]
            )

            # Sample the image values at each point
            samples = selected_bands.sampleRegions(
                collection=grid.geometry().coveringGrid("EPSG:4326", scale),
                properties=["B4", "B3", "B2", "B8"],
                scale=scale,
                geometries=True,
            ).getInfo()

            # Organize the samples into a grid
            red_array = np.zeros((num_pixels, num_pixels))
            green_array = np.zeros((num_pixels, num_pixels))
            blue_array = np.zeros((num_pixels, num_pixels))
            nir_array = np.zeros((num_pixels, num_pixels))

            for i, sample in enumerate(samples["features"]):
                props = sample["properties"]
                row = i // num_pixels
                col = i % num_pixels

                if row < num_pixels and col < num_pixels:
                    red_array[row, col] = props["B4"]
                    green_array[row, col] = props["B3"]
                    blue_array[row, col] = props["B2"]
                    nir_array[row, col] = props["B8"]

            height, width = red_array.shape

    print(f"Image data retrieved with shape: {height}x{width}")

    # Normalize data
    red_array = red_array / 10000.0  # Typical scaling for reflectance values
    green_array = green_array / 10000.0
    blue_array = blue_array / 10000.0
    nir_array = nir_array / 10000.0

    # Calculate NDVI: (NIR - Red) / (NIR + Red)
    ndvi = (nir_array - red_array) / (
        nir_array + red_array + 1e-8
    )  # Add small epsilon to avoid division by zero

    # Scale NDVI from [-1,1] to [0,1]
    ndvi = (ndvi + 1) / 2.0

    # Stack RGB bands
    rgb = np.stack([red_array, green_array, blue_array], axis=0)

    # Stack all features: RGB (3) + NDVI (1)
    features = np.vstack([rgb, ndvi[np.newaxis, :, :]])

    # Resize to patch_size if needed
    if height != patch_size or width != patch_size:
        from skimage.transform import resize

        resized_features = np.zeros((4, patch_size, patch_size))
        for i in range(4):
            resized_features[i] = resize(
                features[i], (patch_size, patch_size), preserve_range=True
            )
        features = resized_features

    # Reshape for Keras: (samples, height, width, channels)
    feature_patch = np.transpose(features, (1, 2, 0))  # (height, width, channels)
    feature_patch = np.expand_dims(
        feature_patch, axis=0
    )  # Add batch dimension: (1, height, width, channels)

    # Get the metadata
    metadata = {
        "latitude": latitude,
        "longitude": longitude,
        "year": year,
        "image_date": image.date().format("YYYY-MM-dd").getInfo(),
        "cloud_percentage": image.get("CLOUDY_PIXEL_PERCENTAGE").getInfo(),
        "image_id": image.id().getInfo(),
    }

    return {"features": feature_patch, "metadata": metadata}


def analyze_forest_coverage(model, image, thresholds=[0.1, 0.3, 0.5, 0.7, 0.9]):
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


def analyze_forest_change(forest_model, image_t1, image_t2):
    # Get metrics for each time period
    metrics_t1, prediction_t1 = analyze_forest_coverage(
        forest_model, image_t1["features"]
    )
    metrics_t2, prediction_t2 = analyze_forest_coverage(
        forest_model, image_t2["features"]
    )

    # Extract forest coverage at different thresholds
    coverage_t1 = metrics_t1["forest_coverage"]
    coverage_t2 = metrics_t2["forest_coverage"]

    # Calculate change in forest coverage
    coverage_change = {}
    for key in coverage_t1.keys():
        if (
            key.startswith("forest_coverage_threshold_")
            or key == "average_forest_coverage"
        ):
            change = coverage_t2[key] - coverage_t1[key]
            change_percent = (change / (coverage_t1[key] + 1e-8)) * 100
            coverage_change[f"{key}_change"] = float(f"{change:.2f}")
            coverage_change[f"{key}_change_percent"] = float(f"{change_percent:.2f}")

    # Calculate deforested and reforested areas at threshold 0.5
    binary_mask_t1 = prediction_t1[0, :, :, 0] > 0.5
    binary_mask_t2 = prediction_t2[0, :, :, 0] > 0.5

    deforested_mask = np.logical_and(binary_mask_t1, np.logical_not(binary_mask_t2))
    reforested_mask = np.logical_and(np.logical_not(binary_mask_t1), binary_mask_t2)

    deforested_percentage = np.mean(deforested_mask) * 100
    reforested_percentage = np.mean(reforested_mask) * 100

    # Create change metrics
    change_metrics = {
        "forest_coverage_t1": coverage_t1["average_forest_coverage"],
        "forest_coverage_t2": coverage_t2["average_forest_coverage"],
        "absolute_change": float(
            f"{coverage_t2['average_forest_coverage'] - coverage_t1['average_forest_coverage']:.2f}"
        ),
        "relative_change_percentage": float(
            f"{((coverage_t2['average_forest_coverage'] - coverage_t1['average_forest_coverage']) / (coverage_t1['average_forest_coverage'] + 1e-8)) * 100:.2f}"
        ),
        "deforested_percentage": float(f"{deforested_percentage:.2f}"),
        "reforested_percentage": float(f"{reforested_percentage:.2f}"),
        "detailed_changes": coverage_change,
    }

    # Create change details with metadata
    change_details = {
        "change_metrics": change_metrics,
        "metadata": {
            "location": {
                "latitude": image_t1["metadata"]["latitude"],
                "longitude": image_t1["metadata"]["longitude"],
            },
            "time_period": {
                "start_year": image_t1["metadata"]["year"],
                "start_date": image_t1["metadata"]["image_date"],
                "end_year": image_t2["metadata"]["year"],
                "end_date": image_t2["metadata"]["image_date"],
            },
        },
        "metrics_t1": metrics_t1,
        "metrics_t2": metrics_t2,
    }

    return change_details, {
        "prediction_t1": prediction_t1[0, :, :, 0],
        "prediction_t2": prediction_t2[0, :, :, 0],
        "binary_mask_t1": binary_mask_t1,
        "binary_mask_t2": binary_mask_t2,
        "features_t1": image_t1["features"][0],
        "features_t2": image_t2["features"][0],
    }


def save_change_visualization(predictions, output_path, metadata=None):
    # Create figure
    plt.figure(figsize=(15, 10))

    # If we have metadata, add a title
    if metadata:
        plt.suptitle(
            f"Forest Change Analysis: {metadata['location']['latitude']}, {metadata['location']['longitude']}\n"
            f"Period: {metadata['time_period']['start_year']} to {metadata['time_period']['end_year']}",
            fontsize=16,
            y=0.98,
        )

    # Extract data
    features_t1 = predictions["features_t1"]
    features_t2 = predictions["features_t2"]
    prediction_t1 = predictions["prediction_t1"]
    prediction_t2 = predictions["prediction_t2"]
    binary_mask_t1 = predictions["binary_mask_t1"]
    binary_mask_t2 = predictions["binary_mask_t2"]

    # Calculate change mask
    change_map = np.zeros_like(binary_mask_t1, dtype=np.uint8)
    change_map[np.logical_and(~binary_mask_t1, ~binary_mask_t2)] = (
        0 
    )
    change_map[np.logical_and(binary_mask_t1, binary_mask_t2)] = 1
    change_map[np.logical_and(binary_mask_t1, ~binary_mask_t2)] = 2
    change_map[np.logical_and(~binary_mask_t1, binary_mask_t2)] = 3

    # Time 1 (Earlier)
    plt.subplot(2, 4, 1)
    plt.imshow(np.clip(features_t1[:, :, :3], 0, 1))
    plt.title(f"RGB Image ({metadata['time_period']['start_year']})")
    plt.axis("off")

    plt.subplot(2, 4, 2)
    plt.imshow(features_t1[:, :, 3], cmap="viridis")
    plt.title(f"NDVI ({metadata['time_period']['start_year']})")
    plt.axis("off")

    plt.subplot(2, 4, 3)
    plt.imshow(prediction_t1, cmap="viridis")
    plt.title(f"Forest Prediction ({metadata['time_period']['start_year']})")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.subplot(2, 4, 4)
    plt.imshow(binary_mask_t1, cmap="Greens")
    plt.title(f"Forest Mask ({metadata['time_period']['start_year']})")
    plt.axis("off")

    # Time 2 (Later)
    plt.subplot(2, 4, 5)
    plt.imshow(np.clip(features_t2[:, :, :3], 0, 1))
    plt.title(f"RGB Image ({metadata['time_period']['end_year']})")
    plt.axis("off")

    plt.subplot(2, 4, 6)
    plt.imshow(features_t2[:, :, 3], cmap="viridis")
    plt.title(f"NDVI ({metadata['time_period']['end_year']})")
    plt.axis("off")

    plt.subplot(2, 4, 7)
    plt.imshow(prediction_t2, cmap="viridis")
    plt.title(f"Forest Prediction ({metadata['time_period']['end_year']})")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    # Change Map
    plt.subplot(2, 4, 8)
    cmap = plt.cm.get_cmap("viridis", 4)
    plt.imshow(change_map, cmap=cmap, vmin=0, vmax=3)
    plt.title("Forest Change Map")
    cbar = plt.colorbar(ticks=[0.5, 1.5, 2.5, 3.5], fraction=0.046, pad=0.04)
    cbar.set_ticklabels(
        ["No Forest", "Stable Forest", "Deforestation", "Reforestation"]
    )
    plt.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze forest change using Google Earth Engine and ML models"
    )
    parser.add_argument(
        "--forest_model", required=True, help="Path to the forest detection model"
    )
    parser.add_argument(
        "--latitude", type=float, required=True, help="Latitude coordinate"
    )
    parser.add_argument(
        "--longitude", type=float, required=True, help="Longitude coordinate"
    )
    parser.add_argument("--year1", type=int, required=True, help="First year (earlier)")
    parser.add_argument(
        "--year2",
        type=int,
        default=None,
        help="Second year (later, defaults to year1+3)",
    )
    parser.add_argument(
        "--buffer", type=int, default=1000, help="Buffer size in meters (default: 1000)"
    )
    parser.add_argument(
        "--cloud_percent",
        type=int,
        default=20,
        help="Maximum cloud percentage (default: 20)",
    )
    parser.add_argument(
        "--output_dir", default="results", help="Output directory (default: results)"
    )

    args = parser.parse_args()

    if args.year2 is None:
        args.year2 = args.year1 + 3

    os.makedirs(args.output_dir, exist_ok=True)

    # Load forest detection model
    forest_model = load_forest_detection_model(args.forest_model)

    # Fetch satellite images
    print(
        f"Fetching satellite image for {args.latitude}, {args.longitude} in {args.year1}..."
    )
    image_t1 = get_sentinel_image(
        args.latitude,
        args.longitude,
        args.year1,
        buffer_size=args.buffer,
        max_cloud_percent=args.cloud_percent,
    )

    print(
        f"Fetching satellite image for {args.latitude}, {args.longitude} in {args.year2}..."
    )
    image_t2 = get_sentinel_image(
        args.latitude,
        args.longitude,
        args.year2,
        buffer_size=args.buffer,
        max_cloud_percent=args.cloud_percent,
    )

    # Analyze forest change
    print("Analyzing forest change...")
    change_details, predictions = analyze_forest_change(
        forest_model, image_t1, image_t2
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_file = os.path.join(
        args.output_dir,
        f"forest_change_{args.latitude}_{args.longitude}_{args.year1}_{args.year2}_{timestamp}.json",
    )

    with open(results_file, "w") as f:
        json.dump(change_details, f, indent=2)

    print(f"Results saved to {results_file}")

    # Save visualization
    viz_file = os.path.join(
        args.output_dir,
        f"forest_change_{args.latitude}_{args.longitude}_{args.year1}_{args.year2}_{timestamp}.png",
    )

    save_change_visualization(predictions, viz_file, change_details["metadata"])

    print("\n=== Forest Change Analysis ===")
    print(f"Location: {args.latitude}, {args.longitude}")
    print(f"Period: {args.year1} to {args.year2}")
    print(
        f"Forest Coverage ({args.year1}): {change_details['change_metrics']['forest_coverage_t1']:.2f}%"
    )
    print(
        f"Forest Coverage ({args.year2}): {change_details['change_metrics']['forest_coverage_t2']:.2f}%"
    )
    print(
        f"Absolute Change: {change_details['change_metrics']['absolute_change']:.2f}%"
    )
    print(
        f"Relative Change: {change_details['change_metrics']['relative_change_percentage']:.2f}%"
    )
    print(
        f"Deforested Area: {change_details['change_metrics']['deforested_percentage']:.2f}%"
    )
    print(
        f"Reforested Area: {change_details['change_metrics']['reforested_percentage']:.2f}%"
    )
    print("==============================")


if __name__ == "__main__":
    main()
