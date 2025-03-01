import os
import numpy as np
import rasterio
import tensorflow as tf
from skimage.transform import resize
import pandas as pd

from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    def __init__(self, historical_data):
        """
        Initialize anomaly detection based on historical forest data

        Args:
            historical_data (pd.DataFrame): Historical forest cover data
        """
        # Prepare features for anomaly detection
        features = historical_data[["forest_percentage", "ndvi", "evi", "savi"]]

        # Train Isolation Forest
        self.detector = IsolationForest(
            contamination=0.1, random_state=42  # Assume 10% of data might be anomalous
        )
        self.detector.fit(features)

    def detect_anomalies(self, current_data):
        """
        Detect anomalies in current forest data

        Args:
            current_data (dict): Current forest metrics

        Returns:
            dict: Anomaly detection results
        """
        # Prepare current data for prediction
        features = [
            current_data["forest_percentage"],
            current_data["ndvi"],
            current_data["evi"],
            current_data["savi"],
        ]

        # Predict anomaly
        anomaly_prediction = self.detector.predict([features])

        return {
            "is_anomaly": anomaly_prediction[0] == -1,
            "anomaly_score": self.detector.score_samples([features])[0],
        }

    def comprehensive_inference(self, rgb_path, ndvi_path, historical_data=None):
        """
        Perform comprehensive forest analysis

        Args:
            rgb_path (str): Path to RGB image
            ndvi_path (str): Path to NDVI image
            historical_data (list, optional): Historical image data

        Returns:
            dict: Comprehensive forest analysis
        """
        # Basic forest cover prediction
        forest_cover = self.predict_forest_cover(rgb_path, ndvi_path)

        # Advanced vegetation indices
        vegetation_indices = self.calculate_advanced_vegetation_indices(
            rgb_path, ndvi_path
        )

        # Landscape context
        landscape_context = self.analyze_landscape_context(rgb_path, ndvi_path)

        # Temporal analysis (if historical data provided)
        temporal_analysis = (
            self.analyze_temporal_changes(historical_data) if historical_data else None
        )

        # Anomaly detection (if historical data provided)
        anomaly_detection = None
        if historical_data:
            anomaly_detector = AnomalyDetector(historical_data)
            anomaly_detection = anomaly_detector.detect_anomalies(
                {**forest_cover, **vegetation_indices}
            )

        # Combine all analyses
        comprehensive_report = {
            "forest_cover": forest_cover,
            "vegetation_indices": vegetation_indices,
            "landscape_context": landscape_context,
            "temporal_analysis": temporal_analysis,
            "anomaly_detection": anomaly_detection,
        }

        return comprehensive_report


class ForestCoverInference:
    def __init__(self, model_path, metadata_path=None):
        """
        Initialize forest cover inference pipeline

        Args:
            model_path (str): Path to trained model (.h5 file)
            metadata_path (str, optional): Path to metadata CSV
        """
        # Load trained model
        self.model = tf.keras.models.load_model(
            model_path, custom_objects={"iou_threshold": self._iou_threshold}
        )

        # Load metadata if provided
        self.metadata = pd.read_csv(metadata_path) if metadata_path else None

    def _iou_threshold(self, y_true, y_pred, threshold=0.1):
        """
        Intersection over Union (IoU) metric with a specific threshold

        Args:
            y_true (tf.Tensor): Ground truth mask
            y_pred (tf.Tensor): Predicted mask
            threshold (float): Threshold for binary classification

        Returns:
            tf.Tensor: IoU score
        """
        y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        intersection = tf.reduce_sum(y_pred_binary * y_true)
        union = tf.reduce_sum(y_pred_binary + y_true) - intersection

        iou = tf.math.divide_no_nan(intersection, union)
        return iou

    def preprocess_image(self, rgb_path, ndvi_path, patch_size=64):
        """
        Preprocess satellite image for forest cover prediction

        Args:
            rgb_path (str): Path to RGB GeoTIFF
            ndvi_path (str): Path to NDVI GeoTIFF
            patch_size (int): Size of image patch to extract

        Returns:
            np.ndarray: Preprocessed image features
        """
        try:
            # Read RGB data
            with rasterio.open(rgb_path) as src:
                rgb = src.read()
                height, width = rgb.shape[1], rgb.shape[2]

            # Read NDVI data
            with rasterio.open(ndvi_path) as src:
                ndvi = src.read(1)

                # Reshape to match RGB dimensions if needed
                if ndvi.shape != (height, width):
                    ndvi = resize(ndvi, (height, width), preserve_range=True)

            # Normalize data
            rgb = rgb / 10000.0  # Typical scaling for reflectance values
            ndvi = (ndvi + 1) / 2.0  # Scale NDVI from [-1,1] to [0,1]

            # Stack features: RGB (3) + NDVI (1)
            features = np.vstack([rgb, ndvi[np.newaxis, :, :]])

            # Extract center patch
            h_start = (height - patch_size) // 2
            w_start = (width - patch_size) // 2

            if h_start < 0 or w_start < 0:
                raise ValueError(f"Image is smaller than patch size {patch_size}")

            feature_patch = features[
                :, h_start : h_start + patch_size, w_start : w_start + patch_size
            ]

            # Reshape for model input: (height, width, channels)
            feature_patch = np.transpose(feature_patch, (1, 2, 0))

            return feature_patch

        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None

    def analyze_temporal_changes(self, historical_images):
        """
        Compare forest cover across multiple time periods

        Args:
            historical_images (list): List of (year, rgb_path, ndvi_path)

        Returns:
            dict: Temporal forest change metrics
        """
        temporal_results = []

        for year, rgb_path, ndvi_path in historical_images:
            result = self.predict_forest_cover(rgb_path, ndvi_path)
            temporal_results.append(
                {
                    "year": year,
                    "forest_percentage": result["forest_percentage"],
                    "is_forest_area": result["is_forest_area"],
                }
            )

        # Calculate change rates
        if len(temporal_results) > 1:
            change_rates = [
                (
                    temporal_results[i + 1]["forest_percentage"]
                    - temporal_results[i]["forest_percentage"]
                )
                / temporal_results[i]["forest_percentage"]
                for i in range(len(temporal_results) - 1)
            ]

            return {
                "temporal_results": temporal_results,
                "forest_change_rates": change_rates,
                "total_forest_loss": sum(rate for rate in change_rates if rate < 0),
                "total_forest_gain": sum(rate for rate in change_rates if rate > 0),
            }

        return {"temporal_results": temporal_results}

    def calculate_advanced_vegetation_indices(self, rgb_path, ndvi_path):
        """
        Calculate additional vegetation and spectral indices

        Args:
            rgb_path (str): Path to RGB image
            ndvi_path (str): Path to NDVI image

        Returns:
            dict: Advanced vegetation metrics
        """
        # Read images
        with rasterio.open(rgb_path) as src_rgb, rasterio.open(ndvi_path) as src_ndvi:
            rgb = src_rgb.read()
            ndvi = src_ndvi.read(1)

        # Calculate additional indices
        # Soil-Adjusted Vegetation Index (SAVI)
        L = 0.5  # Soil brightness correction factor
        savi = ((1 + L) * (rgb[3] - rgb[2])) / (rgb[3] + rgb[2] + L)

        # Enhanced Vegetation Index (EVI)
        G = 2.5  # Gain factor
        C1 = 6.0  # Coefficient 1
        C2 = 7.5  # Coefficient 2
        L = 1.0  # Canopy background adjustment
        evi = G * ((rgb[3] - rgb[2]) / (rgb[3] + C1 * rgb[2] - C2 * rgb[0] + L))

        return {
            "ndvi": np.mean(ndvi),
            "savi": np.mean(savi),
            "evi": np.mean(evi),
            "spectral_diversity": np.std(rgb, axis=(1, 2)),
        }

    def predict_forest_cover(self, rgb_path, ndvi_path, threshold=0.5):
        """
        Predict forest cover for a given satellite image

        Args:
            rgb_path (str): Path to RGB GeoTIFF
            ndvi_path (str): Path to NDVI GeoTIFF
            threshold (float): Threshold for forest cover classification

        Returns:
            dict: Prediction results including forest cover metrics
        """
        # Preprocess image
        features = self.preprocess_image(rgb_path, ndvi_path)

        if features is None:
            return {"success": False, "error": "Image preprocessing failed"}

        # Add batch dimension for prediction
        features = np.expand_dims(features, axis=0)

        # Predict forest cover
        prediction = self.model.predict(features)[0]

        # Calculate forest cover metrics
        forest_mask = prediction > threshold
        forest_percentage = np.mean(forest_mask)

        return {
            "success": True,
            "forest_mask": prediction.squeeze(),
            "forest_percentage": float(forest_percentage),
            "is_forest_area": forest_percentage > threshold,
        }


def analyze_landscape_context(self, rgb_path, ndvi_path, buffer_size=1000):
    """
    Analyze surrounding landscape context

    Args:
        rgb_path (str): Path to RGB image
        ndvi_path (str): Path to NDVI image
        buffer_size (int): Buffer size in meters

    Returns:
        dict: Landscape context metrics
    """
    # Use rasterio to get image metadata
    with rasterio.open(rgb_path) as src:
        # Get image resolution and transform
        resolution = src.res[0]  # Pixel size
        transform = src.transform

    # Calculate buffer in pixels
    buffer_pixels = int(buffer_size / resolution)

    # Load full images
    with rasterio.open(rgb_path) as src_rgb, rasterio.open(ndvi_path) as src_ndvi:
        full_rgb = src_rgb.read()
        full_ndvi = src_ndvi.read(1)

    # Extract buffered region
    h, w = full_ndvi.shape
    center_h, center_w = h // 2, w // 2

    buffered_ndvi = full_ndvi[
        max(0, center_h - buffer_pixels) : min(h, center_h + buffer_pixels),
        max(0, center_w - buffer_pixels) : min(w, center_w + buffer_pixels),
    ]

    return {
        "landscape_ndvi_mean": np.mean(buffered_ndvi),
        "landscape_ndvi_std": np.std(buffered_ndvi),
        "landscape_forest_connectivity": self._calculate_forest_connectivity(
            buffered_ndvi
        ),
    }

    def _calculate_forest_connectivity(self, ndvi_mask, threshold=0.6):
        """
        Calculate forest patch connectivity

        Args:
            ndvi_mask (np.ndarray): NDVI values
            threshold (float): NDVI threshold for forest

        Returns:
            float: Forest connectivity metric
        """
        # Create binary forest mask
        forest_mask = ndvi_mask > threshold

        # Use scipy for connected component analysis
        from scipy import ndimage

        # Label connected components
        labeled_forest, num_features = ndimage.label(forest_mask)

        # Calculate forest patch sizes
        patch_sizes = np.bincount(labeled_forest.ravel())[1:]

        # Connectivity metrics
        return {
            "num_forest_patches": num_features,
            "largest_patch_proportion": (
                max(patch_sizes) / forest_mask.size if num_features > 0 else 0
            ),
            "mean_patch_size": np.mean(patch_sizes) if num_features > 0 else 0,
        }

    def batch_inference(self, data_dir, output_path=None):
        """
        Perform inference on multiple images in a directory

        Args:
            data_dir (str): Directory containing RGB and NDVI images
            output_path (str, optional): Path to save inference results

        Returns:
            pd.DataFrame: Inference results for all images
        """
        # Prepare results storage
        results = []

        # If metadata is provided, use it to match files
        if self.metadata is not None:
            for _, row in self.metadata.iterrows():
                rgb_path = os.path.join(data_dir, row["rgb_file"])
                ndvi_path = os.path.join(data_dir, row["ndvi_file"])

                # Perform inference
                inference_result = self.predict_forest_cover(rgb_path, ndvi_path)

                # Combine with original metadata
                result_row = row.to_dict()
                result_row.update(inference_result)
                results.append(result_row)

        # If no metadata, search for matching RGB and NDVI files
        else:
            # Find all unique base names of RGB and NDVI files
            rgb_files = [f for f in os.listdir(data_dir) if f.endswith("_rgb.tif")]

            for rgb_filename in rgb_files:
                # Try to find matching NDVI file
                base_name = rgb_filename.replace("_rgb.tif", "")
                ndvi_filename = f"{base_name}_ndvi.tif"

                rgb_path = os.path.join(data_dir, rgb_filename)
                ndvi_path = os.path.join(data_dir, ndvi_filename)

                if os.path.exists(ndvi_path):
                    # Perform inference
                    inference_result = self.predict_forest_cover(rgb_path, ndvi_path)

                    # Store results
                    result_row = {
                        "rgb_file": rgb_filename,
                        "ndvi_file": ndvi_filename,
                    }
                    result_row.update(inference_result)
                    results.append(result_row)

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Save results if output path provided
        if output_path:
            results_df.to_csv(output_path, index=False)

        return results_df


def main():
    # Configuration
    model_path = (
        "models/forest_detection/models/forest_detection_final_YYYYMMDD-HHMMSS.h5"
    )
    metadata_path = "metadata.csv"
    data_dir = "path/to/satellite/images"
    historical_data_dir = "path/to/historical/images"
    output_path = "forest_cover_comprehensive_analysis.csv"

    # Initialize inference
    inference = ForestCoverInference(model_path=model_path, metadata_path=metadata_path)

    # Prepare comprehensive results
    comprehensive_results = []

    # Load metadata for historical context
    historical_metadata = pd.read_csv(metadata_path) if metadata_path else None

    # Batch process images
    current_images = [
        (os.path.join(data_dir, rgb_file), os.path.join(data_dir, ndvi_file))
        for rgb_file, ndvi_file in zip(
            [f for f in os.listdir(data_dir) if f.endswith("_rgb.tif")],
            [f for f in os.listdir(data_dir) if f.endswith("_ndvi.tif")],
        )
    ]

    # Prepare historical data if available
    historical_images = []
    if historical_metadata is not None:
        for _, row in historical_metadata.iterrows():
            historical_images.append(
                (
                    row["year"],
                    os.path.join(historical_data_dir, row["rgb_file"]),
                    os.path.join(historical_data_dir, row["ndvi_file"]),
                )
            )

    # Comprehensive analysis for each image
    for rgb_path, ndvi_path in current_images:
        try:
            # Perform comprehensive inference
            comprehensive_result = inference.comprehensive_inference(
                rgb_path, ndvi_path, historical_images
            )

            # Add file paths and other metadata
            comprehensive_result["rgb_file"] = os.path.basename(rgb_path)
            comprehensive_result["ndvi_file"] = os.path.basename(ndvi_path)

            comprehensive_results.append(comprehensive_result)

        except Exception as e:
            print(f"Error processing {rgb_path}: {e}")

    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(comprehensive_results)

    # Save comprehensive results
    results_df.to_csv(output_path, index=False)

    # Generate fraud risk summary
    fraud_risk_summary = {
        "total_images": len(results_df),
        "anomalous_images": results_df[
            results_df["anomaly_detection"].apply(
                lambda x: x["is_anomaly"] if x is not None else False
            )
        ],
        "forest_cover_statistics": {
            "mean_forest_percentage": results_df["forest_cover"]
            .apply(lambda x: x["forest_percentage"])
            .mean(),
            "median_forest_percentage": results_df["forest_cover"]
            .apply(lambda x: x["forest_percentage"])
            .median(),
        },
        "high_risk_indicators": [],
    }

    # Identify high-risk indicators
    for index, row in results_df.iterrows():
        risk_indicators = []

        # Anomaly detection
        if (
            row["anomaly_detection"] is not None
            and row["anomaly_detection"]["is_anomaly"]
        ):
            risk_indicators.append("Anomalous Forest Metrics")

        # Temporal change analysis
        if (
            row["temporal_analysis"] is not None
            and row["temporal_analysis"].get("total_forest_loss", 0) > 0.2
        ):
            risk_indicators.append("Significant Forest Loss")

        # Landscape context
        if (
            row["landscape_context"]
            .get("landscape_forest_connectivity", {})
            .get("num_forest_patches", 0)
            > 5
        ):
            risk_indicators.append("Fragmented Forest Area")

        if risk_indicators:
            fraud_risk_summary["high_risk_indicators"].append(
                {"image": row["rgb_file"], "risks": risk_indicators}
            )

    # Print summary
    print("\n--- Comprehensive Forest Analysis Summary ---")
    print(f"Total Images Processed: {fraud_risk_summary['total_images']}")
    print(
        f"Mean Forest Percentage: {fraud_risk_summary['forest_cover_statistics']['mean_forest_percentage']:.2%}"
    )
    print("\nHigh-Risk Indicators:")
    for risk in fraud_risk_summary["high_risk_indicators"]:
        print(f"Image: {risk['image']}")
        print(f"Risks: {', '.join(risk['risks'])}")

    # Optional: Generate detailed report
    with open("fraud_risk_report.txt", "w") as f:
        json.dump(fraud_risk_summary, f, indent=2)

    print("\nDetailed fraud risk report saved to 'fraud_risk_report.txt'")


if __name__ == "__main__":
    main()
