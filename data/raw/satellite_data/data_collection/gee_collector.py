import ee
import pandas as pd
import os
import time
import datetime
import random
import math
import logging
from concurrent.futures import ThreadPoolExecutor
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("gee_collector.log"), logging.StreamHandler()],
)

# Constants
PROJECT_ID = "dlw-hackathon-452319"
DEFAULT_BUFFER = 500  # meters
MAX_PIXELS = 1e9
BATCH_SIZE = 10  # Process coordinates in batches to avoid overloading
MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds
RATE_LIMIT_SLEEP = 5  # seconds between task starts
BATCH_PAUSE = 60  # seconds between batches


def initialize_ee():
    """Initialize Earth Engine with appropriate error handling."""
    try:
        ee.Initialize(project=PROJECT_ID)
        logging.info("Earth Engine initialized successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize Earth Engine: {e}")
        return False


def create_folder_if_not_exists(folder_path):
    """Create a folder if it doesn't exist."""
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            logging.info(f"Created directory: {folder_path}")
    except Exception as e:
        logging.error(f"Error creating directory {folder_path}: {e}")


def run_test_export():
    """Run a small test export to verify permissions and connection."""
    # Singapore coordinates for test
    test_point = ee.Geometry.Point([103.8198, 1.3521])
    test_region = test_point.buffer(100)

    try:
        test_collection = (
            ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(test_region)
            .filterDate("2022-01-01", "2022-12-31")
            .select(["B4", "B3", "B2"])
            .first()
        )

        if test_collection:
            test_task = ee.batch.Export.image.toDrive(
                image=test_collection.select(["B4", "B3", "B2"]).divide(10000),
                description=f"test_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                folder="carbon_verification_data",
                scale=30,
                region=test_region,
                maxPixels=MAX_PIXELS,
            )
            test_task.start()
            logging.info("Test task started - checking if GEE exports are working")
            time.sleep(5)

            # Check task status
            status = test_task.status()
            logging.info(f"Test task status: {status['state']}")
            return True
        else:
            logging.warning(
                "Test collection empty, might indicate data availability issues"
            )
            return False
    except Exception as e:
        logging.error(f"Test task failed: {e}")
        return False


def with_retry(func, *args, max_retries=MAX_RETRIES, **kwargs):
    """Run a function with retry logic."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                delay = RETRY_DELAY * (2**attempt)  # Exponential backoff
                logging.warning(
                    f"Attempt {attempt+1} failed: {e}. Retrying in {delay}s..."
                )
                time.sleep(delay)
            else:
                logging.error(f"Function failed after {max_retries} attempts: {e}")
                raise


def get_sentinel_collection(region, year, select_bands=None):
    """Get Sentinel-2 collection with explicit band selection to avoid heterogeneity errors."""
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    # Default bands if none specified
    if select_bands is None:
        select_bands = ["B2", "B3", "B4", "B8"]  # RGB + NIR for NDVI

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )

    # Count before band selection
    count = collection.size().getInfo()

    if count > 0:
        # Explicitly select common bands to avoid heterogeneity errors
        collection = collection.select(select_bands)
        return collection, count
    else:
        return None, 0


def get_dynamic_world_collection(region, year):
    """Get Dynamic World land cover collection."""
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    collection = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(region)
        .filterDate(start_date, end_date)
    )

    count = collection.size().getInfo()
    return collection, count


def export_image(image, description, folder, scale, region):
    """Export an image to Google Drive with proper error handling."""
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        scale=scale,
        region=region,
        maxPixels=MAX_PIXELS,
    )

    try:
        task.start()
        logging.info(f"Started export: {description}")
        return task
    except Exception as e:
        logging.error(f"Failed to start task {description}: {e}")
        return None


def process_coordinate(coords, years_range, output_dir):
    """Process a single coordinate across multiple years."""
    point_id = coords["id"]
    lon = coords["longitude"]
    lat = coords["latitude"]
    known_forest = coords.get("known_forest", None)

    logging.info(f"Processing point {point_id} ({lon}, {lat})")

    # Create a point geometry with buffer
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(DEFAULT_BUFFER)

    metadata_rows = []
    tasks = []

    for year in range(years_range[0], years_range[1] + 1):
        try:
            # Get Sentinel-2 imagery
            s2_collection, s2_count = get_sentinel_collection(region, year)

            if not s2_collection or s2_count == 0:
                logging.warning(
                    f"No Sentinel-2 data found for point {point_id} in {year}"
                )
                continue

            # Get median composite
            s2_median = s2_collection.median()

            # Get RGB image
            rgb_image = s2_median.select(["B4", "B3", "B2"]).divide(10000)

            # Get NDVI
            if "B8" in s2_median.bandNames().getInfo():
                ndvi = s2_median.normalizedDifference(["B8", "B4"]).rename("NDVI")
            else:
                logging.warning(
                    f"B8 band not available for point {point_id} in {year}, skipping NDVI"
                )
                ndvi = None

            # Get Dynamic World land cover
            dw_collection, dw_count = get_dynamic_world_collection(region, year)

            # Start exports with appropriate delays
            task_rgb = export_image(
                rgb_image.clip(region),
                f"rgb_{point_id}_{year}",
                "carbon_verification_data",
                10,
                region,
            )

            if task_rgb:
                tasks.append(task_rgb)
                time.sleep(RATE_LIMIT_SLEEP)  # Avoid rate limiting

            if ndvi:
                task_ndvi = export_image(
                    ndvi.clip(region),
                    f"ndvi_{point_id}_{year}",
                    "carbon_verification_data",
                    10,
                    region,
                )

                if task_ndvi:
                    tasks.append(task_ndvi)
                    time.sleep(RATE_LIMIT_SLEEP)

            if dw_count > 0:
                landcover = dw_collection.select("label").mode()

                task_lc = export_image(
                    landcover.clip(region),
                    f"landcover_{point_id}_{year}",
                    "carbon_verification_data",
                    10,
                    region,
                )

                if task_lc:
                    tasks.append(task_lc)
                    time.sleep(RATE_LIMIT_SLEEP)
            else:
                logging.warning(f"No Dynamic World data for point {point_id} in {year}")

            # Add to metadata
            metadata_rows.append(
                {
                    "id": point_id,
                    "longitude": lon,
                    "latitude": lat,
                    "year": year,
                    "rgb_file": f"rgb_{point_id}_{year}.tif",
                    "ndvi_file": f"ndvi_{point_id}_{year}.tif" if ndvi else None,
                    "landcover_file": (
                        f"landcover_{point_id}_{year}.tif" if dw_count > 0 else None
                    ),
                    "known_forest": known_forest,
                    "s2_image_count": s2_count,
                    "dw_image_count": dw_count,
                }
            )

        except Exception as e:
            logging.error(f"Error processing point {point_id} for year {year}: {e}")
            continue

    return metadata_rows, tasks


def process_batch(batch_coords, years_range, output_dir):
    """Process a batch of coordinates."""
    all_metadata = []
    all_tasks = []

    for coords in batch_coords:
        metadata_rows, tasks = process_coordinate(coords, years_range, output_dir)
        all_metadata.extend(metadata_rows)
        all_tasks.extend(tasks)

        # Small sleep between coordinates within a batch
        time.sleep(RATE_LIMIT_SLEEP)

    return all_metadata, all_tasks


def save_metadata(metadata, output_dir, batch_num=None):
    """Save metadata to CSV."""
    if not metadata:
        logging.warning("No metadata to save")
        return

    suffix = f"_batch{batch_num}" if batch_num is not None else ""
    metadata_path = os.path.join(output_dir, f"metadata{suffix}.csv")

    try:
        pd.DataFrame(metadata).to_csv(metadata_path, index=False)
        logging.info(f"Metadata saved to {metadata_path}")
    except Exception as e:
        logging.error(f"Error saving metadata: {e}")


def create_training_dataset(
    coordinates_csv, output_dir, years_range=[2018, 2023], batch_size=BATCH_SIZE
):
    """
    Create a training dataset from GEE satellite imagery with improved robustness.

    Args:
        coordinates_csv (str): Path to CSV with coordinate points
        output_dir (str): Directory to save metadata
        years_range (list): Start and end years [start, end] inclusive
        batch_size (int): Number of coordinates to process in each batch
    """
    if not initialize_ee():
        return

    # Create output directory
    create_folder_if_not_exists(output_dir)

    # Run test export to verify permissions
    if not run_test_export():
        logging.warning("Test export failed, but continuing with main tasks")

    try:
        # Load coordinates
        coords_df = pd.read_csv(coordinates_csv)
        total_points = len(coords_df)
        logging.info(f"Loaded {total_points} points from coordinates CSV")

        if total_points == 0:
            logging.error("No coordinates found in CSV")
            return

        # Convert DataFrame to list of dictionaries for easier processing
        coords_list = coords_df.to_dict("records")

        # Process in batches to avoid overwhelming GEE
        all_metadata = []
        batch_count = math.ceil(total_points / batch_size)

        for i in range(batch_count):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_points)
            batch = coords_list[start_idx:end_idx]

            logging.info(
                f"Processing batch {i+1}/{batch_count} ({len(batch)} coordinates)"
            )

            batch_metadata, batch_tasks = process_batch(batch, years_range, output_dir)
            all_metadata.extend(batch_metadata)

            # Save intermediate metadata for each batch
            save_metadata(batch_metadata, output_dir, i + 1)

            # Check status of a few tasks from this batch
            if batch_tasks:
                sample_size = min(3, len(batch_tasks))
                for j in range(sample_size):
                    try:
                        task = batch_tasks[j]
                        status = task.status()
                        logging.info(f"Sample task {j+1} status: {status['state']}")
                    except Exception as e:
                        logging.error(f"Could not check task status: {e}")

            # Pause between batches to avoid rate limiting
            if i < batch_count - 1:  # Don't pause after the last batch
                pause = BATCH_PAUSE + random.randint(0, 30)  # Add some randomness
                logging.info(f"Pausing for {pause}s before next batch")
                time.sleep(pause)

        # Save final complete metadata
        save_metadata(all_metadata, output_dir)

        logging.info(
            f"Dataset creation complete. Total metadata entries: {len(all_metadata)}"
        )
        logging.info(
            "Check your Google Drive folder 'carbon_verification_data' for exported images"
        )

    except Exception as e:
        logging.error(f"Error in create_training_dataset: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Google Earth Engine Data Collector")
    parser.add_argument(
        "--coordinates",
        type=str,
        default="coordinates.csv",
        help="Path to coordinates CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data_collection",
        help="Output directory for metadata",
    )
    parser.add_argument(
        "--start-year", type=int, default=2018, help="Start year for data collection"
    )
    parser.add_argument(
        "--end-year", type=int, default=2023, help="End year for data collection"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Number of coordinates to process in each batch",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.info(f"Starting GEE data collection with parameters:")
    logging.info(f"  Coordinates file: {args.coordinates}")
    logging.info(f"  Output directory: {args.output}")
    logging.info(f"  Year range: {args.start_year}-{args.end_year}")
    logging.info(f"  Batch size: {args.batch_size}")

    create_training_dataset(
        coordinates_csv=args.coordinates,
        output_dir=args.output,
        years_range=[args.start_year, args.end_year],
        batch_size=args.batch_size,
    )
