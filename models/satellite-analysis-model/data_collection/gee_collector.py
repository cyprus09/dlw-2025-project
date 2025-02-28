import ee
import pandas as pd
import os
import time
import certifi
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

ee.Initialize(project="PROJECT-ID")


# Create a training dataset from GEE satellite imagery.
def create_training_dataset(coordinates_csv, output_dir, years_range=[2018, 2023]):

    coords_df = pd.read_csv(coordinates_csv)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Metadata for the dataset
    metadata = []

    for idx, row in coords_df.iterrows():
        point_id = row["id"]
        lon = row["longitude"]
        lat = row["latitude"]
        known_forest = row["known_forest"]  # Ground truth label if available

        # Create a point geometry
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(1000)  # 1km buffer

        # Collect imagery for each year in the range
        for year in range(years_range[0], years_range[1] + 1):
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"

            # Get Sentinel-2 imagery with less than 20% cloud cover
            s2_collection = (
                ee.ImageCollection("COPERNICUS/S2_SR")
                .filterBounds(region)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            )

            if s2_collection.size().getInfo() > 0:

                # Get median composite (reduces clouds and noise)
                s2_median = s2_collection.median()

                # Select RGB bands for visual imagery and rescale
                rgb_image = s2_median.select(["B4", "B3", "B2"]).divide(10000)

                # Also get NDVI (vegetation index) for training features
                ndvi = s2_median.normalizedDifference(["B8", "B4"]).rename("NDVI")

                # Get land cover data from Dynamic World dataset
                dw_collection = (
                    ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                    .filterBounds(region)
                    .filterDate(start_date, end_date)
                )

                # Get the most common land cover class
                if dw_collection.size().getInfo() > 0:
                    landcover = dw_collection.select("label").mode()

                    # Download RGB image
                    task = ee.batch.Export.image.toDrive(
                        image=rgb_image.clip(region),
                        description=f"rgb_{point_id}_{year}",
                        folder="carbon_verification_data",
                        scale=10,  # 10 meter resolution
                        region=region,
                    )
                    print(f"Starting export: rgb_{point_id}_{year}")
                    task.start()
                    print(f"Export for {point_id}_{year} completed.")

                    # Download NDVI
                    task_ndvi = ee.batch.Export.image.toDrive(
                        image=ndvi.clip(region),
                        description=f"ndvi_{point_id}_{year}",
                        folder="carbon_verification_data",
                        scale=10,
                        region=region,
                    )
                    print(f"Starting export: ndvi_{point_id}_{year}")
                    task_ndvi.start()
                    print(f"Export for {point_id}_{year} completed.")

                    # Download land cover
                    task_lc = ee.batch.Export.image.toDrive(
                        image=landcover.clip(region),
                        description=f"landcover_{point_id}_{year}",
                        folder="carbon_verification_data",
                        scale=10,
                        region=region,
                    )
                    print(f"Starting export: landcover_{point_id}_{year}")
                    task_lc.start()
                    print(f"Export for {point_id}_{year} completed.")

                    # Add to metadata
                    metadata.append(
                        {
                            "id": point_id,
                            "longitude": lon,
                            "latitude": lat,
                            "year": year,
                            "rgb_file": f"rgb_{point_id}_{year}.tif",
                            "ndvi_file": f"ndvi_{point_id}_{year}.tif",
                            "landcover_file": f"landcover_{point_id}_{year}.tif",
                            "known_forest": known_forest,
                        }
                    )

                    # Sleep to avoid rate limiting
                    time.sleep(3)

    # Save metadata
    pd.DataFrame(metadata).to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    print(f"Dataset creation tasks submitted to Earth Engine. Check your Google Drive.")
