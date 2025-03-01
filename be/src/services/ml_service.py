# """Service for interacting with ML model."""

# import asyncio
# import os
# from typing import Any, Dict, Tuple


# from .geocode_service import get_coordinates_from_location_name
# from models.carbon_sequestration.inference import get_co2_sequestration
# from models.carbon_sequestration.tree_growth_rate.calculate import TreeGrowthPredictor 
# from models.carbon_sequestration.carbon_content.services.calculate import get_carbon_content_from_coordinates

# current_file = os.path.abspath(__file__)
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(current_file), "../../.."))

# PREDICTOR = TreeGrowthPredictor(
#     X_features_file=os.path.join(PROJECT_ROOT, "data/processed/image_datasets/processed_data/image_datasets/processed_data/processed_X_features.csv"),
#     y_masks_file=os.path.join(PROJECT_ROOT, "data/processed/image_datasets/processed_data/image_datasets/processed_data/processed_y_masks.csv"),
#     metadata_file=os.path.join(PROJECT_ROOT, "data/processed/image_datasets/processed_data/image_datasets/processed_data/processed_metadata.csv")
# )

# class MLService:
#     """Service for ML model operations."""

#     @staticmethod
#     async def get_growth_rate():
#         """
#         Compute tree growth rate using TreeGrowthPredictor.
#         This does NOT depend on lat/lon.
#         """


#         await asyncio.to_thread(PREDICTOR.match_locations_by_coordinates)
    

#         growth_metrics = await asyncio.to_thread(PREDICTOR.model_growth_rate_fixed_weights)

#         if growth_metrics.empty:
#             raise ValueError("Growth rate calculation returned empty results.")

#         # Extract the tree growth rate column
#         return growth_metrics["tree_growth_rate"].tolist()
    
#     @staticmethod
#     async def get_forest_area():
#         """
#         Compute forest area using the formula:
#         Forest Area = (500^2) * (forest_percentage / 100)
#         """
        
#         if PREDICTOR.metadata is None or PREDICTOR.metadata.empty:
#             raise ValueError("Metadata is not loaded. Ensure correct CSV paths.")

#         PREDICTOR.metadata["forest_area"] = (500 ** 2) * (PREDICTOR.metadata["forest_percentage"]) * 0.0001

#         return PREDICTOR.metadata["forest_area"].tolist()
    


#     @staticmethod
#     async def process_input(
#         location_name: str,
#     ) -> Tuple[Dict[str, float], Dict[str, Any]]:
#         """
#         Process input data through ML model.
#         """
#         coordinates = get_coordinates_from_location_name(location_name)
#         if not coordinates:
#             raise ValueError("Invalid location coordinates")
#         lat, lon = coordinates
#         # print(f"Coordinates: {lat}, {lon}")

#         carbon_content = await asyncio.to_thread(get_carbon_content_from_coordinates, lat, lon)
        
#         if isinstance(carbon_content, dict):
#             if 'value' in carbon_content:
#                 carbon_content = carbon_content['value']
#             elif 'content' in carbon_content:
#                 carbon_content = carbon_content['content']
#             elif 'carbon_content' in carbon_content:
#                 carbon_content = carbon_content['carbon_content']
#             else:
#                 carbon_content = next(iter(carbon_content.values()))

#         growth_rate = await MLService.get_growth_rate()
        
#         forest_area = await MLService.get_forest_area()

#         result = get_co2_sequestration(growth_rate, carbon_content, forest_area)
        
#         location = {"lat": lat, "lon": lon}
#         return result, location

"""Service for interacting with ML model."""

import asyncio
import os
import subprocess
import json
from typing import Any, Dict, Tuple

from .geocode_service import get_coordinates_from_location_name
from models.carbon_sequestration.inference import get_co2_sequestration
from models.carbon_sequestration.tree_growth_rate.calculate import TreeGrowthPredictor 
from models.carbon_sequestration.carbon_content.services.calculate import get_carbon_content_from_coordinates
from models.satellite_analysis_model.change_detection.load_result  import load_satellite_result

current_file = os.path.abspath(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(current_file), "../../.."))

PREDICTOR = TreeGrowthPredictor(
    X_features_file=os.path.join(PROJECT_ROOT, "data/processed/image_datasets/processed_data/image_datasets/processed_data/processed_X_features.csv"),
    y_masks_file=os.path.join(PROJECT_ROOT, "data/processed/image_datasets/processed_data/image_datasets/processed_data/processed_y_masks.csv"),
    metadata_file=os.path.join(PROJECT_ROOT, "data/processed/image_datasets/processed_data/image_datasets/processed_data/processed_metadata.csv")
)

class MLService:
    """Service for ML model operations."""

    @staticmethod
    async def get_growth_rate():
        """
        Compute tree growth rate using TreeGrowthPredictor.
        This does NOT depend on lat/lon.
        """
        await asyncio.to_thread(PREDICTOR.match_locations_by_coordinates)
        growth_metrics = await asyncio.to_thread(PREDICTOR.model_growth_rate_fixed_weights)

        if growth_metrics.empty:
            raise ValueError("Growth rate calculation returned empty results.")

        # Extract the tree growth rate column
        return growth_metrics["tree_growth_rate"].tolist()
    
    @staticmethod
    async def get_forest_area():
        """
        Compute forest area using the formula:
        Forest Area = (500^2) * (forest_percentage) * 0.0001
        """
        if PREDICTOR.metadata is None or PREDICTOR.metadata.empty:
            raise ValueError("Metadata is not loaded. Ensure correct CSV paths.")

        PREDICTOR.metadata["forest_area"] = (500 ** 2) * (PREDICTOR.metadata["forest_percentage"]) * 0.0001

        return PREDICTOR.metadata["forest_area"].tolist()
    
    @staticmethod
    async def run_satellite_change_detection(
        latitude: float,
        longitude: float,
        year1: int = 2019,
        year2: int = 2022,
    ) -> Dict[str, Any]:
        """
        Run satellite imagery change detection analysis.
        """
        # Define paths
        script_path = os.path.join(
            PROJECT_ROOT, 
            "models/satellite-analysis-model/change_detection/earth_engine_metrics.py"
        )
        forest_model_path = os.path.join(
            PROJECT_ROOT,
            "models/satellite-analysis-model/forest_detection/models/forest_detection/trained_models/forest_detection_final_20250301-165353.h5"
        )
        output_dir = os.path.join(
            PROJECT_ROOT,
            "models/satellite-analysis-model/change_detection/results"
        )
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct command
        cmd = [
            "python3", script_path,
            "--forest_model", forest_model_path,
            "--latitude", str(latitude),
            "--longitude", str(longitude),
            "--year1", str(year1),
            "--year2", str(year2),
            "--output_dir", output_dir
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
    
        
        return load_satellite_result()
    
    @staticmethod
    async def process_input(
        location_name: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process input data through ML model.
        """
        coordinates = get_coordinates_from_location_name(location_name)
        if not coordinates:
            raise ValueError("Invalid location coordinates")
        lat, lon = coordinates

        carbon_content = await asyncio.to_thread(get_carbon_content_from_coordinates, lat, lon)
        
        if isinstance(carbon_content, dict):
            if 'value' in carbon_content:
                carbon_content = carbon_content['value']
            elif 'content' in carbon_content:
                carbon_content = carbon_content['content']
            elif 'carbon_content' in carbon_content:
                carbon_content = carbon_content['carbon_content']
            else:
                carbon_content = next(iter(carbon_content.values()))

        growth_rate = await MLService.get_growth_rate()
        forest_area = await MLService.get_forest_area()

        result = get_co2_sequestration(growth_rate, carbon_content, forest_area)
        
        location = {"lat": lat, "lon": lon}
        return result, location
    
    @staticmethod
    async def analyze_location(
        location_name: str,
        year1: int = 2019,
        year2: int = 2022
    ) -> Dict[str, Any]:
        """
        Complete analysis of a location, combining carbon sequestration and satellite data.
        """
        coordinates = get_coordinates_from_location_name(location_name)
        if not coordinates:
            raise ValueError("Invalid location coordinates")
        lat, lon = coordinates
        
        sequestration_data, location = await MLService.process_input(location_name)
        
        try:
    # Add more detailed error logging
          import traceback
          import logging
        
          logging.basicConfig(level=logging.DEBUG)
        
          satellite_data = await MLService.run_satellite_change_detection(
            latitude=lat,
            longitude=lon,
            year1=year1,
            year2=year2
        )
        except Exception as e:
            error_details = traceback.format_exc()
            logging.error(f"Full error traceback: {error_details}")
            
            satellite_data = load_satellite_result()
        
        # Combine results without additional integrated analysis
        return {
            "carbon_sequestration": sequestration_data,
            "satellite_analysis": satellite_data,
            "location": location
        }