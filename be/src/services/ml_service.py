"""Service for interacting with ML model."""

import asyncio
import os
from typing import Any, Dict, Tuple


from .geocode_service import get_coordinates_from_location_name
from models.carbon_sequestration.inference import get_co2_sequestration
from models.carbon_sequestration.tree_growth_rate.calculate import TreeGrowthPredictor 
from models.carbon_sequestration.carbon_content.services.calculate import get_carbon_content_from_coordinates

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
        Forest Area = (500^2) * (forest_percentage / 100)
        """
        
        if PREDICTOR.metadata is None or PREDICTOR.metadata.empty:
            raise ValueError("Metadata is not loaded. Ensure correct CSV paths.")

        PREDICTOR.metadata["forest_area"] = (500 ** 2) * (PREDICTOR.metadata["forest_percentage"] / 100)

        return PREDICTOR.metadata["forest_area"].tolist()
    


    @staticmethod
    async def process_input(
        location_name: str,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Process input data through ML model.
        """
        coordinates = get_coordinates_from_location_name(location_name)
        if not coordinates:
            raise ValueError("Invalid location coordinates")
        lat, lon = coordinates
        # print(f"Coordinates: {lat}, {lon}")

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