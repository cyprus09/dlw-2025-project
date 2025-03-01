"""Service for interacting with ML model."""

from typing import Any, Dict, Tuple

from .geocode_service import get_coordinates_from_location_name


class MLService:
    """Service for ML model operations."""

    @staticmethod
    async def process_input(
        location_name: str,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Process input data through ML model.

        This is a placeholder implementation. Replace with actual ML model call.
        """
        print("Processing input data:", location_name)
        lat, lon = get_coordinates_from_location_name(location_name)
        if not lat or not lon:
            raise ValueError("Invalid location coordinates")
        # TODO: Replace with actual ML model API call
        result = {
            "co2_sequestration": 100,
            "time_series_land_change": 0.5,
            "forest_area_detection": 0.8,
        }
        location = {"lat": lat, "lon": lon}
        return result, location
