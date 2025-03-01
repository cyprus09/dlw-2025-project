import numpy as np

from .ecological_zone_api import get_ecological_zone


def get_ndvi_data(lat, lon):
    """
    Get NDVI data for a location.

    In a full implementation, this would connect to a satellite imagery API
    like Google Earth Engine, Sentinel Hub, or Planet API.

    For this demo, we'll generate synthetic NDVI data based on the ecological zone.

    Args:
        lat (float): Latitude
        lon (float): Longitude

    Returns:
        array: NDVI data (synthetic for demonstration)
    """
    eco_zone = get_ecological_zone(lat, lon)

    eco_zone_ndvi_means = {
        "tropical_rainforest": 0.7,
        "tropical_moist_forest": 0.65,
        "tropical_dry_forest": 0.5,
        "temperate_broadleaf": 0.6,
        "temperate_coniferous": 0.55,
        "temperate_mixed": 0.58,
        "boreal_forest": 0.5,
        "subtropical_forest": 0.6,
        "mangrove": 0.6,
        "woodland_savanna": 0.4,
        "mediterranean": 0.45,
        "montane": 0.5,
        "alpine": 0.35,
    }

    ndvi_mean = eco_zone_ndvi_means.get(eco_zone, 0.5)

    ndvi_data = np.random.normal(ndvi_mean, 0.1, 100)

    ndvi_data = np.clip(ndvi_data, -1, 1)

    return ndvi_data


print(get_ndvi_data(37.7749, -122.4194))  # San Francisco
print(get_ndvi_data(-22.9068, -43.1729))  # Rio de Janeiro
