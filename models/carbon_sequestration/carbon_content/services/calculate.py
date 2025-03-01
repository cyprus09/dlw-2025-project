import numpy as np

from .ecological_zone_api import get_ecological_zone
from .nasa_power_api import get_nasa_power_climate_data
from .ndvi_api import get_ndvi_data
from .open_meteo_elevation_api import get_elevation_data


def get_carbon_content_from_coordinates(lat, lon):
    _, climate_features = get_nasa_power_climate_data(lat, lon)
    ndvi_data = get_ndvi_data(lat, lon)
    elevation = get_elevation_data(lat, lon)
    eco_zone = get_ecological_zone(lat, lon)

    return calculate_carbon_content(climate_features, ndvi_data, elevation, eco_zone)


def calculate_carbon_content(climate_features, ndvi_data, elevation, eco_zone):
    """
    Calculate carbon content based on climate features, NDVI, elevation
    and ecological zone using a purely formula-based approach (no ML models).

    Args:
        climate_features (dict): Dictionary containing climate variables
        ndvi_data (array): NDVI data for the location
        elevation (float): Elevation in meters
        eco_zone (str): Ecological zone classification

    Returns:
        dict: Carbon content results with detailed components
    """
    base_carbon = calculate_eco_zone_potential(eco_zone)
    climate_factor = calculate_climate_factor(climate_features)
    ndvi_mean = np.mean(ndvi_data)
    vegetation_factor = calculate_vegetation_factor(ndvi_mean, eco_zone)
    elevation_factor = calculate_elevation_factor(elevation, eco_zone)
    carbon_content = base_carbon * climate_factor * vegetation_factor * elevation_factor
    uncertainty = estimate_uncertainty(climate_features, ndvi_mean, eco_zone)

    result = {
        "carbon_content": carbon_content,
        "uncertainty": uncertainty,
        "components": {
            "base_carbon_potential": base_carbon,
            "climate_factor": climate_factor,
            "vegetation_factor": vegetation_factor,
            "elevation_factor": elevation_factor,
        },
        "breakdown": {
            "above_ground_carbon": carbon_content * 0.65,
            "below_ground_carbon": carbon_content * 0.35,
        },
    }

    return result


def calculate_eco_zone_potential(eco_zone):
    """
    Calculate baseline carbon storage potential based on ecological zone.

    Args:
        eco_zone (str): Ecological zone classification

    Returns:
        float: Base carbon potential in tC/ha
    """
    eco_zone_potential = {
        "tropical_rainforest": 250,
        "tropical_moist_forest": 180,
        "tropical_dry_forest": 120,
        "temperate_broadleaf": 150,
        "temperate_coniferous": 120,
        "temperate_mixed": 135,
        "boreal_forest": 90,
        "subtropical_forest": 160,
        "mangrove": 280,
        "woodland_savanna": 60,
        "mediterranean": 100,
        "montane": 110,
        "alpine": 50,
    }

    return eco_zone_potential.get(eco_zone, 120)


def calculate_climate_factor(climate_features):
    """
    Calculate climate suitability factor for carbon storage.
    Combines the effects of temperature, precipitation, and other climate variables.

    Args:
        climate_features (dict): Climate features

    Returns:
        float: Climate adjustment factor
    """
    annual_precip = climate_features.get("annual_precip", 1000)
    mean_temp = climate_features.get("mean_temp", 15)
    min_temp = climate_features.get("min_temp", 5)
    max_temp = climate_features.get("max_temp", 25)
    precip_seasonality = climate_features.get("precip_seasonality", 50)
    aridity_index = climate_features.get("aridity_index", 50)
    gdd = climate_features.get("gdd", 2000)
    solar_radiation = climate_features.get("solar_radiation", 200)

    if mean_temp < 0:
        temp_factor = 0.7
    elif 0 <= mean_temp < 10:
        temp_factor = 0.8 + (0.02 * mean_temp)
    elif 10 <= mean_temp < 25:
        temp_factor = 1.0
    else:
        temp_factor = 1.2 - (0.01 * mean_temp)

    if min_temp < -15:
        temp_factor *= 0.8
    elif min_temp < -5:
        temp_factor *= 0.9

    if max_temp > 35:
        temp_factor *= 0.85
    elif max_temp > 30:
        temp_factor *= 0.95

    if annual_precip < 250:
        precip_factor = 0.5
    elif 250 <= annual_precip < 500:
        precip_factor = 0.7
    elif 500 <= annual_precip < 1000:
        precip_factor = 0.9
    elif 1000 <= annual_precip < 2000:
        precip_factor = 1.0
    elif 2000 <= annual_precip < 4000:
        precip_factor = 0.95
    else:
        precip_factor = 0.9

    if precip_seasonality < 30:
        season_factor = 1.05
    elif 30 <= precip_seasonality < 100:
        season_factor = 1.0
    else:
        season_factor = 0.85

    if gdd < 1000:
        gdd_factor = 0.8
    elif 1000 <= gdd < 3000:
        gdd_factor = 1.0
    else:
        gdd_factor = 1.05

    if aridity_index < 20:
        aridity_factor = 0.7
    elif 20 <= aridity_index < 50:
        aridity_factor = 0.9
    else:
        aridity_factor = 1.0

    if solar_radiation < 150:
        solar_factor = 0.9
    elif 150 <= solar_radiation < 300:
        solar_factor = 1.0
    else:
        solar_factor = 0.95

    climate_factor = (
        temp_factor * 0.25
        + precip_factor * 0.25
        + season_factor * 0.15
        + gdd_factor * 0.15
        + aridity_factor * 0.1
        + solar_factor * 0.1
    )

    climate_factor = max(0.5, min(1.3, climate_factor))

    return climate_factor


def calculate_vegetation_factor(ndvi_mean, eco_zone):
    """
    Calculate vegetation density factor based on NDVI.
    Adjusts for how close the vegetation is to its potential in that eco-zone.

    Args:
        ndvi_mean (float): Mean NDVI value for the area
        eco_zone (str): Ecological zone type

    Returns:
        float: Vegetation density factor
    """
    eco_zone_ndvi_thresholds = {
        "tropical_rainforest": 0.7,
        "tropical_moist_forest": 0.65,
        "tropical_dry_forest": 0.55,
        "temperate_broadleaf": 0.65,
        "temperate_coniferous": 0.6,
        "temperate_mixed": 0.62,
        "boreal_forest": 0.55,
        "subtropical_forest": 0.65,
        "mangrove": 0.6,
        "woodland_savanna": 0.45,
        "mediterranean": 0.5,
        "montane": 0.55,
        "alpine": 0.4,
    }

    ndvi_threshold = eco_zone_ndvi_thresholds.get(eco_zone, 0.6)

    if ndvi_mean <= 0:
        return 0.1
    elif ndvi_mean >= ndvi_threshold:
        return 1.0
    else:
        normalized_ndvi = ndvi_mean / ndvi_threshold
        vegetation_factor = 1 / (1 + np.exp(-10 * (normalized_ndvi - 0.5)))
        return max(0.1, vegetation_factor)


def calculate_elevation_factor(elevation, eco_zone):
    """
    Calculate elevation adjustment factor.
    Different forest types have different optimal elevation ranges.

    Args:
        elevation (float): Elevation in meters
        eco_zone (str): Ecological zone type

    Returns:
        float: Elevation adjustment factor
    """
    eco_zone_elevation_ranges = {
        "tropical_rainforest": {
            "min": 0,
            "optimal_min": 0,
            "optimal_max": 1000,
            "max": 1500,
        },
        "tropical_moist_forest": {
            "min": 0,
            "optimal_min": 0,
            "optimal_max": 1500,
            "max": 2000,
        },
        "tropical_dry_forest": {
            "min": 0,
            "optimal_min": 0,
            "optimal_max": 1200,
            "max": 2000,
        },
        "temperate_broadleaf": {
            "min": 0,
            "optimal_min": 200,
            "optimal_max": 1800,
            "max": 2500,
        },
        "temperate_coniferous": {
            "min": 0,
            "optimal_min": 500,
            "optimal_max": 2500,
            "max": 3500,
        },
        "temperate_mixed": {
            "min": 0,
            "optimal_min": 300,
            "optimal_max": 2000,
            "max": 2800,
        },
        "boreal_forest": {"min": 0, "optimal_min": 0, "optimal_max": 1200, "max": 1800},
        "subtropical_forest": {
            "min": 0,
            "optimal_min": 100,
            "optimal_max": 2000,
            "max": 2500,
        },
        "mangrove": {"min": 0, "optimal_min": 0, "optimal_max": 10, "max": 20},
        "woodland_savanna": {
            "min": 0,
            "optimal_min": 0,
            "optimal_max": 1500,
            "max": 2000,
        },
        "mediterranean": {"min": 0, "optimal_min": 0, "optimal_max": 1200, "max": 2000},
        "montane": {"min": 800, "optimal_min": 1000, "optimal_max": 3000, "max": 3500},
        "alpine": {"min": 1500, "optimal_min": 1800, "optimal_max": 2800, "max": 3500},
    }

    default_range = {"min": 0, "optimal_min": 0, "optimal_max": 2000, "max": 3000}
    elevation_range = eco_zone_elevation_ranges.get(eco_zone, default_range)

    if elevation < elevation_range["min"]:
        return 0.7
    elif elevation < elevation_range["optimal_min"]:
        range_fraction = (elevation - elevation_range["min"]) / (
            elevation_range["optimal_min"] - elevation_range["min"] + 1e-8
        )
        return 0.7 + (0.3 * range_fraction)
    elif elevation <= elevation_range["optimal_max"]:
        return 1.0
    elif elevation <= elevation_range["max"]:
        range_fraction = (elevation - elevation_range["optimal_max"]) / (
            elevation_range["max"] - elevation_range["optimal_max"] + 1e-8
        )
        return 1.0 - (0.3 * range_fraction)
    else:
        return 0.7


def estimate_uncertainty(climate_features, ndvi_mean, eco_zone):
    """
    Estimate uncertainty in carbon content calculation.

    Args:
        climate_features (dict): Climate features
        ndvi_mean (float): Mean NDVI value
        eco_zone (str): Ecological zone

    Returns:
        float: Uncertainty estimate as percentage
    """
    base_uncertainty = 20.0

    if ndvi_mean < 0.2:
        ndvi_uncertainty = 10.0
    elif 0.2 <= ndvi_mean < 0.4:
        ndvi_uncertainty = 5.0
    else:
        ndvi_uncertainty = 2.0

    climate_reliability = 0.8

    mean_temp = climate_features.get("mean_temp", 15)
    annual_precip = climate_features.get("annual_precip", 1000)

    if mean_temp < -10 or mean_temp > 30:
        climate_reliability -= 0.1

    if annual_precip < 200 or annual_precip > 5000:
        climate_reliability -= 0.1

    climate_uncertainty = 10.0 * (1.0 - climate_reliability)

    eco_zone_uncertainty = {
        "tropical_rainforest": 5.0,
        "tropical_moist_forest": 7.0,
        "temperate_broadleaf": 5.0,
        "temperate_coniferous": 6.0,
        "boreal_forest": 6.0,
        "mangrove": 10.0,
        "woodland_savanna": 12.0,
        "mediterranean": 8.0,
        "montane": 10.0,
        "alpine": 12.0,
    }

    zone_uncertainty = eco_zone_uncertainty.get(eco_zone, 8.0)

    total_uncertainty = (
        base_uncertainty + ndvi_uncertainty + climate_uncertainty + zone_uncertainty
    )

    return min(50.0, max(15.0, total_uncertainty))


print(get_carbon_content_from_coordinates(37.7749, -122.4194))  # San Francisco
print(get_carbon_content_from_coordinates(-22.9068, -43.1729))  # Rio de Janeiro
