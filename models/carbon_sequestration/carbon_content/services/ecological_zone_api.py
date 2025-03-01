from .nasa_power_api import get_nasa_power_climate_data
from .open_meteo_elevation_api import get_elevation_data


def get_ecological_zone(lat, lon):
    """
    Determine the ecological zone for a given latitude and longitude.
    Uses a combination of FAO Global Ecological Zones and climate data.

    Args:
        lat (float): Latitude
        lon (float): Longitude

    Returns:
        str: Ecological zone classification
    """
    annual_climate, _ = get_nasa_power_climate_data(lat, lon)

    if annual_climate:
        mean_temp = annual_climate.get("mean_temp", 15)
        annual_precip = annual_climate.get("annual_precip", 1000)

        elevation = get_elevation_data(lat, lon)

        abs_lat = abs(lat)

        if abs_lat < 23.5:
            if annual_precip > 2000:
                return "tropical_rainforest"
            elif annual_precip > 1000:
                return "tropical_moist_forest"
            else:
                return "tropical_dry_forest"

        elif abs_lat < 35:
            if elevation > 1000:
                return "montane"
            elif annual_precip > 1000:
                return "subtropical_forest"
            else:
                return "mediterranean"

        elif abs_lat < 55:
            if elevation > 1800:
                return "alpine"
            elif elevation > 1000:
                return "montane"
            elif mean_temp < 5:
                return "temperate_coniferous"
            elif mean_temp < 12:
                return "temperate_mixed"
            else:
                return "temperate_broadleaf"

        elif abs_lat < 65:
            return "boreal_forest"

        else:
            if elevation > 1000:
                return "alpine"
            else:
                return "boreal_forest"
    else:
        abs_lat = abs(lat)

        if abs_lat < 23.5:
            return "tropical_moist_forest"
        elif abs_lat < 35:
            return "subtropical_forest"
        elif abs_lat < 55:
            return "temperate_mixed"
        else:
            return "boreal_forest"


print(get_ecological_zone(37.7749, -122.4194))  # San Francisco
print(get_ecological_zone(-22.9068, -43.1729))  # Rio de Janeiro
