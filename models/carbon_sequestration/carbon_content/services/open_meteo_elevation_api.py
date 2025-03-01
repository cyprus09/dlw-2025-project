import requests


def get_elevation_data(lat, lon):
    """
    Get elevation data for a given latitude and longitude using Open-Elevation API.

    Args:
        lat (float): Latitude
        lon (float): Longitude

    Returns:
        float: Elevation in meters
    """
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            elevation = data["elevation"][0]
            return elevation
        else:
            print(f"Error fetching elevation data: {response.status_code}")
            return get_elevation_fallback(lat, lon)
    except Exception as e:
        print(f"Exception while fetching elevation data: {e}")
        return get_elevation_fallback(lat, lon)


def get_elevation_fallback(lat, lon):
    """
    Fallback method to get elevation using USGS Elevation Point Query Service.

    Args:
        lat (float): Latitude
        lon (float): Longitude

    Returns:
        float: Elevation in meters
    """
    url = (
        f"https://nationalmap.gov/epqs/pqs.php?x={lon}&y={lat}&units=Meters&output=json"
    )

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            elevation = data["USGS_Elevation_Point_Query_Service"]["Elevation_Query"][
                "Elevation"
            ]
            return float(elevation)
        else:
            print(
                f"Error fetching elevation data from fallback: {response.status_code}"
            )
            return 500
    except Exception as e:
        print(f"Exception while fetching elevation fallback data: {e}")
        return 500


print(get_elevation_data(37.7749, -122.4194))  # San Francisco
