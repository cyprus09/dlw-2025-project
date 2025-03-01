import os

import requests
from dotenv import load_dotenv

load_dotenv()


def get_coordinates_from_location_name(location_name):
    """
    Get latitude and longitude for a location using OpenCage Geocoding API.

    Args:
        location_name (str): Location name

    Returns:
        tuple: Latitude and Longitude
    """
    api_key = os.getenv("GEOCODIFY_API_KEY")
    url = f"https://api.geocodify.com/v2/geocode?q={location_name}&api_key={api_key}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "response" in data and len(data["response"]) > 0:
                lat = data["response"]["features"][0]["geometry"]["coordinates"][1]
                lon = data["response"]["features"][0]["geometry"]["coordinates"][0]
                return lat, lon
            else:
                print("No results found in the response.")
                return None
        else:
            print(f"Error fetching coordinates: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception while fetching coordinates: {e}")
        return None
