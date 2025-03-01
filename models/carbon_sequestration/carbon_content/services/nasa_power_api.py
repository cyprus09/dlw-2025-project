from statistics import stdev

import numpy as np
import requests


def get_nasa_power_climate_data(lat, lon, start_date="2018", end_date="2023"):
    """
    Retrieve climate data from NASA POWER API for a specific location and time period.

    Parameters:
    lat (float): Latitude
    lon (float): Longitude
    start_date (int): Start date in YYYY format
    end_date (int): End date in YYYY format

    Returns:
    dict: Climate variables for the location
    """
    base_url = "https://power.larc.nasa.gov/api/application/indicators/point"

    params = {
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON",
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()

        climate_variables = data

        annual_climate = {
            "mean_temp": calculate_annual_mean(climate_variables["CD_T2M_AVG"]),
            "annual_precip": calculate_annual_sum(climate_variables["CD_PRECTOTCORR"]),
            "solar_radiation": calculate_annual_mean(
                climate_variables["CD_RADIATION_AVG"]
            ),
            "relative_humidity": calculate_annual_mean(
                climate_variables["CD_MCDBR_DB"]
            ),
            "wind_speed": calculate_annual_mean(climate_variables["CD_WS10M_MONTH"]),
            "temp_seasonality": calculate_seasonality(climate_variables["CD_T2M_AVG"]),
        }

        climate_features = process_climate_features(data)

        return annual_climate, climate_features
    else:
        print(f"Error fetching NASA POWER data: {response.status_code}")
        return None


def calculate_annual_mean(daily_values):
    """Calculate mean from daily values dictionary."""
    return sum(float(v) for v in daily_values.values()) / len(daily_values)


def calculate_annual_sum(daily_values):
    """Calculate sum from daily values dictionary."""
    return sum(float(v) for v in daily_values.values())


def calculate_seasonality(daily_values):
    """Calculate stdev from daily dictionary."""
    return stdev(float(v) for v in daily_values.values())


def process_climate_features(climate_data):
    """
    Process climate data into useful features.

    Args:
        climate_data (dict): Climate variables

    Returns:
        array: Feature vector of climate characteristics
    """
    annual_precip = climate_data.get("CD_PRECTOTCORR_SUM", 1000)

    monthly_temps = []
    for m in range(1, 13):
        temp = climate_data.get("CD_T2M_AVG", {}).get(str(m), 15)
        monthly_temps.append(temp)

    mean_temp = np.mean(monthly_temps)
    min_temp = np.min(monthly_temps)
    max_temp = np.max(monthly_temps)
    temp_range = max_temp - min_temp

    monthly_precip = []
    for m in range(1, 13):
        precip = climate_data.get("CD_PRECTOTCORR", {}).get(str(m), annual_precip / 12)
        monthly_precip.append(precip)

    precip_seasonality = np.std(monthly_precip)
    precip_wettest = np.max(monthly_precip)
    precip_driest = np.min(monthly_precip)

    aridity_index = annual_precip / (mean_temp + 10)

    gdd = sum(max(0, t - 10) for t in monthly_temps)

    solar_radiation = np.mean(
        [
            climate_data.get("CD_RADIATION_AVG", {}).get(str(m), 200)
            for m in range(1, 13)
        ]
    )

    wind_speed = climate_data.get("CD_WS10M_ANNUAL", 3.0)

    features = {
        "annual_precip": annual_precip,
        "mean_temp": mean_temp,
        "min_temp": min_temp,
        "max_temp": max_temp,
        "temp_range": temp_range,
        "precip_seasonality": precip_seasonality,
        "precip_wettest": precip_wettest,
        "precip_driest": precip_driest,
        "aridity_index": aridity_index,
        "gdd": gdd,
        "solar_radiation": solar_radiation,
        "wind_speed": wind_speed,
    }

    return features


print(get_nasa_power_climate_data(-22.9068, -43.1729))
