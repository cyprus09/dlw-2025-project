import tensorflow as tf
import numpy as np
import os
import ee
import io
from PIL import Image
import requests

# Initialize Earth Engine
ee.Initialize()

def verify_forest_claim(model_path, coordinates, claimed_area, year):
    """
    Verify a forest area claim using the trained model.
    
    Args:
        model_path: Path to trained forest detection model
        coordinates: Dict with 'longitude' and 'latitude'
        claimed_area: Claimed forest area in hectares
        year: Year of the claim
    
    Returns:
        Dict with verification results
    """
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Get satellite imagery for the claimed location and year
    image_data = get_satellite_imagery(coordinates, year)
    
    # Preprocess the imagery for model input
    processed_image = preprocess_imagery(image_data)
    
    # Run forest detection model
    forest_mask = model.predict(processed_image)
    
    # Calculate detected forest area
    detected_area = calculate_forest_area(forest_mask, coordinates)
    
    # Calculate discrepancy
    area_discrepancy = abs(claimed_area - detected_area) / claimed_area if claimed_area > 0 else 1.0
    
    # Create verification report
    verification_score = 100 - (area_discrepancy * 100)
    verification_score = max(0, min(100, verification_score))  # Clamp to 0-100
    
    # Determine verification status
    if verification_score >= 80:
        status = "VERIFIED"
    elif verification_score >= 60:
        status = "MOSTLY_VERIFIED"
    elif verification_score >= 40:
        status = "PARTIALLY_VERIFIED"
    elif verification_score >= 20:
        status = "MOSTLY_UNVERIFIED"
    else:
        status = "UNVERIFIED"
    
    # Return verification results
    return {
        'coordinates': coordinates,
        'year': year,
        'claimed_area_hectares': claimed_area,
        'detected_area_hectares': detected_area,
        'area_discrepancy_percent': area_discrepancy * 100,
        'verification_score': verification_score,
        'verification_status': status
    }

def get_satellite_imagery(coordinates, year):
    """Get satellite imagery from GEE for the specified coordinates and year."""
    lon = coordinates['longitude']
    lat = coordinates['latitude']
    
    # Create point and region
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(1000)  # 1km buffer
    
    # Date range for the year
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    # Get Sentinel-2 imagery
    s2_collection = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(region) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    
    # Get median composite
    s2_median = s2_collection.median()
    
    # Select RGB bands and rescale
    rgb_image = s2_median.select(['B4', 'B3', 'B2']).divide(10000)
    
    # Get NDVI
    ndvi = s2_median.normalizedDifference(['B8', 'B4']).rename('NDVI')
    
    # Get land cover data
    dw_collection = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
        .filterBounds(region) \
        .filterDate(start_date, end_date)
    
    # Get the most common land cover class
    landcover = dw_collection.select('label').mode()
    
    # Combine into a multiband image
    combined = rgb_image.addBands(ndvi).addBands(landcover)
    
    # Get download URL for the region
    url = combined.getThumbURL({
        'region': region,
        'dimensions': '256x256',
        'format': 'png'
    })
    
    # Download the image
    response = requests.get(url)
    image_data = Image.open(io.BytesIO(response.content))
    
    return image_data

def preprocess_imagery(image_data):
    """Preprocess the satellite imagery for model input."""
    # Convert to numpy array
    image_array = np.array(image_data)
    
    # Normalize values
    image_array = image_array / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def calculate_forest_area(forest_mask, coordinates):
    """Calculate forest area in hectares from the forest mask."""
    # Count forest pixels
    forest_pixels = np.sum(forest_mask > 0.5)
    
    # Convert to area (approximate based on latitude)
    lat = coordinates['latitude']
    
    # Area per pixel varies by latitude (crude approximation)
    pixel_area_m2 = 100  # 10m x 10m Sentinel-2 pixel
    
    # Adjust for latitude (pixels get smaller near poles)
    adjusted_pixel_area = pixel_area_m2 * np.cos(np.radians(lat))
    
    # Calculate total area in hectares (1 hectare = 10,000 m²)
    area_hectares = (forest_pixels * adjusted_pixel_area) / 10000
    
    return area_hectares