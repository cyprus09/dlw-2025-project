import pandas as pd

coordinates = [
    [1, -74.0060, 40.7128, 0],   # New York (non-forest)
    [2, -79.3832, 43.6532, 0],   # Toronto (non-forest)
    [3, -0.1278, 51.5074, 0],    # London (non-forest)
    [4, -123.1207, 49.2827, 1],  # Vancouver (nearby forests)
    [5, -73.5673, 45.5017, 1],   # Montreal (nearby forests)
    [6, -77.0369, 38.9072, 1],   # Washington D.C. (has parks/forests)
    [7, -122.4194, 37.7749, 0],  # San Francisco (mainly urban)
    [8, -122.3321, 47.6062, 1],  # Seattle (forested areas)
    [9, -90.1994, 38.6270, 0],   # St. Louis (mainly urban)
    [10, -58.3816, -34.6037, 0], # Buenos Aires (mainly urban)
    [11, -46.6333, -23.5505, 0], # São Paulo (urban)
    [12, 2.3522, 48.8566, 0],    # Paris (urban)
    [13, 139.6917, 35.6895, 0],  # Tokyo (urban)
    [14, 151.2093, -33.8688, 0], # Sydney (urban)
    [15, 144.9631, -37.8136, 1], # Melbourne (near forests)
    [16, 116.4074, 39.9042, 0],  # Beijing (urban)
    [17, 103.8198, 1.3521, 0],   # Singapore (urban)
    [18, -43.1729, -22.9068, 1], # Rio de Janeiro (forests nearby)
    [19, -0.4796, 39.4699, 0],   # Valencia (urban)
    [20, -96.7970, 32.7767, 0],  # Dallas (urban)
]

import random

for i in range(21, 201):
    lat = random.uniform(-55, 55)
    lon = random.uniform(-180, 180)
    is_forest = random.choice([0, 1]) 
    coordinates.append([i, lon, lat, is_forest])

# Create a DataFrame
df = pd.DataFrame(coordinates, columns=['id', 'longitude', 'latitude', 'known_forest'])

# Save to CSV
df.to_csv('coordinates.csv', index=False)
print("Sample coordinates.csv file created with 200 locations.")