import pandas as pd

# Sample data
data = {
    'Date/Time': ['2025-02-23 10:00', '2025-02-23 11:00', '2025-02-23 12:00', '2025-02-23 13:00', '2025-02-23 14:00'],
    'Temp_C': [15.0, 16.0, 17.0, 18.0, 19.0],
    'Dew Point Temp_C': [10.0, 11.0, 12.0, 13.0, 14.0],
    'Rel Hum_%': [70, 72, 74, 76, 78],
    'Wind Speed_km/h': [10, 12, 14, 16, 18],
    'Visibility_km': [10.0, 9.5, 9.0, 8.5, 8.0],
    'Press_kPa': [101.2, 101.3, 101.4, 101.5, 101.6],
    'Weather': ['Clear', 'Clear', 'Partly Cloudy', 'Cloudy', 'Rain']
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('new_data.csv', index=False)
