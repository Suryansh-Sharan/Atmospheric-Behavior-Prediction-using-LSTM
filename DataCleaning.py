import pandas as pd

def load_and_clean_data(file_path):
    data = pd.read_csv(file_path)
    # Try parsing with the original format
    try:
        data['Date/Time'] = pd.to_datetime(data['Date/Time'], format='%m/%d/%Y %H:%M')
    except Exception:
        # If parsing fails, try the new format
        data['Date/Time'] = pd.to_datetime(data['Date/Time'], format='%Y-%m-%d %H:%M')
    return data

def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

def clean_data(file_path):
    data = load_and_clean_data(file_path)
    outliers_temp = detect_outliers(data, 'Temp_C')
    outliers_pressure = detect_outliers(data, 'Press_kPa')
    data_cleaned = data[~data.index.isin(outliers_temp.index) & ~data.index.isin(outliers_pressure.index)].copy()
    return data_cleaned
