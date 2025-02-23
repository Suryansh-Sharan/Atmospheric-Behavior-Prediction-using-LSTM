def engineer_features(data):
    data['Temp_C_lag1'] = data['Temp_C'].shift(1)
    data['Press_kPa_lag1'] = data['Press_kPa'].shift(1)
    data['Temp_C_rolling_mean'] = data['Temp_C'].rolling(window=3).mean()
    data['Press_kPa_rolling_mean'] = data['Press_kPa'].rolling(window=3).mean()
    data.dropna(inplace=True)
    return data
