import numpy as np
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf

def prepare_data_lstm(data):
    X = data[['Temp_C', 'Press_kPa', 'Rel Hum_%']].values
    y = data['Temp_C'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    return X_train_lstm, X_test_lstm, y_train, y_test

def create_model(units=50, learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=(1, 3)))
    model.add(LSTM(units, return_sequences=False))
    model.add(Dense(1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def train_and_evaluate_model(X_train, y_train, X_test, y_test, units, learning_rate, batch_size, epochs):
    model = create_model(units=units, learning_rate=learning_rate)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def hyperparameter_tuning(X_train, y_train, X_test, y_test):
    param_dist = {
        'units': [50, 100],
        'learning_rate': [0.001, 0.01],
        'batch_size': [16, 32, 64],
        'epochs': [10, 20]
    }

    best_mse = float('inf')
    best_params = None

    for params in ParameterSampler(param_dist, n_iter=10, random_state=42):
        mse = train_and_evaluate_model(X_train, y_train, X_test, y_test, **params)
        if mse < best_mse:
            best_mse = mse
            best_params = params

    print(f"Best parameters: {best_params}")
    print(f"Best MSE: {best_mse}")
    return best_params

if __name__ == "__main__":
    data = clean_data('D:\ModelTraining\Model\Weather Data.csv')
    X_train_lstm, X_test_lstm, y_train, y_test = prepare_data_lstm(data)
    best_params = hyperparameter_tuning(X_train_lstm, y_train, X_test_lstm, y_test)
    final_model = create_model(units=best_params['units'], learning_rate=best_params['learning_rate'])
    final_model.fit(X_train_lstm, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'])
    final_model.save('lstm_model.h5')
