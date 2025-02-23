from DataCleaning import clean_data
from FeatureEngineering import engineer_features
from ModelTraining import prepare_data_lstm, hyperparameter_tuning, create_model
from Prediction import load_and_predict
from sklearn.metrics import mean_squared_error

def main():
    # Load and clean data
    data = clean_data('D:/ModelTraining/Model/Weather Data.csv')

    # Engineer features
    data = engineer_features(data)

    # Prepare data for LSTM
    X_train_lstm, X_test_lstm, y_train, y_test = prepare_data_lstm(data)

    # Perform hyperparameter tuning
    best_params = hyperparameter_tuning(X_train_lstm, y_train, X_test_lstm, y_test)

    # Train the final model with the best hyperparameters
    final_model = create_model(units=best_params['units'], learning_rate=best_params['learning_rate'])
    final_model.fit(X_train_lstm, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'])

    # Evaluate the model
    y_pred = final_model.predict(X_test_lstm)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Final Model Mean Squared Error: {mse}')

    # Save the model in the Keras format
    final_model.save('lstm_model.keras')

    # Load the model and make predictions
    new_data_frame = clean_data('D:/ModelTraining/Model/new_data.csv')
    new_data_frame = engineer_features(new_data_frame)
    new_data = prepare_data_lstm(new_data_frame)[0]  # Use only the X part for prediction
    predictions = load_and_predict('lstm_model.keras', new_data)
    print(predictions)

if __name__ == "__main__":
    main()
