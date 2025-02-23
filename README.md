# Atmospheric Behavior Prediction using LSTM

This project aims to analyze variations in temperature, pressure, and altitude to predict atmospheric behavior using historical CanSat telemetry data. The model used is a Long Short-Term Memory (LSTM) network, which is well-suited for time-series forecasting.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Validation](#model-validation)
- [Feature Engineering](#feature-engineering)
- [Deployment](#deployment)

## Project Overview

The goal of this project is to predict future temperature and pressure based on historical data. The LSTM model is trained to identify temperature/pressure anomalies and weather trends over multiple launches.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/atmospheric-prediction.git
   cd atmospheric-prediction

2. Create a virtual environment and activate it:
    ```bash
    python -m venv cansat_env
    source cansat_env/bin/activate # On Windows use `cansat_env\Scripts\activate`

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Project Structure

 The project is organized into several Python files, each responsible for a specific part of the workflow:

1. DataCleaning.py: Handles data loading, cleaning, and preprocessing.
2. FeatureEngineering.py: Contains code for creating new features.
3. ModelTraining.py: Includes code for hyperparameter tuning, model training, and validation.
4. Prediction.py: Contains code for making predictions using the trained model.
5. Main.py: The entry point of your application, which ties everything together.

## Usage
1. Data Cleaning:

Run DataCleaning.py to clean and preprocess the data.

2. Feature Engineering:

Run FeatureEngineering.py to create new features.

3. Model Training:

Run ModelTraining.py to train and evaluate the model.

4. Prediction:

Use Prediction.py to make predictions with the trained model.

5. Main Workflow:

Run Main.py to execute the entire workflow, including data cleaning, feature engineering, model training, evaluation, and prediction.

### Hyperparameter Tuning
Hyperparameter tuning is performed using a grid search approach to find the best configuration for the LSTM model. The hyperparameters tuned include the number of LSTM units, batch size, learning rate, and number of epochs.

### Model Validation
Cross-validation is used to ensure the model generalizes well to unseen data. The performance is evaluated using Mean Squared Error (MSE).

### Feature Engineering
Feature engineering involves creating lag features and rolling statistics to capture temporal dependencies and trends in the data.

### Deployment
Once satisfied with the model's performance, the model is saved and can be loaded for real-time predictions. The saved model is in the native Keras format (lstm_model.keras).


