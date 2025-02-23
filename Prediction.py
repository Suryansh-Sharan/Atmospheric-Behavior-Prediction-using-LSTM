from tensorflow.keras.models import load_model

def load_and_predict(model_path, new_data):
    model = load_model(model_path)
    predictions = model.predict(new_data)
    return predictions
