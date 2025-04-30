import joblib
import pandas as pd
import argparse
import os

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path)

def load_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    return pd.read_csv(data_path)

def make_prediction(model, data):
    return model.predict(data)

def main(model_path, data_path):
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    
    print(f"Loading new data from: {data_path}")
    new_data = load_data(data_path)

    print("Making predictions...")
    predictions = make_prediction(model, new_data)

    print("Predictions:")
    print(predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions using a trained model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the .pkl model file")
    parser.add_argument("--data", type=str, required=True, help="Path to the new data CSV file")

    args = parser.parse_args()
    main(args.model, args.data)
