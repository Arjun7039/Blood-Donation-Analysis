#Import all necessary packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Web server imports
from flask import Flask, request, jsonify, send_from_directory
import threading
import webbrowser
import os
import pickle

# Check if model exists, if not train it
model_file = "blood_donation_model.pkl"
scaler_file = "scaler.pkl"

def train_model():
    #Load Dataset
    path = "blood-train.csv"
    df = pd.read_csv(path)

    # Drop unnecessary column (ID)
    df.drop(df.columns[0], axis=1, inplace=True)

    # Rename columns to match the code expectations
    df.columns = ['Months_since_last_donation', 'Number_of_donations', 'Total_volume_donated', 'Months_since_first_donation', 'Made_donation_in_march_2007']

    # Outlier Treatment using IQR
    cols_with_outliers = ['Months_since_last_donation', 'Number_of_donations', 'Total_volume_donated']
    for col in cols_with_outliers:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower_bound, lower_bound,
                           np.where(df[col] > upper_bound, upper_bound, df[col]))

    #Feature Engineering
    df['donation_rate'] = df['Number_of_donations'] / (df['Months_since_first_donation'] + 1)
    df['recency_inverse'] = 1 / (df['Months_since_last_donation'] + 1)

    #Define Features and Target
    X = df.drop('Made_donation_in_march_2007', axis=1)
    y = df['Made_donation_in_march_2007']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #Handle Class Imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Train XGBoost model (simplified without hyperparameter tuning for faster startup)
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(x_train, y_train)
    
    # Save model and scaler
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Evaluate model
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n📊 XGBoost Model Accuracy: {accuracy * 100:.2f}%")
    
    return model, scaler

# Load or train model
if os.path.exists(model_file) and os.path.exists(scaler_file):
    with open(model_file, 'rb') as f:
        best_model = pickle.load(f)
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    print("Loaded pre-trained model")
else:
    print("Training model...")
    best_model, scaler = train_model()
    print("Model training completed")

#Prediction function for API
def predict_donation_api(months_last, num_donations, total_volume, months_first):
    donation_rate = num_donations / (months_first + 1)
    recency_inverse = 1 / (months_last + 1)

    features = np.array([months_last, num_donations, total_volume, months_first,
                         donation_rate, recency_inverse]).reshape(1, -1)

    features_scaled = scaler.transform(features)
    pred = best_model.predict(features_scaled)
    probability = best_model.predict_proba(features_scaled)

    return {
        "prediction": "Will Donate" if pred[0] == 1 else "Will Not Donate",
        "probability": float(max(probability[0]))
    }

# Flask web server
app = Flask(__name__, static_folder='.')

# Serve the frontend
@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        data = request.get_json()
        
        # Extract features from request
        months_last = float(data['monthsLast'])
        num_donations = float(data['numDonations'])
        total_volume = float(data['totalVolume'])
        months_first = float(data['monthsFirst'])
        
        # Make prediction
        result = predict_donation_api(months_last, num_donations, total_volume, months_first)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def run_server():
    # Use port 5001 instead of 5000 to avoid conflicts
    app.run(host='localhost', port=5001, debug=False, use_reloader=False)

if __name__ == '__main__':
    # Start the web server in a separate thread
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Open the browser
    print("Starting web server...")
    print("Opening browser at http://localhost:5001")
    webbrowser.open('http://localhost:5001')
    
    # Keep the main thread alive
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Server stopped.")