# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib  # For saving the model

# Load the dataset (assuming you have it as a CSV after web scraping)
data = pd.read_csv('air_quality_data.csv')

# Feature Engineering (Assuming relevant features exist)
X = data[['temperature', 'humidity', 'pollutant1', 'pollutant2']]  # Replace with actual features
y = data['AQI']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')

# Save the model
joblib.dump(rf, 'random_forest_aqi_model.pkl')

# Deployment (Flask Web App)
# Import Flask to create an API endpoint
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the saved model
model = joblib.load('random_forest_aqi_model.pkl')

@app.route('/predict', methods=['POST'])
def predict_aqi():
    data = request.json
    features = np.array([data['temperature'], data['humidity'], data['pollutant1'], data['pollutant2']]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'AQI': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
