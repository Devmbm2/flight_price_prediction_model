from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from datetime import datetime
import asyncio

# Load the trained regression model and scaler using pickle
with open('../saved_models/extra_trees_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)  # Your regression model

with open('../saved_models/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)  # Scaler used for feature scaling

with open('../saved_models/label_encoder.pkl', 'rb') as le_file:
    label_encoders = pickle.load(le_file)  # Label encoders used for categorical feature encoding

# Initialize FastAPI app
app = FastAPI()

# Example model input format for customer data
class FlightData(BaseModel):
    airline: str
    flight: str
    source_city: str
    departure_time: str  # ISO formatted datetime string (e.g., 'Morning', 'Evening', etc.)
    stops: str  # Modified to handle 'zero' or other categorical values
    arrival_time: str  # ISO formatted datetime string (e.g., 'Morning', 'Night', etc.)
    destination_city: str
    class_type: str  # Assuming this refers to a 'class' feature (economy, business, etc.)
    duration: float
    days_left: int

# Function to clean and limit text (if needed, based on the customer's inputs)
def clean_text(text: str, max_length: int = 500):
    # Remove unwanted characters (e.g., special characters, numbers, etc.)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Limit the text length
    text = text[:max_length]  # Limit the text to max_length characters
    
    return text

# Function to convert categorical features and datetime strings to numerical features
def convert_datetime_features(departure_time: str, arrival_time: str):
    # Extract features based on time of day (Morning, Afternoon, Evening, Night)
    time_of_day_mapping = {
        "Early_Morning": 0,
        "Morning": 1,
        "Afternoon": 2,
        "Evening": 3,
        "Night": 4
    }
    
    departure_time = time_of_day_mapping.get(departure_time, -1)  # Default to -1 for unknown time
    arrival_time = time_of_day_mapping.get(arrival_time, -1)
    
    return [departure_time, arrival_time]

# Define a prediction endpoint for price prediction
@app.post("/predict/")
async def predict(flight_data: FlightData):
    # Simulate an asynchronous delay (e.g., time for preprocessing or model inference)
    await asyncio.sleep(1)
    
    # Clean and preprocess the flight data
    time_features = convert_datetime_features(flight_data.departure_time, flight_data.arrival_time)
    
    # Prepare data as a DataFrame
    flight_data_df = pd.DataFrame([{
        'Airline': flight_data.airline,
        'Flight': flight_data.flight,
        'SourceCity': flight_data.source_city,
        'Stops': 0 if flight_data.stops == 'zero' else 1,  # Convert 'zero' to 0
        'DestinationCity': flight_data.destination_city,
        'ClassType': flight_data.class_type,
        'Duration': flight_data.duration,
        'DaysLeft': flight_data.days_left,
        'DepartureTime': time_features[0],
        'ArrivalTime': time_features[1]
    }])
    
    # Apply label encoding for categorical variables
    for column in ['Airline', 'Flight', 'SourceCity', 'DestinationCity', 'ClassType']:
        le = label_encoders[column]  # Load the corresponding LabelEncoder
        flight_data_df[column] = le.transform(flight_data_df[column])
    
    # Define the preprocessing pipeline (Scaling numerical features)
    numerical_features = ['Stops', 'Duration', 'DaysLeft', 'DepartureTime', 'ArrivalTime']
    
    # Preprocess the flight data (scale numerical features using MinMaxScaler)
    flight_data_transformed = scaler.transform(flight_data_df[numerical_features])
    
    # Predict the flight price using the trained regression model
    predicted_price = model.predict(flight_data_transformed)[0]
    
    # Return the predicted price for the flight
    return {"predicted_price": predicted_price}
