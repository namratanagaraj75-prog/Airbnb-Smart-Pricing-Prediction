import pickle
import pandas as pd

# Load model
with open("price_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature columns
with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Sample input
sample_data = pd.DataFrame({
    "minimum_nights": [3],
    "number_of_reviews": [25],
    "calculated_host_listings_count": [2],
    "availability_365": [200],
    "neighbourhood_group": ["Manhattan"],
    "room_type": ["Entire home/apt"]
})

# Create dummy variables
sample_data = pd.get_dummies(sample_data)

# Add missing columns
for col in feature_columns:
    if col not in sample_data.columns:
        sample_data[col] = 0

# Arrange columns in correct order
sample_data = sample_data[feature_columns]

# Predict
prediction = model.predict(sample_data)

print(f"Predicted Airbnb Price: ${prediction[0]:.2f}")