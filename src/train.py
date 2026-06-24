import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load Dataset
df = pd.read_csv("data/AB_NYC_2019.csv")

# Select Features
df = df[
    [
        "neighbourhood_group",
        "room_type",
        "minimum_nights",
        "number_of_reviews",
        "calculated_host_listings_count",
        "availability_365",
        "price",
    ]
]

# Remove Missing Values
df = df.dropna()

# Encode Categorical Features
df = pd.get_dummies(
    df,
    columns=["neighbourhood_group", "room_type"],
    drop_first=True
)

# Features and Target
X = df.drop("price", axis=1)
y = df["price"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)

print("=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)
print(f"MAE: {mae:.2f}")

# Cross Validation
cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="neg_mean_absolute_error"
)

print(f"Cross Validation MAE: {-cv_scores.mean():.2f}")

# Save Model
with open("price_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save Feature Names
with open("feature_columns.pkl", "wb") as file:
    pickle.dump(X.columns.tolist(), file)

print("\nModel saved successfully!")
print("price_model.pkl created")
print("feature_columns.pkl created")