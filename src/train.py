import pandas as pd
import json
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

print("Name: Panduga Maheswar Reddy")
print("Roll Number: 2022BCS0185")
os.makedirs("outputs", exist_ok=True)
# Load dataset
df = pd.read_csv("data/winequality-red.csv", sep=";")


X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline with preprocessing


#model = Pipeline([
#    ("scaler", StandardScaler()),
#    ("model", LinearRegression())
#])

model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("MSE:", mse)
print("R2 Score:", r2)

# Save model
with open("outputs/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save results
results = {
    "MSE": mse,
    "R2": r2
}

with open("outputs/results.json", "w") as f:
    json.dump(results, f)

print("Model and results saved successfully.")