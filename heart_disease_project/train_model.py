import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
from collections import Counter

# Generate synthetic dataset
np.random.seed(42)
num_samples = 1000
data = {
    "Age": np.random.randint(30, 80, num_samples),
    "Gender": np.random.choice([0, 1], num_samples),  # 0 = Female, 1 = Male
    "PulseRate": np.random.randint(60, 120, num_samples),
    "HeartBeatRate": np.random.randint(60, 180, num_samples),
    "Weight": np.random.randint(50, 120, num_samples),
    "HeartDisease": np.random.choice([0, 1], num_samples, p=[0.85, 0.15])  # 🔴 85% NO, 15% YES (imbalanced)
}

df = pd.DataFrame(data)

# Check balance
print("Class distribution:", Counter(df["HeartDisease"]))

# Fix imbalance using equal samples
min_count = min(Counter(df["HeartDisease"]).values())
df_0 = df[df["HeartDisease"] == 0].sample(min_count, random_state=42)
df_1 = df[df["HeartDisease"] == 1].sample(min_count, random_state=42)
df = pd.concat([df_0, df_1])

# Split data
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = LogisticRegression()
model.fit(X_scaled, y)

# Save the model and scaler
joblib.dump(model, "predictor/model.pkl")
joblib.dump(scaler, "predictor/scaler.pkl")

print("Model trained with balanced data!")
