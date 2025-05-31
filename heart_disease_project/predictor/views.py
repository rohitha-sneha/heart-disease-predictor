from django.shortcuts import render
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load("predictor/model.pkl")
scaler = joblib.load("predictor/scaler.pkl")

def home(request):
    return render(request, "predictor/home.html")

def predict(request):
    if request.method == "POST":
        age = int(request.POST["age"])
        gender = int(request.POST["gender"])
        pulse = int(request.POST["pulse"])
        heart_rate = int(request.POST["heart_rate"])
        weight = int(request.POST["weight"])
        
        # Preprocess input
        input_data = np.array([[age, gender, pulse, heart_rate, weight]])
        input_scaled = scaler.transform(input_data)  # ✅ FIX: Ensure proper scaling
        
        # Predict
        prediction = model.predict(input_scaled)
        result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"

        return render(request, "predictor/result.html", {"result": result})

    return render(request, "predictor/home.html")

