# 🫀 Heart Disease Predictor

This project is a *machine learning-based web application* that predicts the likelihood of a person having heart disease based on medical input parameters. It uses a trained model and provides an intuitive interface for users to enter their health data and receive a prediction.

## 🚀 Features

- Predicts the risk of heart disease based on key medical indicators
- User-friendly web interface built with Django
- Model trained on a reliable heart disease dataset
- Includes input validation and result interpretation
- Secure and modular project structure

## 🧠 Tech Stack

- *Frontend*: HTML, CSS, Bootstrap
- *Backend*: Python, Django
- *Machine Learning*: Scikit-learn, Pandas, NumPy
- *Model*: Trained using Logistic Regression / Random Forest (specify your model)
- *Deployment*: Localhost (can be extended to Heroku, Render, etc.)

## 📁 Project Structure
heart_disease_project/ │ ├── predictor/               # Django app │   ├── templates/ │   │   └── predictor/ │   │       ├── home.html │   │       └── result.html │   ├── static/ │   │   └── images/ │   │       └── heart.jpg, gpt.jpg │   ├── migrations/ │   ├── views.py │   ├── urls.py │   ├── models.py │   └── forms.py │ ├── train_model.py          # Script to train and export the ML model ├── model.pkl               # Saved ML model ├── scaler.pkl              # Feature scaler └── manage.py
## screenshot
![Screenshot 2025-05-31 214702](https://github.com/user-attachments/assets/cee6d1b3-7786-4a17-9ed3-ddf5af0da295)



## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/rohitha-sneha/heart-disease-predictor.git
