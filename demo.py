import joblib
import numpy as np

def interactive_predict(model_path="models/salary_model.pkl"):
    print("🔍 Loading model for prediction...")
    data = joblib.load(model_path)
    model = data["model"]
    le_degree = data["le_degree"]
    le_location = data["le_location"]

    # Get user input
    print("🧑 Please enter the profile details:")
    age = int(input("Age: "))
    years_schooling = int(input("Years of Schooling: "))
    degree = input("Degree (High School/Bachelor/Master/PhD/Diploma): ")
    years_exp = int(input("Years of Work Experience: "))
    gender_str = input("Gender (Male/Female): ")
    location = input("Location (e.g., New York, Mumbai, Berlin): ")

    # Encode categorical values
    degree_encoded = le_degree.transform([degree])[0]
    location_encoded = le_location.transform([location])[0]
    gender_encoded = 1 if gender_str.lower() == "male" else 0

    # Arrange features
    features = np.array([[age, years_schooling, degree_encoded,
                          years_exp, gender_encoded, location_encoded]])

    # Predict salary
    predicted_salary = model.predict(features)[0]
    print(f"\n💰 Predicted Salary for this profile: ${predicted_salary:.2f}")
