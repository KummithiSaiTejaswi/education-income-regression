Education Income Regression Project
📌 Project Overview

This project predicts a person’s salary based on educational background, work experience, and location using a Linear Regression model.

The model uses features such as:

Age

Years of Schooling

Degree (High School, Bachelor, Master, PhD, Diploma)

Years of Work Experience

Gender

Location

The project allows you to train a model and predict salaries interactively.

📂 Project Structure
education_income_regression/
│── data/
│   └── student_income.csv        # Dataset file (100+ synthetic entries recommended)
│── models/
│   └── salary_model.pkl          # Saved trained model
│── preprocess.py                 # Data preprocessing code
│── train.py                      # Model training code
│── demo.py                       # Interactive prediction code
│── main.py                       # Entry point to train or predict
│── requirements.txt              # Required Python libraries
│── README.md                     # This file

📝 Requirements

Install the required Python packages:

pandas
numpy
scikit-learn
matplotlib
seaborn
joblib


Install them using:

pip install -r requirements.txt

⚙️ Steps to Run
1️⃣ Activate Virtual Environment
.\venv\Scripts\activate

2️⃣ Train the Model

This will load the dataset, preprocess it, train a Linear Regression model, and save it to models/salary_model.pkl.

python main.py --mode train --data data/student_income.csv


Expected Output:

🚀 Starting training...
📂 Loading dataset from: data/student_income.csv
✅ Dataset loaded. Features: (100, 6), Target: (100,)
✅ Model trained. R²: 0.95, MSE: 4500.32
💾 Model saved at models/salary_model.pkl


⚠️ Small datasets may show R² = 1.0 due to overfitting. For real projects, use a larger dataset and train/test split.

3️⃣ Predict Salary Interactively

After training, run:

python main.py --mode predict


You will be prompted to enter profile details:

🧑 Please enter the profile details:
Age: 28
Years of Schooling: 16
Degree (High School/Bachelor/Master/PhD/Diploma): Master
Years of Work Experience: 5
Gender (Male/Female): Male
Location (e.g., New York, Mumbai, Berlin): New York

💰 Predicted Salary for this profile: $69872.83

🛠 Features Used

Age – Age of the person (numeric)

Years_of_Schooling – Total years of formal education (numeric)

Degree – Highest degree attained (categorical)

Years_of_Work_Experience – Work experience in years (numeric)

Gender – Male/Female (categorical, converted to numeric)

Location – City of residence (categorical)

Target Variable: Salary (annual in USD)

⚡ Notes

Dataset can include Indian, American, and European cities.

Linear Regression is used for simplicity; other regression models can improve accuracy.

For small datasets, the model will overfit; consider splitting data into train/test for evaluation.

💡 Optional Improvements

Add visualizations like scatter plots, boxplots, and residual plots.

Expand dataset to cover more cities and realistic salary ranges.

Experiment with Random Forest, XGBoost, or other regression algorithms.