import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(data_path):
    # Load dataset
    df = pd.read_csv(data_path)

    # Handle missing values (drop rows with NaN)
    df = df.dropna()

    # Encode categorical variables
    le_degree = LabelEncoder()
    le_location = LabelEncoder()

    df['Degree'] = le_degree.fit_transform(df['Degree'])
    df['Location'] = le_location.fit_transform(df['Location'])

    # Convert Gender to numeric (0: Female, 1: Male)
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # Features and target
    X = df[['Age', 'Years_of_Schooling', 'Degree',
            'Years_of_Work_Experience', 'Gender', 'Location']]
    y = df['Salary']

    return X, y, le_degree, le_location
