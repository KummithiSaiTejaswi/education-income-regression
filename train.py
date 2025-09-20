import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from preprocess import load_and_preprocess

def train_model(data_path, model_path="models/salary_model.pkl"):
    print("🚀 Starting training...")
    print(f"📂 Loading dataset from: {data_path}")

    # Load and preprocess data
    X, y, le_degree, le_location = load_and_preprocess(data_path)
    print(f"✅ Dataset loaded. Features: {X.shape}, Target: {y.shape}")

    # Train regression model
    model = LinearRegression()
    model.fit(X, y)

    # Evaluate model
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    print(f"✅ Model trained. R²: {r2:.2f}, MSE: {mse:.2f}")

    # Save model and encoders
    joblib.dump({
        "model": model,
        "le_degree": le_degree,
        "le_location": le_location
    }, model_path)
    print(f"💾 Model saved at {model_path}")
