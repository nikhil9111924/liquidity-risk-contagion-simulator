import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import pickle

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

# Ensure model directory exists
MODEL_DIR.mkdir(exist_ok=True)

def train_execution_model():
    print("Loading Machine Learning Feature Matrix...")
    input_path = PROCESSED_DATA_DIR / "ml_feature_matrix.csv"
    df = pd.read_csv(input_path, index_col='Date', parse_dates=True)
    
    # Isolate Features (X) and Target (y)
    X = df.drop(columns=['Target_Amihud'])
    y = df['Target_Amihud']
    
    # Chronological split (80% Train, 20% Test) to prevent look-ahead bias
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training XGBoost Regressor on {len(X_train)} historical days...")
    
    # Initialize and train the model
    # Hyperparameters tuned for early-stage financial time-series stability
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Model Evaluation
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    print("\nModel Evaluation Metrics (Test Set):")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Feature Importance
    importance = pd.Series(model.feature_importances_, index=X.columns)
    print("\nFeature Importance:")
    print(importance.sort_values(ascending=False).to_string())
    
    # Serialize model to disk
    model_path = MODEL_DIR / "xgboost_liquidity_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"\nSerialized model weights saved to {model_path}")

if __name__ == "__main__":
    train_execution_model()