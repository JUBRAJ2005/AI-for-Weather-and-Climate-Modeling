import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series):
    """Trains a simple baseline Linear Regression model."""
    print("\n[MODEL] Training Linear Regression Baseline...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    """Trains a Random Forest Regressor optimized model."""
    print("[MODEL] Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=150, max_depth=15, 
                                  random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict:
    """Evaluates the model using MAE, RMSE, and R² Score."""
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    
    print(f"\n--- {model_name} Evaluation ---")
    print(f"🔹 MAE  (Mean Absolute Error): {mae:.4f}")
    print(f"🔹 RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"🔹 R²   (Coefficient of Determination): {r2:.4f}")
    
    return metrics

def plot_feature_importance(model, feature_names: list, output_dir: str):
    """Generates feature importance visualization for Tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Print rankings in terminal
        print("\n[BONUS] Feature Importances (Random Forest):")
        for i in range(len(feature_names)):
            print(f" {i+1}. {feature_names[indices[i]]:<20} ({importances[indices[i]]:.4f})")
            
        # Plot Rankings
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette="viridis")
        plt.title('Random Forest Feature Importance', fontsize=16, weight='bold')
        plt.xlabel('Relative Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, '5_feature_importance.png'), dpi=300)
        plt.close()

def save_best_model(model, scaler, model_dir: str):
    """Saves the best trained algorithm and its data scaler to disk."""
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'best_weather_model.pkl')
    scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\n[SAVE] Pipeline Assets Exported:")
    print(f"  -> Model Pipeline: {model_path}")
    print(f"  -> Scaler Object:  {scaler_path}")
