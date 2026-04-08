import os
from src.preprocessing import (
    generate_synthetic_data, load_data, clean_data, 
    engineer_features, split_and_scale
)
from src.visualization import (
    plot_temperature_trends, plot_correlation_heatmap, 
    plot_humidity_vs_temp, plot_rainfall_distribution
)
from src.train_model import (
    train_linear_regression, train_random_forest, 
    evaluate_model, plot_feature_importance, save_best_model
)

def ensure_directories():
    """Enforces strict project folder structure."""
    dirs = ['data', 'outputs', 'model']
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def main():
    print("="*60)
    print("🌦️  AI for Weather and Climate Modeling - PHASE 1")
    print("="*60)
    
    ensure_directories()
    
    # ---------------------------------------------------------
    # 1. Dataset Requirements & Data Loading
    # ---------------------------------------------------------
    data_path = 'data/weather_raw_data.csv'
    
    if not os.path.exists(data_path):
        print("\n[INFO] Weather dataset not found. Generating realistic synthetic dataset...")
        raw_df = generate_synthetic_data(data_path, num_records=2500)
    else:
        print("\n[INFO] Loading historical weather dataset...")
        raw_df = load_data(data_path)
        
    # Copy specifically used for visualizations to retain datetime column
    eda_df = raw_df.copy()
    
    # ---------------------------------------------------------
    # 2. Exploratory Data Analysis (EDA)
    # ---------------------------------------------------------
    print("\n[INFO] Performing Exploratory Data Analysis... Generating plots...")
    plot_temperature_trends(eda_df, 'outputs')
    import numpy as np # Local import for corr heatmap
    plot_correlation_heatmap(eda_df, 'outputs')
    plot_humidity_vs_temp(eda_df, 'outputs')
    plot_rainfall_distribution(eda_df, 'outputs')
    print("[SUCCESS] All EDA Visualizations have been exported to the '/outputs' directory.")

    # ---------------------------------------------------------
    # 3. Data Preprocessing Pipeline
    # ---------------------------------------------------------
    print("\n[INFO] Initializing Data Preprocessing Pipeline...")
    df_cleaned = clean_data(raw_df)
    df_engineered = engineer_features(df_cleaned)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(
        df_engineered, target_col='temperature', test_size=0.2, random_state=42
    )

    # ---------------------------------------------------------
    # 4. Machine Learning Model Training
    # ---------------------------------------------------------
    print("\n[INFO] Initiating Machine Learning Model Training Sequence...")
    lr_model = train_linear_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # ---------------------------------------------------------
    # 5. Model Evaluation
    # ---------------------------------------------------------
    print("\n[INFO] Executing Model Evaluation and Metrics Validation...")
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Linear Regression (Baseline)")
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest (Advanced)")
    
    # Bonus: Compute and Plot Feature Importance
    plot_feature_importance(rf_model, list(X_train.columns), 'outputs')

    # Select Best Model based on Lowest RMSE
    print("\n[INFO] Concluding Selection Strategy...")
    if rf_metrics['RMSE'] < lr_metrics['RMSE']:
        best_model = rf_model
        best_name = "Random Forest Regressor"
    else:
        best_model = lr_model
        best_name = "Linear Regression"
    print(f"🏆 Best Performing Model Identified: {best_name}")

    # ---------------------------------------------------------
    # 6. Model Saving
    # ---------------------------------------------------------
    print("\n[INFO] Archiving Model Architecture for Future Scalability...")
    save_best_model(best_model, scaler, 'model')

    # ---------------------------------------------------------
    # 9. Phase-2 Integration Placeholders & Next Steps
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("✅ Phase-1 Pipeline Executed Successfully!")
    print("="*60)
    print("\n🚀 Next Steps for Phase-2 Upgrade Roadmap:")
    print(" 1. Deep Learning: Replace RF with sequence models (LSTM/GRU) for temporal dependencies.")
    print(" 2. Live Data Feed: Connect APIs like OpenWeatherMap to inject real-time telemetry.")
    print(" 3. Forecasting Engine: Implement N-day sliding window forecasting loops.")
    print(" 4. Dashboarding: Wrap the inference pipeline inside a clean Streamlit UI application.")
    
    # TODO: Phase-2 Placeholders
    # - Add deep learning model (LSTM/GRU)
    # - Integrate real-time weather API
    # - Build Streamlit dashboard
    # - Add forecasting capability

if __name__ == "__main__":
    main()
