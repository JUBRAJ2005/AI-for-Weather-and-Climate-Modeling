import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_synthetic_data(output_path: str, num_records: int = 1500) -> pd.DataFrame:
    """
    Generates a synthetic weather dataset with realistic patterns if real data is not provided.
    """
    np.random.seed(42)
    
    # 1. Generate dates
    dates = pd.date_range(start='2018-01-01', periods=num_records, freq='D')
    
    # 2. Generate weather features with relationships
    days = np.arange(num_records)
    
    # Temperature: Strong seasonal sine wave + noise
    temperature = 20 + 15 * np.sin(2 * np.pi * days / 365.25) + np.random.normal(0, 3, num_records)
    
    # Humidity: Inversely correlated with temperature + noise
    humidity = 85 - 0.7 * temperature + np.random.normal(0, 8, num_records)
    humidity = np.clip(humidity, 10, 100) # Keep within 10-100%
    
    # Wind speed: Gamma distribution (mostly low speeds, occasional high winds)
    wind_speed = np.random.gamma(shape=2.5, scale=2.0, size=num_records)
    
    # Atmospheric pressure: Normally distributed around average sea level pressure
    atmospheric_pressure = np.random.normal(1013.25, 8, num_records)
    
    # Rainfall: Zero-inflated exponential distribution (dry most days, exponential when raining)
    is_raining = np.random.rand(num_records) < 0.25 # 25% chance of rain
    rainfall = np.where(is_raining, np.random.exponential(scale=6.0, size=num_records), 0)
    
    df = pd.DataFrame({
        'date': dates,
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'atmospheric_pressure': atmospheric_pressure,
        'rainfall': rainfall
    })
    
    # 3. Inject missing values and outliers for pipeline robustness testing
    # Missing values
    missing_idx = np.random.choice(num_records, size=30, replace=False)
    df.loc[missing_idx, 'humidity'] = np.nan
    
    missing_idx_pressure = np.random.choice(num_records, size=20, replace=False)
    df.loc[missing_idx_pressure, 'atmospheric_pressure'] = np.nan
    
    # Duplicates
    df = pd.concat([df, df.iloc[np.random.choice(num_records, 15)]], ignore_index=True)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[PREPROCESSING] Synthetic data saved -> {output_path}")
    
    return df

def load_data(filepath: str) -> pd.DataFrame:
    """Loads dataset using pandas."""
    try:
        df = pd.read_csv(filepath)
        print(f"[PREPROCESSING] Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed loading data: {e}")
        return pd.DataFrame()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values, duplicates, and performs basic outlier treatment."""
    print("[PREPROCESSING] Cleaning data...")
    # 1. Handle Duplicates
    initial_len = len(df)
    df = df.drop_duplicates()
    print(f"  -> Dropped {initial_len - len(df)} duplicate records.")
    
    # 2. Handle Missing Values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  -> Imputed missing values in '{col}' with median: {median_val:.2f}")
            
    # 3. Basic Outlier Handling using Interquartile Range (IQR) bounding
    # We apply this specifically to features we expect to be well-bounded
    for col in ['temperature', 'humidity', 'atmospheric_pressure']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower_bound, upper_bound)
        
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Converts date strings into day, month, year formats."""
    print("[PREPROCESSING] Engineering features...")
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
    # Drop original date to prevent model issues with datetime objects
    df = df.drop('date', axis=1)
    return df

def split_and_scale(df: pd.DataFrame, target_col: str = 'temperature', test_size: float = 0.2, random_state: int = 42):
    """Splits dataset into features/target and scales numerical features."""
    print("[PREPROCESSING] Splitting and scaling data (Target = Temperature)...")
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Train-test split (80-20 as requested)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Feature Scaling: StandardScaler is preferred when dealing with models like Linear Regression
    scaler = StandardScaler()
    
    feature_cols = X_train.columns
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)
    
    print(f"  -> Train set size: {len(X_train_scaled)}")
    print(f"  -> Test set size: {len(X_test_scaled)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
