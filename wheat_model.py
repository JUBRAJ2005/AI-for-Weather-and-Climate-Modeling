import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("=== Wheat Yield Climate Model ===")
print("Loading dataset...")

# 1. LOAD & INSPECT DATASET
try:
    df = pd.read_csv('Custom_Crops_yield_Historical_Dataset.csv')
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nColumns:", list(df.columns))
    print("\nCrops available:", df['Crop'].unique() if 'Crop' in df.columns else "No 'Crop' column")
    
    # Check for wheat
    if 'wheat' in df['Crop'].str.lower().values:
        df_wheat = df[df['Crop'].str.lower() == 'wheat'].copy()
    else:
        print("⚠️ No 'wheat' found. Using all crops for demo (filter later)")
        df_wheat = df.copy()
    
    print(f"Wheat/other samples: {len(df_wheat)}")
    
except FileNotFoundError:
    print("❌ ERROR: Download 'Custom_Crops_yield_Historical_Dataset.csv' from https://www.kaggle.com/datasets/zoya77/indian-historical-crop-yield-and-weather-data")
    print("Place in folder and rerun!")
    exit()

# 2. AUTO-FIND FEATURES & TARGET (robust to column names)
numeric_cols = df_wheat.select_dtypes(include=[np.number]).columns.tolist()
print("\nNumeric columns:", numeric_cols)

# Common targets (yield/production)
target_col = None
for col in ['Yield', 'yield', 'Production', 'production', 'kg_per_ha', 'Yield_kg_per_ha']:
    if col in df_wheat.columns:
        target_col = col
        break

if target_col is None:
    # Fallback: use last numeric as target
    target_col = numeric_cols[-1]
    print(f"Using '{target_col}' as target (edit if wrong)")

y = df_wheat[target_col].fillna(df_wheat[target_col].median())

# Climate-like features (exclude target, codes)
exclude = ['Dist Code', 'State Code', 'Year', 'Area_code', target_col, 'code']
features = [col for col in numeric_cols if col not in exclude][:7]  # Top 7
X = df_wheat[features].fillna(df_wheat[features].median())

print(f"Using features: {features}")
print(f"Target: {target_col}, range: {y.min():.1f}-{y.max():.1f}")

# 3. TRAIN MODEL
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. RESULTS
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"\n✅ SUCCESS! R² Score: {r2:.4f} (good: >0.7)")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")

# Feature importance
importances = pd.DataFrame({
    'feature': features, 
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop Features:")
print(importances.head())

# 5. PLOTS
plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Scatter plot
ax1.scatter(y_test, y_pred, alpha=0.6, color='green')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('Actual'); ax1.set_ylabel('Predicted')
ax1.set_title(f'Performance (R²={r2:.3f})')

# Bar chart
top5 = importances.head(5)
ax2.barh(range(len(top5)), top5['importance'], color='skyblue')
ax2.set_yticks(range(len(top5)))
ax2.set_yticklabels(top5['feature'])
ax2.set_xlabel('Importance')
ax2.set_title('Key Climate Drivers')

plt.tight_layout()
plt.show()

# 6. Odisha/Cuttack Prediction Example
print("\n🔮 Sample prediction (Odisha climate):")
sample = np.array([[22, 70, 800, 4, 18, 6.5, 1000]])  # temp,hum,precip,wind,solar,pH,area
cols_order = [X.columns.get_loc(c) for c in features[:len(sample[0])]]
pred = model.predict(sample[:, cols_order])[0]
print(f"Predicted yield: {pred:.0f} {target_col.split('_')[0]}")