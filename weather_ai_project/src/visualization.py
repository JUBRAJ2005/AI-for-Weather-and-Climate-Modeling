import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def set_aesthetics():
    """Configures modern plotting aesthetics."""
    sns.set_theme(style="whitegrid", context="talk")
    # Custom color palette suitable for weather variables
    sns.set_palette("muted")

def plot_temperature_trends(df: pd.DataFrame, output_dir: str):
    """Visualizes temperature trends over time."""
    set_aesthetics()
    plt.figure(figsize=(15, 6))
    
    if 'date' in df.columns:
        x_data = pd.to_datetime(df['date'])
    else:
        x_data = df.index
        
    plt.plot(x_data, df['temperature'], color='#E74C3C', alpha=0.7, linewidth=1.5)
    plt.title('Historical Temperature Trends', fontsize=20, weight='bold', pad=15)
    plt.xlabel('Date / Time', fontsize=14)
    plt.ylabel('Temperature (°C)', fontsize=14)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '1_temperature_trends.png'), dpi=300)
    plt.close()

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str):
    """Plots correlation heatmap of numeric features."""
    set_aesthetics()
    plt.figure(figsize=(10, 8))
    
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool)) if 'np' in globals() else None
    
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', 
                linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Heatmap', fontsize=18, weight='bold', pad=15)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, '2_correlation_heatmap.png'), dpi=300)
    plt.close()
    
    # Print strong correlations for insights
    print("\n[EDA INSIGHTS] Feature correlations with Temperature:")
    print(corr['temperature'].sort_values(ascending=False).to_string())

def plot_humidity_vs_temp(df: pd.DataFrame, output_dir: str):
    """Scatter plot of Humidity vs Temperature."""
    set_aesthetics()
    plt.figure(figsize=(12, 7))
    
    sns.scatterplot(x='humidity', y='temperature', data=df, 
                    alpha=0.6, color='#3498DB', edgecolor='w', s=60)
    
    # Add a regression line for visual flow
    sns.regplot(x='humidity', y='temperature', data=df, 
                scatter=False, color='#2C3E50', line_kws={'linestyle':'--'})
                
    plt.title('Humidity vs Temperature Analysis', fontsize=18, weight='bold', pad=15)
    plt.xlabel('Humidity (%)', fontsize=14)
    plt.ylabel('Temperature (°C)', fontsize=14)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, '3_humidity_vs_temp.png'), dpi=300)
    plt.close()

def plot_rainfall_distribution(df: pd.DataFrame, output_dir: str):
    """Visualizes the distribution of rainfall values."""
    set_aesthetics()
    plt.figure(figsize=(12, 6))
    
    # Filter to only days where it rained for a more meaningful distribution visually
    rain_days = df[df['rainfall'] > 0]['rainfall']
    
    sns.histplot(rain_days, bins=35, kde=True, color='#2ECC71', stat="density")
    plt.title('Rainfall Distribution (On Rainy Days)', fontsize=18, weight='bold', pad=15)
    plt.xlabel('Rainfall (mm)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, '4_rainfall_distribution.png'), dpi=300)
    plt.close()
