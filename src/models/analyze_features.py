import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

def analyze_model_features():
    """
    Loads trained models and data to analyze and save feature importances and correlations.
    """
    # --- Load Data and Models ---
    print("Loading data and trained models...")
    try:
        X = pd.read_csv('data/processed_features.csv')
        rf_model = joblib.load('results/models/Random_Forest_model.joblib')
        xgb_model = joblib.load('results/models/XGBoost_model.joblib')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have run the preprocessing and model training scripts first.")
        return

    feature_names = X.columns

    # --- Feature Importance ---
    print("Calculating feature importances...")
    rf_importances = rf_model.feature_importances_
    xgb_importances = xgb_model.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'RandomForest_Importance': rf_importances,
        'XGBoost_Importance': xgb_importances
    })

    importance_df = importance_df.sort_values(by='RandomForest_Importance', ascending=False)

    # Save feature importances to CSV
    metrics_dir = 'results/metrics'
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    importance_path = os.path.join(metrics_dir, 'feature_importances.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"Feature importances saved to '{importance_path}'")
    print("\nTop 10 Features (by Random Forest):")
    print(importance_df.head(10))

    # --- Feature Correlation ---
    print("\nCalculating feature correlation matrix...")
    corr_matrix = X.corr()

    # Save correlation matrix to CSV
    correlation_path = os.path.join(metrics_dir, 'feature_correlations.csv')
    corr_matrix.to_csv(correlation_path)
    print(f"Feature correlation matrix saved to '{correlation_path}'")

    # Plot and save correlation heatmap
    plots_dir = 'results/plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.figure(figsize=(20, 18))
    sns.heatmap(corr_matrix, annot=False, cmap='viridis')
    plt.title('Feature Correlation Matrix')
    heatmap_path = os.path.join(plots_dir, 'feature_correlation_heatmap.png')
    plt.savefig(heatmap_path)
    print(f"Correlation heatmap saved to '{heatmap_path}'")

if __name__ == '__main__':
    analyze_model_features()
