import pandas as pd
import os

def prepare_data_with_feature_selection(n_features=6):
    """
    Loads the processed data and selects the top N most important features.
    """
    print("Loading processed data and feature importances...")
    try:
        X = pd.read_csv('data/processed_features.csv')
        importances_df = pd.read_csv('results/metrics/feature_importances.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the main preprocessing and analysis scripts first.")
        return

    # Select the top N features based on RandomForest importance
    top_features = importances_df['Feature'].head(n_features).tolist()

    print(f"Selecting the top {n_features} most important features:")
    for feature in top_features:
        print(f"  - {feature}")

    # Filter the original feature set
    X_selected = X[top_features]

    # Save the feature-selected data
    output_path = 'data/processed_features_feature_selection.csv'
    X_selected.to_csv(output_path, index=False)

    print(f"\nData with selected features saved to '{output_path}'")

if __name__ == '__main__':
    prepare_data_with_feature_selection()
