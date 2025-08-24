import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

def scale_features_for_qml(input_path, output_path):
    """
    Loads a dataset, scales its features to the [0, 2*pi] range for angle encoding,
    and saves it to a new file.
    """
    print(f"--- Processing: {input_path} ---")
    try:
        X = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'. Please generate it first.")
        return

    print(f"Scaling features to the range [0, 2*pi]...")
    # Use MinMaxScaler to scale features to the desired range for angle encoding
    scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))
    X_scaled = scaler.fit_transform(X)

    # Create a new DataFrame with the scaled features, preserving column names
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # Save the scaled data
    X_scaled_df.to_csv(output_path, index=False)
    print(f"Successfully saved scaled data to '{output_path}'\n")


if __name__ == '__main__':
    # Define the datasets to be processed
    datasets_to_process = {
        'pca': 'data/processed_features_pca.csv',
        'feature_selection': 'data/processed_features_feature_selection.csv',
        'selectkbest': 'data/processed_features_selectkbest.csv'
    }

    # Define output paths for the scaled datasets
    output_dir = 'data'
    scaled_datasets_paths = {
        'pca': os.path.join(output_dir, 'qml_data_pca.csv'),
        'feature_selection': os.path.join(output_dir, 'qml_data_feature_selection.csv'),
        'selectkbest': os.path.join(output_dir, 'qml_data_selectkbest.csv')
    }

    # Loop through and process each dataset
    for key, input_file in datasets_to_process.items():
        output_file = scaled_datasets_paths[key]
        scale_features_for_qml(input_file, output_file)

    print("All QML data scaling complete.")
