import pandas as pd
from sklearn.model_selection import train_test_split
import os

def create_train_test_split(key, input_path, y_data):
    """
    Loads a feature set, splits it into training and testing sets (80/20),
    and saves them to new CSV files.
    """
    print(f"--- Processing: {key} dataset ---")
    try:
        X = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'. Please generate it first.")
        return None, None

    # Split into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_data, test_size=0.2, random_state=42, stratify=y_data
    )

    # Define output paths
    output_dir = 'data'
    X_train_path = os.path.join(output_dir, f'X_train_{key}.csv')
    X_test_path = os.path.join(output_dir, f'X_test_{key}.csv')

    # Save the split feature data
    X_train.to_csv(X_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)

    print(f"Saved training features to '{X_train_path}'")
    print(f"Saved testing features to '{X_test_path}'\n")

    # Return y splits so we can save them once
    return y_train, y_test

if __name__ == '__main__':
    # Load the common target variable
    try:
        y = pd.read_csv('data/processed_target.csv').iloc[:, 0]
    except FileNotFoundError:
        print("Error: Target data 'data/processed_target.csv' not found. Please run main preprocessing first.")
        exit()

    # Define the datasets to be split
    datasets_to_split = {
        'pca': 'data/qml_data_pca.csv',
        'feature_selection': 'data/qml_data_feature_selection.csv',
        'selectkbest': 'data/qml_data_selectkbest.csv'
    }

    y_sets_saved = False
    for key, input_file in datasets_to_split.items():
        y_train, y_test = create_train_test_split(key, input_file, y)

        if y_train is not None and not y_sets_saved:
            # Save the y splits only once, as they are identical for all datasets
            y_train_path = 'data/y_train.csv'
            y_test_path = 'data/y_test.csv'
            
            y_train.to_csv(y_train_path, index=False, header=True)
            y_test.to_csv(y_test_path, index=False, header=True)
            
            print(f"Saved training targets to '{y_train_path}'")
            print(f"Saved testing targets to '{y_test_path}'\n")
            y_sets_saved = True

    print("All QML data splitting complete.")
