import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
import os

def prepare_data_with_selectkbest(k=6):
    """
    Loads the processed data and uses SelectKBest to select the top k features.
    """
    print("Loading processed data...")
    try:
        X = pd.read_csv('data/processed_features.csv')
        y = pd.read_csv('data/processed_target.csv').iloc[:, 0]
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have run the main preprocessing script first.")
        return

    print(f"Applying SelectKBest to find the top {k} features based on ANOVA F-test...")
    
    # Initialize SelectKBest to select the top k features for classification
    selector = SelectKBest(score_func=f_classif, k=k)
    
    # Fit the selector to the data and transform the data
    X_new = selector.fit_transform(X, y)
    
    # Get the names of the columns that were kept
    selected_features_mask = selector.get_support()
    selected_features = X.columns[selected_features_mask]
    
    print(f"\nSelected the following {k} features:")
    for feature in selected_features:
        print(f"  - {feature}")

    # Create a new DataFrame with only the selected features
    X_selected_df = pd.DataFrame(X_new, columns=selected_features)

    # Save the feature-selected data
    output_path = 'data/processed_features_selectkbest.csv'
    X_selected_df.to_csv(output_path, index=False)

    print(f"\nData with features selected by SelectKBest saved to '{output_path}'")

if __name__ == '__main__':
    prepare_data_with_selectkbest()
