import pandas as pd
from sklearn.decomposition import PCA
import os

def prepare_data_with_pca(n_components=6):
    """
    Loads the processed data and applies PCA to reduce dimensionality.
    """
    print("Loading processed data...")
    try:
        X = pd.read_csv('data/processed_features.csv')
    except FileNotFoundError:
        print("Error: 'data/processed_features.csv' not found. Please run the main preprocessing script first.")
        return

    print(f"Applying PCA to reduce to {n_components} principal components...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Create a new DataFrame with the principal components
    pca_cols = [f'PC_{i+1}' for i in range(n_components)]
    X_pca_df = pd.DataFrame(data=X_pca, columns=pca_cols)

    # Save the PCA-transformed data
    output_path = 'data/processed_features_pca.csv'
    X_pca_df.to_csv(output_path, index=False)

    print(f"PCA-transformed data with {n_components} components saved to '{output_path}'")
    print("\nExplained variance ratio of the components:")
    print(pca.explained_variance_ratio_)
    print(f"\nTotal variance explained by {n_components} components: {sum(pca.explained_variance_ratio_):.4f}")

if __name__ == '__main__':
    prepare_data_with_pca()
