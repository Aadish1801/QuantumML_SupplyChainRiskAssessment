import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os

def test_gps_clustering():
    """
    Loads the supply chain data, clusters GPS coordinates using K-Means,
    and saves a visualization of the clusters.
    """
    print("Loading dataset...")
    try:
        df = pd.read_csv('data/dynamic_supply_chain_logistics_dataset.csv')
    except FileNotFoundError:
        print("Error: Dataset not found. Make sure 'data/dynamic_supply_chain_logistics_dataset.csv' exists.")
        return

    gps_coords = df[['vehicle_gps_latitude', 'vehicle_gps_longitude']]

    # --- K-Means Clustering ---
    # The number of clusters (n_clusters) is a hyperparameter.
    # We're starting with 15 as a reasonable number for geographical zones.
    # This can be tuned later.
    num_clusters = 15
    print(f"Performing K-Means clustering with {num_clusters} clusters...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(gps_coords)

    # Add cluster labels to a new DataFrame for inspection
    clustered_df = gps_coords.copy()
    clustered_df['location_cluster'] = cluster_labels

    print("\nClustering complete. Displaying head of the new clustered data:")
    print(clustered_df.head())

    # --- Visualization ---
    print("\nGenerating and saving cluster visualization...")
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x='vehicle_gps_longitude',
        y='vehicle_gps_latitude',
        hue='location_cluster',
        data=clustered_df,
        palette='viridis',
        s=50, # size of points
        alpha=0.7
    )
    plt.title('Geographical Distribution of Supply Chain Location Clusters')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(title='Cluster ID')
    plt.grid(True)

    # Save the plot
    output_dir = 'results/plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'gps_location_clusters.png')
    plt.savefig(output_path)
    print(f"Successfully saved cluster plot to '{output_path}'")

if __name__ == '__main__':
    test_gps_clustering()
