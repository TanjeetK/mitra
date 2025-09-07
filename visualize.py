import pandas as pd
import matplotlib.pyplot as plt
from clustering import ClusteringModel

def plot_clusters(data, centroids, retailer_number):
    plt.scatter(data['lon'], data['lat'], c=data['Cluster'], cmap='viridis', s=30)
    plt.scatter(centroids[:, 1], centroids[:, 0], c='black', marker='X', s=200)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Cluster Visualization for Retailer {retailer_number}")
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("sample.csv")
    retailer_number = df['retailer_number'].iloc[0]
    subset = df[df['retailer_number'] == retailer_number]

    kmeans = ClusteringModel(n_clusters=4, random_state=42)
    clustered_subset = kmeans.fit_kmeans(subset)

    plot_clusters(clustered_subset, kmeans.model.cluster_centers_, retailer_number)
