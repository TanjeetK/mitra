import pandas as pd
from sklearn.cluster import KMeans
from utils import sresults

class ClusteringModel:
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None

    def fit_kmeans(self, data: pd.DataFrame):
        if not {"lat", "lon"}.issubset(data.columns):
            raise ValueError("Data must contain 'lat' and 'lon' columns.")

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        clusters = kmeans.fit_predict(data[["lat", "lon"]])
        self.model = kmeans

        clustered_data = data.copy()
        clustered_data["Cluster"] = clusters
        return clustered_data

    def main_cluster(self, clustered_data: pd.DataFrame):
        if self.model is None:
            raise RuntimeError("fit_kmeans must be called first.")

        main_cluster = clustered_data["Cluster"].value_counts().idxmax()
        main_center = self.model.cluster_centers_[main_cluster]
        return main_center.tolist()


if __name__ == "__main__":
    df = pd.read_csv("sample.csv")
    results = []

    for retailer_number, group_data in df.groupby("retailer_number"):
        clustering = ClusteringModel(n_clusters=4, random_state=42)
        clustered_group = clustering.fit_kmeans(group_data)
        main_center = clustering.main_cluster(clustered_group)

        results.append({
            "retailer_number": retailer_number,
            "lat": main_center[0],
            "lon": main_center[1]
        })

    sresults(results, "retailers_main_clusters.csv") 