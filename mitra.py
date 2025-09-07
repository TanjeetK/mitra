#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from clustering import ClusteringModel
from utils import sresults

def main():
    df = pd.read_csv("sample.csv")

    results = []
    for retailer, group_data in df.groupby("retailer_number"):
        if len(group_data) > 1:
            clustering = ClusteringModel(n_clusters=4, random_state=42)
            clustered_group = clustering.fit_kmeans(group_data)
            main_center = clustering.main_cluster(clustered_group)

            results.append({
                "retailer_number": retailer,
                "lat": main_center[0],
                "lon": main_center[1]
            })

    sresults(results, "f_retailerloc.csv")

if __name__ == "__main__":
    main()