#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv("sample.csv")
print(df.head())

sretailers = df['retailer_number'].drop_duplicates().sample(10, random_state=42)
print(sretailers)

for retailer in sretailers:
    r_df = df[df['retailer_number'] == retailer]
    coords = r_df[['lat', 'lon']].values

    inertias = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(coords)
        inertias.append(kmeans.inertia_)

    plt.figure()
    plt.plot(range(1, 11), inertias, 'ro-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title(f'Elbow Method for {retailer}')
    plt.show()

retailer_number = df['retailer_number'].iloc[0]
subset = df[df['retailer_number'] == retailer_number]
coords = subset[['lat', 'lon']].values

kmeans = KMeans(n_clusters=4, random_state=42)
subset['cluster'] = kmeans.fit_predict(coords)

plt.scatter(subset['lon'], subset['lat'], c=subset['cluster'], cmap='viridis', s=30)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 1], centroids[:, 0], c='black', marker='X', s=200)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"Retailer {retailer_number}")
plt.show()

newdf = []
for retailer, group in df.groupby("retailer_number"):
    coords = group[['lat', 'lon']].values
    kmeans = KMeans(n_clusters=4, random_state=42)
    group['cluster'] = kmeans.fit_predict(coords)

    main_cluster = group['cluster'].value_counts().idxmax()
    center = kmeans.cluster_centers_[main_cluster]

    newdf.append({
        'retailer_number': retailer,
        'latt': center[0],
        'lonn': center[1]
    })

result_df = pd.DataFrame(newdf)
print(result_df)