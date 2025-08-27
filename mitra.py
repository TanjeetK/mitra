#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# In[28]:


df = pd.read_csv("sample.csv")
print(df.head())


# In[29]:


#elbow
sretailers = df['retailer_number'].drop_duplicates().sample(10, random_state=42)
print(sretailers)


# In[30]:


for retailer in sretailers:
    retailerdf = df[df['retailer_number'] == retailer]

    coords = retailerdf[['lat', 'lon']].values

    inertias = []
    krange = range(1, 11)  
    for k in krange:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(coords)
        inertias.append(kmeans.inertia_)

    plt.figure()
    plt.plot(krange, inertias, 'ro-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title(f'Elbow Method for {retailer}')
    plt.show()


# In[ ]:


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


# In[71]:


newdf = []
for retailer, group in df.groupby("retailer_number"):
    coords = group[['lat', 'lon']].values

    kmeans = KMeans(n_clusters=4, random_state=42)
    group['cluster'] = kmeans.fit_predict(coords)

    clusterr = group['cluster'].value_counts().idxmax()
    cluster_center = kmeans.cluster_centers_[clusterr]

    newdf.append({
        'retailer_number': retailer,
        'latt': cluster_center[0],
        'lonn': cluster_center[1]
    })

result_df = pd.DataFrame(newdf)
print(result_df)

