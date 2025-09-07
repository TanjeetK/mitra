import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

def elbow(coords, retailer):
    inertias = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(coords)
        inertias.append(kmeans.inertia_)

    #plott
    plt.figure()
    plt.plot(range(1, 11), inertias, 'ro-')
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Retailer", retailer)
    plt.show()

# df = pd.read_csv("sample.csv")
# ret = df['retailer_number'].unique()[0:10] #unique retsiler

# for retailer in ret:
#     coords = df[df['retailer_number'] == retailer][['lat', 'lon']].values
#     if len(coords) > 1: 
#         elbow(coords, retailer)