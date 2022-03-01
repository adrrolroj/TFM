from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering, AffinityPropagation
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from analysis import pie_graph


def elbow_diagram_KMEANS(data, colums, name):
    df = data[colums]
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(df)
        distortions.append(kmeanModel.inertia_)

    plt.figure(figsize=(16, 8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.savefig(f'exports/clustering/{name}_kmeans_elbow_diagram.png')
    plt.show()


def k_means_dataframe(data, data_norm, colums, k, name_column):
    df = data_norm[colums]
    kmean_model = KMeans(n_clusters=k, n_init=10).fit(df)
    category, number = np.unique(kmean_model.labels_, return_counts=True)
    pie_graph(category, number, f'kmeans_{name_column}')
    data[name_column] = kmean_model.labels_.tolist()
    return data, kmean_model.cluster_centers_


def spectral_clustering_dataframe(data, data_norm, colums, k, name_column):
    df = data_norm[colums]
    spectral_clustering_model = SpectralClustering(n_clusters=k, eigen_solver="arpack",
                                                   affinity="nearest_neighbors", assign_labels='discretize',
                                                   n_init=10, random_state=10).fit(df)
    category, number = np.unique(spectral_clustering_model.labels_, return_counts=True)
    pie_graph(category, number, f'spectral_{name_column}')
    data[name_column] = spectral_clustering_model.labels_.tolist()
    data_norm[name_column] = spectral_clustering_model.labels_.tolist()
    col = colums[:]
    col.append(name_column)
    centers = data_norm[col].groupby([name_column]).mean().values
    return data, centers


def aglomerative_clustering_dataframe(data, data_norm, colums, k, name_column):
    df = data_norm[colums]
    aglomerative_clustering_model = AgglomerativeClustering(n_clusters=k, affinity='euclidean').fit(df)
    category, number = np.unique(aglomerative_clustering_model.labels_, return_counts=True)
    pie_graph(category, number, f'aglomerative_{name_column}')
    data[name_column] = aglomerative_clustering_model.labels_.tolist()
    data_norm[name_column] = aglomerative_clustering_model.labels_.tolist()
    col = colums[:]
    col.append(name_column)
    centers = data_norm[col].groupby([name_column]).mean().values
    return data, centers


def elbow_diagram_DBSCAN(data, colums, k):
    df = data[colums]
    nn = NearestNeighbors(n_neighbors=k).fit(df)
    distances, indices = nn.kneighbors(df)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.figure(figsize=(10, 8))
    plt.plot(distances)
    plt.savefig(f'exports/clustering/elbow_distances_dbscam.png')
    plt.show()


def pca_analysis(data, colums):
    df = data[colums].copy()
    # Standardize the data to have a mean of ~0 and a variance of 1
    X_std = StandardScaler().fit_transform(df)
    # Create a PCA instance: pca
    pca = PCA()
    principalComponents = pca.fit_transform(X_std)
    PCA_components = pd.DataFrame(principalComponents)
    # Plot the explained variances
    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_ratio_, color='black')
    plt.xlabel('PCA features')
    plt.ylabel('variance %')
    plt.xticks(features)
    plt.show()
    return PCA_components


def get_optimal_values_DBSCAM(data, colums):
    df = data[colums]
    min_samples = range(1, 30)
    eps = np.arange(0.1338, 0.1438, 0.01)
    output = []

    for ms in min_samples:
        for ep in eps:
            labels = DBSCAN(min_samples=ms, eps=ep).fit(df).labels_
            score = silhouette_score(df, labels)
            output.append((ms, ep, score))
    min_samples, eps, score = sorted(output, key=lambda x: x[-1])[-1]
    print(f"Best silhouette_score: {score}")
    print(f"min_samples: {min_samples}")
    print(f"eps: {eps}")

    labels = DBSCAN(min_samples=min_samples, eps=eps).fit(df).labels_
    clusters = len(Counter(labels))
    print(f"Number of clusters: {clusters}")
    print(f"Number of outliers: {Counter(labels)[-1]}")
    print(f"Silhouette_score: {silhouette_score(df, labels)}")
    return min_samples, eps


def DBSCAN_dataframe(data, data_norm, colums, eps, min_samples):
    df = data_norm[colums]
    dbscam_model = DBSCAN(eps=eps, min_samples=min_samples).fit(df)
    category, number = np.unique(dbscam_model.labels_, return_counts=True)
    for i, c in enumerate(category):
        print(f'Category: {c}, number: {number[i]}')
    pie_graph(category, number, 'dbscan')
    data['category'] = dbscam_model.labels_.tolist()
    return data
