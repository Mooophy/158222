import numpy as np
from numpy import linalg

def evaluate_clusters(df, cluster, features, centroid, within_cluster):
    sum=0
    clusted_df = df[df.cluster==cluster] if within_cluster else df[df.cluster!=cluster]
    clusted_df = clusted_df[features]
    for i, row in clusted_df.iterrows():
        sum += linalg.norm(np.array(centroid) - np.array([row[features[0]], row[features[1]], row[features[0]]]))
    return sum / len(clusted_df)

def get_coh(df, cluster, features, centroid):
    return evaluate_clusters(df, cluster, features, centroid, True)

def get_sep(df, cluster, features, centroid):
    return evaluate_clusters(df, cluster, features, centroid, False) 