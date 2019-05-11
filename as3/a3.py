import pandas as pd
import numpy as np
import itertools
from numpy import linalg
from sklearn import preprocessing

def data_frame_for_task3(columns_to_drop):
    df = pd.read_csv('happiness.csv')
    df = df.loc[:, df.columns != 'country']
    df = df.drop(columns_to_drop, axis=1)
    return df

def scale(df):
    minmax_scale = preprocessing.MinMaxScaler().fit(df)
    df[df.columns] = minmax_scale.transform(df)
    return df

def create_df_for_all_combinations(df, len):
    all_combinations = [subset for subset in itertools.combinations(df.columns, len)]
    all_combinations_df = pd.DataFrame()
    for c in all_combinations:
        all_combinations_df = all_combinations_df.append({f: 1 if f in c else 0 for f in df.columns}, ignore_index=True)
    all_combinations_df['score'] = 0
    return all_combinations_df   


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