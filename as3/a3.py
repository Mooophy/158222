import pandas as pd
import numpy as np
import itertools
from numpy import linalg
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def create_df_and_scale_for_task2():
    df = pd.read_csv('happiness.csv')
    df = df.loc[:, df.columns != 'country']
    minmax_scale = preprocessing.MinMaxScaler().fit(df)
    df[[c for c in df.columns]] = minmax_scale.transform(df)
    return df

def create_categorical_feature_as_target(df):
    df['democratic level'] = ['a' if d > 0.7 else 'c' if d < 0.45 else 'b' for d in df['democratic']]
    democracy = df.copy()
    democracy = democracy.loc[:, democracy.columns != 'democratic']
    return democracy

def explore_features(df, feature, ax):           
    zipped = zip(['a', 'b', 'c'], ('blue', 'red', 'green'))   
    ax.set_title(feature)
    for label, color in zipped:
        ax.hist(df[feature][df['democratic level'] == label], alpha=0.2, color=color, bins=15)

def plot_hists_for_features(df):
    f, ax = plt.subplots(3, 7, figsize=(36, 16), sharex=True)
    feature_ax = [(df.columns[i], ax[i//7][i%7]) for i in range(0, len(df.columns) - 1)]  

    for feature, ax in feature_ax:
        explore_features(df, feature, ax)
    plt.show()


#####################################
#####################################
#####################################

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


def train_kmeans_and_fill_score(df, combinations_df):
    for i, row in combinations_df.iterrows():
        
        if(i%10 == 0):
            print(str(i) + '/' + str(len(combinations_df)))
    
        features = [f for f in df.columns if row[f] == 1]
        km = KMeans(n_clusters=3, init='random')
        km.fit(df[features].values)
        prediction = km.predict(df[features].values)
        score = metrics.silhouette_score(df[features], prediction)     
        combinations_df.iloc[i, combinations_df.columns.get_loc("score")] = score    

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

def plot_3d_clustering_for_combinations(df, combinations_df):
    colors = {0:'r', 1:'b', 2:'g'}

    for i, row in combinations_df.iterrows():    
    
        # fit
        features = [f for f in combinations_df.columns if row[f] == 1]
        km = KMeans(n_clusters=3, init='random')
        km.fit(df[features].values)
        prediction = km.predict(df[features].values)
    
        # plot
        ax = Axes3D(plt.figure(figsize=(15, 12)), rect=[.01, 0, 0.95, 1], elev=30, azim=134)
        ax.scatter(df[features[0]], df[features[1]], df[features[2]], c=[colors[p] for p in prediction], s=10, alpha=0.5)
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_zlabel(features[2])        
        ax.set_title( 'Subset ' + str(i) + ' : ' + ' vs '.join(features), fontsize = 18)
    
        df['cluster'] = prediction
        for cluster in [0, 1, 2]:
        
            # centorid
            x = df[df.cluster==cluster][features[0]].mean()
            y = df[df.cluster==cluster][features[1]].mean()
            z = df[df.cluster==cluster][features[2]].mean()    
            centroid = [x, y, z]
            ax.scatter([x], [y], [z], c=colors[cluster], s=180)
        
            # evaluation 
            coh = get_coh(df, cluster, features, centroid)
            sep = get_sep(df, cluster, features, centroid)   
                        
            # text
            t = ' , '.join(
                [
                    "coh=" + "{:.2f}".format(coh),
                    "sep=" + "{:.2f}".format(sep),
                    "c/s=" + "{:.2f}".format(coh/sep),
                ]
            )
            ax.text(0, 0.2, -1 * cluster/16 - 0.45, t, color=colors[cluster], alpha=0.8, fontsize=14)    
        
        plt.show()