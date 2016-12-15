"""Contour dictionary learning and singing clusters"""

import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import spherical_kmeans
import interactive_plot


def align_metadata(contour_files, meta_file):
    '''Align metadata to correspond to the order the contour files were processed.
    
    Parameters
    ----------
    contour_files : np.array    
        List of file names corresponding to each contour.
        
    meta_file : str
        Path to a file containing metadata for each recording.
        
    Returns
    -------
    df : pd.DataFrame
        Metadata for the whole dataset ordered in the same way as the contour files. 
    '''
    df = pd.read_csv(meta_file)
    uniq_files = np.unique(contour_files)
    
    inds = []
    for uniq_file in uniq_files:
        inds.append(np.where(df['Csv']==uniq_file)[0][0])
    inds = np.array(inds)

    df = df.iloc[inds, :].reset_index()
    return df


def dictionary_learning(X):
    '''Apply spherical Kmeans to learn a dictionary of contour features and 
    return the cluster encoding.
    
    Parameters
    ----------
    X : np.array
        The dataset of countour features (n_samples, n_features).
    
    Returns
    -------
    embed_matrix : np.array
        Spherical K means projection using linear encoding.
    '''
    # preprocessing
    X = StandardScaler().fit_transform(X)  # scaling
    X = PCA(whiten=True).fit_transform(X)  # whitening
    
    # spherical Kmeans for dictionary learning
    centroids = spherical_kmeans.spherical_kmeans(X, 100, num_iterations=200)
    embed_matrix = spherical_kmeans.encode_linear(X, centroids)    
    return embed_matrix


def histogram_activations(embed_matrix, contour_files):
    '''Compute a histogram of kmeans activations for each recording.
    
    Parameters
    ----------
    embed_matrix : np.array
        Spherical K means projection.
    contour_files : np.array
        List of file names corresponding to each contour.
    
    Returns
    -------
    hist_activations : np.array
        Histogram of activations for each recording.
    '''
    uniq_files = np.unique(contour_files)
    
    histograms = []
    for uniq_file in uniq_files:
        inds = np.where(contour_files == uniq_file)[0]
        hist = np.sum(embed_matrix[inds, :], axis=0)
        histograms.append((hist-hist.mean()) / hist.std())  # standardize histogram
    
    hist_activations = np.array(histograms)    
    return hist_activations


def silhouette_K(X, min_ncl=5, max_ncl=20, metric='euclidean'):
    '''Run K-means clustering for K in range [min_ncl, max_ncl] and return the 
    average silhouette score and the number of clusters K with the highest score.
    
    Parameters
    ----------
    X : np.array
        The data to be clustered
    min_ncl : int
        The minimum number of clusters to consider. 
    max_ncl : int
        The maximum number of clusters to consider. 
    metric : str
        The distance metric used in the estimation of the silhouette score, 
        choice between 'euclidean', 'cosine', 'mahalanobis' etc.
        
    Returns
    -------
    best_K : int
        The K number of clusters with highest silhouette score
    average_silhouette : np.array
        The average silhouette score for each K in the range [0, max_ncl], 
        nan values added for K < min_ncl.
    '''
    average_silhouette = []
    for i in range(min_ncl):
        average_silhouette.append(np.nan)     
    for ncl in range(min_ncl, max_ncl):
        cl_pred = KMeans(n_clusters=ncl, random_state=50).fit_predict(X)
        average_silhouette.append(silhouette_score(X, cl_pred, metric=metric))
    
    average_silhouette = np.array(average_silhouette)
    best_K = np.nanargmax(average_silhouette)    
    return best_K, average_silhouette


def create_clusters(data):
    '''Find the optimal K number of clusters using silhouette score and predict 
    cluster assignment for each sample in the data.
    
    Parameters
    ----------
    data : np.array
        The dataset of learned features (n_samples, n_features).
    
    Returns
    -------
    cluster_pred : np.array
        Cluster assignment for each sample in the data.
    '''
    best_K, _ = silhouette_K(data, min_ncl=5, max_ncl=20)
    model = KMeans(n_clusters=best_K, random_state=50).fit(data)
    cluster_pred = model.predict(data)
    return cluster_pred


def fit_TSNE(data):
    '''Fit 2D-TSNE embedding to be able to visualize the high-dimensional data.
    
    Parameters
    ----------
    data : np.array
        The dataset of learned features (n_samples, n_features).
    
    Returns
    -------
    xy_coords : np.array
        The 2D coordinates learned by TSNE. 
    '''
    model2D = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    xy_coords = model2D.fit_transform(data)    
    return xy_coords


def main(pickle_file, meta_file, html_file=None):
    '''Steps through the analysis of contour features for dictionary learning 
    and singing style cluster extraction. 
    
    Parameters
    ----------
    pickle_file : str
        Path to pickle file with precomputed contour features.
    meta_file : str
        Path to csv file with metadata for each recording.
    html_file : str
        Path to html file to store the interactive TSNE visualization.
    '''
    # load precomputed contour features
    contour_features, contour_files = pickle.load(open(pickle_file, 'rb'))
    df = align_metadata(contour_files, meta_file)
    
    embed_matrix = dictionary_learning(contour_features)
    hist_activations = histogram_activations(embed_matrix, contour_files)
    
    cluster_pred = create_clusters(hist_activations)
    
    xy_coords = fit_TSNE(hist_activations)
    if html_file is not None:
        interactive_plot.plot_2D_scatter(xy_coords[:, 0], xy_coords[:, 1], 
                              labels=cluster_pred, df=df, html_file=html_file)


if __name__ == "__main__":
    pickle_file = '../data/contour_data.pickle'
    meta_file = '../data/metadata.csv'
    #html_file = '../data/TSNE.html'
    html_file = None
    
    main(pickle_file, meta_file, html_file)