from matplotlib import pyplot as plt, gridspec
from sklearn.cluster import DBSCAN, OPTICS, cluster_optics_dbscan
from sklearn.metrics.cluster import adjusted_rand_score
from torch.nn import MSELoss
import numpy as np
from scipy.spatial.distance import cdist
from SHADE.shade import SHADE
from tqdm import tqdm
import umap
from typing import List, Iterable, Dict, Tuple
import pandas as pd
from clustpy.deep.autoencoders import FeedforwardAutoencoder
import torch
from typing import Optional, Union
from clustpy.deep._data_utils import get_dataloader
from SHADE.shade.shade import _standardize
from clustpy.deep._train_utils import get_trained_autoencoder
from pathlib import Path
import os
import gzip
import struct
from typeguard import typechecked
from modules.customDbscan import PredictDBSCAN
from modules.baycon.common.Target import Target
from modules.baycon.baycon.run import executeCustom
from sklearn.metrics import pairwise_distances
import json
from dice_ml import Dice
import dice_ml
import networkx as nx
import heapq
from copy import deepcopy


np.random.seed(0)


def find_outliers_and_clusters(dbscan, X_umap, labels):
    """Identifies outliers and core samples."""
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    if hasattr(dbscan, 'core_sample_indices_'):
        core_samples_mask[dbscan.core_sample_indices_] = True

    core_points = X_umap[core_samples_mask]
    core_labels = labels[core_samples_mask]
    outlier_mask = (labels == -1)
    outliers = X_umap[outlier_mask]

    #print(len(outliers))

    distMatrix = cdist(outliers, core_points)

    minDistances = distMatrix.min(axis=1)

    sortedIndices = np.argsort(-minDistances)

    sortedOutliers = outliers[sortedIndices]

    return core_points, core_labels, outliers, outlier_mask, sortedIndices


def select_outlier(outliers, outlier_mask, index):
    """Selects an outlier by index."""
    if len(outliers) == 0:
        raise ValueError("No outliers to analyze.")
    if index >= len(outliers):
        raise IndexError(f"Outlier index {index} is out of range. Total outliers: {len(outliers)}.")

    selected_outlier = outliers[index]
    selected_outlier_index = np.where(outlier_mask)[0][index]
    return selected_outlier, selected_outlier_index


def analyze_outlier(selected_outlier, core_points, core_labels, dbscan):
    """Analyzes an outlier and generates counterfactuals."""
    distances = np.linalg.norm(core_points - selected_outlier, axis=1)
    sorted_indices = np.argsort(distances)
    sorted_core_labels = core_labels[sorted_indices]
    sorted_core_points = core_points[sorted_indices]

    unique_nearby_clusters = []
    closest_core_points = []
    for lbl, pt in zip(sorted_core_labels, sorted_core_points):
        if lbl not in unique_nearby_clusters and lbl != -1:
            unique_nearby_clusters.append(lbl)
            closest_core_points.append(pt)
        #if len(unique_nearby_clusters) == 5:
         #   break

    #if len(unique_nearby_clusters) < 2:
     #   raise ValueError("Less than two distinct clusters found near this outlier.")

    counterfactuals = []#{label:[] for label in np.unique(unique_nearby_clusters)}
    for cp, label in zip(closest_core_points, unique_nearby_clusters):
        vector = cp - selected_outlier
        distance = np.linalg.norm(vector)
        eps = dbscan.eps
        if distance > eps:
            movement = vector * ((distance - eps) / distance)
            counterfactual = selected_outlier + movement
        else:
            counterfactual = selected_outlier
        #counterfactuals[label].append(counterfactual)
        counterfactuals.append(counterfactual)

    return np.array(counterfactuals), unique_nearby_clusters

def visualize_results(X_umap, labels, selected_outlier, counterfactuals, unique_nearby_clusters):
    """Visualises the outlier and counterfactuals."""
    plt.figure(figsize=(12, 8))

    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        if k == -1:
            continue
        class_member_mask = (labels == k)
        plt.scatter(X_umap[class_member_mask, 0], X_umap[class_member_mask, 1], c=[col], edgecolor='k', s=50, label=f'Cluster {k}')

    plt.scatter(selected_outlier[0], selected_outlier[1], c='k', marker='x', s=100, label='Outlier')

    plt.scatter(counterfactuals[:, 0], counterfactuals[:, 1], facecolors='none', edgecolors=['r', 'g'], marker='o', s=150, label='Counterfactuals')

    for cf, c in zip(counterfactuals, ['r', 'g']):
        plt.plot([selected_outlier[0], cf[0]], [selected_outlier[1], cf[1]], color=c, linestyle='--')

    plt.title('Outlier with Counterfactuals')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend()
    plt.grid(True)
    plt.show()

def fit_SHADE(data, epochs, lossFuncion=MSELoss()):
    shade = SHADE(random_state=42, clustering_epochs =epochs, loss_fn=lossFuncion)
    shade.fit(data)

    encoding = shade.encode(data)
    np.savetxt("encoding.csv", encoding, delimiter=",")
    return encoding, shade

def draw_optics(X, best_labels, best_params):
    clust = OPTICS(min_samples=best_params[1], xi=0.05, min_cluster_size=0.05)

    # Run the fit
    clust.fit(X)

    labels_050 = cluster_optics_dbscan(
        reachability=clust.reachability_,
        core_distances=clust.core_distances_,
        ordering=clust.ordering_,
        eps=best_params[0],
    )

    space = np.arange(len(X))
    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]

    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(2, 1)
    ax1 = plt.subplot(G[0, :])
    ax2 = plt.subplot(G[1, :])

    # Reachability plot
    colors = ["darkorange", "indianred", "midnightblue", "turquoise", "lightcoral", "magenta", "lightsteelblue",
              "green", "gold", "forestgreen", "lightblue", "purple", "lightcyan", "black"]
    for klass in range(len(set(best_labels)) - 1):
        color = colors[klass]
        Xk = space[labels == klass]
        Rk = reachability[labels == klass]
        ax1.plot(Xk, Rk, color, marker='o', linestyle='None', alpha=0.3)
    ax1.plot(space[labels == -1], reachability[labels == -1], "k.", alpha=0.3)
    ax1.plot(space, np.full_like(space, best_params[0], dtype=float), "k-", alpha=0.5)
    ax1.set_ylabel("Reachability (epsilon distance)")
    ax1.set_title("Reachability Plot")

    colors = ["darkorange", "indianred", "midnightblue", "turquoise", "lightcoral", "magenta", "lightsteelblue",
              "green", "gold", "forestgreen", "lightblue", "purple", "lightcyan", "black"]
    for klass in range(len(set(best_labels)) - 1):
        color = colors[klass]
        Xk = X[labels_050 == klass]
        ax2.plot(Xk[:, 0], Xk[:, 1], color, marker='o', linestyle='None', alpha=0.3)
    ax2.plot(X[labels_050 == -1, 0], X[labels_050 == -1, 1], "k+", alpha=0.1)
    ax2.set_title(f"Clustering at {best_params[0]} epsilon cut\nDBSCAN")

    plt.tight_layout()
    plt.show()

def dbscan_opt(encoding, eps_values, min_samples_values,verbose=False):
    best_score = -1
    best_noise = 0
    best_params = None
    best_labels = None
    for eps in tqdm(eps_values, desc='DBSCAN grid search'):
        for min_samples in tqdm(min_samples_values, desc='Min Samples loop'):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(encoding)
            unique = np.unique(labels, return_counts=True)
            noise = 0 if unique[0][0] != -1 else unique[1][0]/len(labels)
            unique_labels = set(labels) - {-1}
            if len(unique_labels) > 1:
                try:
                    score = dcsi_score(encoding, labels)
                    score = score * (1-noise)
                except ValueError:
                    score = -1
                if verbose:
                    print(f"Params: eps={eps:.2f}, min_samples={min_samples}, DCSI Score: {score:.4f}, Noise: {noise}")
                
                #print(score, eps, min_samples)
                if score > best_score:
                    #print(score)
                    best_score = score
                    best_noise = noise
                    best_params = (eps, min_samples)
                    best_labels = labels
                    dest_dbscan_instance = dbscan
    if not best_params:
        raise ValueError("No suitable DBSCAN params found.")
    
    return best_params, best_score, best_noise, best_labels, dbscan


def dbscan_ari(encoding, groundtruth, eps_values, min_samples_values,verbose=False):
    best_score = -1
    best_noise = 0
    best_params = None
    best_labels = None
    j=0
    for eps in tqdm(eps_values):
        for min_samples in min_samples_values:
            j+=1
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(encoding)
            unique = np.unique(labels, return_counts=True)
            #print(unique)
            noise = 0 if unique[0][0] != -1 else unique[1][0]/len(labels)
            unique_labels = set(labels) - {-1}
            if len(unique_labels) > 1:
                try:

                    score = adjusted_rand_score(groundtruth, labels)
                    
                    #print(f'N:{j}/{len(eps_values) * len(min_samples_values)}', 'SCORE:', score, 'N CLUSTERS:', len(np.unique(labels)) - 1)
                except ValueError:
                    score = -1
                if verbose:
                    print(f"Params: eps={eps:.2f}, min_samples={min_samples}, ARI Score: {score:.4f}, Noise: {noise}, nClusters: {len(np.unique(labels))}")

                if score > best_score:
                    best_score = score
                    best_noise = noise
                    best_params = (eps, min_samples)
                    best_labels = labels

                    
    if not best_params:
        raise ValueError("No suitable DBSCAN params found.")

    return best_params, best_score, best_noise, best_labels

# Implementation of DCSI by
# - XXXX-2`
# - XXXX-3
# - XXXX-5: -

# XXXX-6
# XXXX-1
# XXXX-4

# Our modifications:
#    (1) translated from R to python


import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree


def dcsi_score(data, partition, min_pts=5):
    clusters = partition
    for i in range(len(clusters)):
        if np.sum(partition == clusters[i]) == 1:
            partition[partition == clusters[i]] = -1
            clusters[i] = -1
    # all clusters except for -1
    clusters = np.setdiff1d(clusters, -1)
    # if no clusters left or just one cluster left return 0
    if len(clusters) == 0 or len(clusters) == 1:
        return 0
    # exclude noise points from dataset
    data = data[partition != -1, :]
    # calculate squared euclidean distance
    dist = squareform(pdist(data)) ** 2

    # original labelling
    poriginal = partition
    # exclude noise points from labeling
    partition = partition[partition != -1]
    cluster_labels = np.unique(partition)
    n_clusters = len(cluster_labels)
    dcsi = 0
    MST = {}
    CORE_PTS = {}
    for i in range(0, n_clusters):
        # indices of objects in cluster i
        objects_cl = np.where(partition == clusters[i])[0]
        # distance in the cluster
        dist_i = dist[np.ix_(objects_cl, objects_cl)]
        epsilon = calculate_epsilon(dist_i, 2 * min_pts)
        CORE_PTS[cluster_labels[i]] = core_points(dist_i, epsilon, min_pts)
        if len(CORE_PTS[cluster_labels[i]]) == 0:
            return -1
        dist_i = dist_i[np.ix_(CORE_PTS[cluster_labels[i]], CORE_PTS[cluster_labels[i]])]
        MST[cluster_labels[i]] = minimal_spanning_tree(dist_i)

    for i in range(0, n_clusters - 1):
        for j in range(i + 1, n_clusters):
            part = pairwise_dcsi(MST, CORE_PTS, data, partition, cluster_labels[i], cluster_labels[j])

            dcsi = dcsi + part
    dcsi = (2 / (n_clusters * (n_clusters - 1))) * dcsi

    return dcsi


def calculate_epsilon(dist_i, k):
    distances = []
    for i in range(0, dist_i.shape[0]):
        dists = np.unique(dist_i[i])
        if k >= len(dists):
            distances.append(dists[-1])
        else:
            distances.append(dists[k])
    epsilon = np.median(distances)
    return epsilon


def core_points(dist, epsilon, min_pts):
    neighborhoods = []
    for i in range(len(dist)):
        row = []
        for j in range(len(dist)):
            if i != j:
                if dist[i, j] <= epsilon:
                    row.append(dist[i, j])
        neighborhoods.append(row)
    core_pts = [i for i in range(len(neighborhoods)) if len(neighborhoods[i]) > min_pts - 1]
    return core_pts


def pairwise_dcsi(MST, CORE_PTS, data, partition, i, j):
    sep_dcsi = pairwise_separation(CORE_PTS, data, partition, i, j)
    conn_dcsi = pairwise_connectedness(MST, i, j)
    q = sep_dcsi / conn_dcsi
    return q / (1 + q)


def pairwise_separation(CORE_POINTS, data, labels, i, j):
    # distances between core points in between C_i and C_j
    # subset to include internal nodes of cluster i only
    subset_i = data[labels == i, :]
    core_pts_i = CORE_POINTS[i]
    subset_i = subset_i[core_pts_i]
    # subset to include internal nodes of cluster j only
    subset_j = data[labels == j, :]
    core_pts_j = CORE_POINTS[j]
    subset_j = subset_j[core_pts_j]
    sep_dcsi_list = cdist(subset_i, subset_j, metric="euclidean") ** 2
    sep_dcsi = np.min(sep_dcsi_list)
    return sep_dcsi


def pairwise_connectedness(MST, i, j):
    conn_dcsi = max(cluster_conn(MST, i), cluster_conn(MST, j))
    return conn_dcsi


def cluster_conn(MST, i):
    """
    Conn_dcsi(C_i) = max d(x_i, x_j), (x_i, x_j) in V

    :param MST:
    :param i:
    :return:
    """
    # maximum edge weight of MST
    conn_dcsi = np.max(MST[i])
    return conn_dcsi


def minimal_spanning_tree(dist_i):
    # transform to array
    dist = np.array(dist_i)
    # calculate minimal spanning tree and extract adjacency matrix
    # this calculates Kruskal
    mst = minimum_spanning_tree(dist).toarray()
    # mst is upper triangular matrix, make it symmetric
    mst_temp = mst + mst.T
    return mst_temp


def optimizeUmap(X, y, dimEmbedding:int = 2):

    nNeighbours:List[float]=[2, 5, 10, 15, 20, 50, 100]
 
    minDists:List[float] = [0.0, 0.1, 0.25, 0.5, 0.8, 0.99]

    resultsDf = pd.DataFrame(columns = ['nNeighbours',
                                        'minDists',
                                        'eps',
                                        'minSample',
                                        'ari',
                                        'score'])


    for nNeighbour in tqdm(nNeighbours):

        for minDist in minDists:

            umapInstance:umap.UMAP = umap.UMAP(n_components=dimEmbedding, n_neighbors=nNeighbour,
                                               min_dist=minDist).fit(X)

            embeddedData:np.ndarray = umapInstance.transform(X)

            eps_values = [0.1, 0.2, 0.5, 1, 2, 5, 10]

            min_samples_values = [2, 3, 5, 7, 10, 15, 50, 100]

            for eps in eps_values:

                for minSample in tqdm(min_samples_values):
            
                    dbscanInstance:DBSCAN = DBSCAN(eps=eps, min_samples=minSample)

                    labels = dbscanInstance.fit_predict(embeddedData)

                    ari = adjusted_rand_score(labels_pred=labels, labels_true=y)

                    score = dcsi_score(embeddedData, labels)

                    resultsDf.loc[len(resultsDf+1)] = [nNeighbour,
                                                       minDist,
                                                       eps,
                                                       minSample,
                                                       ari,
                                                       score]
                    
    return resultsDf

    

def baseAutoencoder(X:np.ndarray, embeddingSize:int, standardize=True, standardizeAxis:int = 0,
                    batchSize:int = 500, pretrain_optimizer_params: dict = {"lr": 1e-3}, 
                    pretrain_epochs: int = 1000, device:str='cpu', optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                    loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), savePath:Optional[Union[str, Path]] = None
                    ):

    

    architecture:List[int] = [X.shape[1], 512, 256, 128, embeddingSize]

    autoencoder:FeedforwardAutoencoder = FeedforwardAutoencoder(architecture).to(device=device)

    if standardize:
        Z = _standardize(X, standardizeAxis)
    else:
        Z = X

    trainloader = get_dataloader(
                Z,
                batchSize,
                drop_last=False,
                shuffle=True,
            )
    
    testloader = get_dataloader(
        Z,
        batchSize,
        drop_last=False,
        shuffle=False,
    )

    #trainloader = trainloader.to(device)

    if os.path.exists(savePath):

        print('Loading autoencoder...')

        autoencoder = torch.load(savePath)

    else:

        print('Fitting autoencoder...')

        if not autoencoder.fitted:
                autoencoder:FeedforwardAutoencoder = get_trained_autoencoder(
                    trainloader,
                    pretrain_optimizer_params,
                    pretrain_epochs,
                    device,
                    optimizer_class,
                    loss_fn,
                    embeddingSize,
                    autoencoder,
                )

    

        torch.save(autoencoder.state_dict(), savePath)

    return autoencoder, testloader


def _load_uint8(f):
    idx_dtype, ndim = struct.unpack('BBBB', f.read(4))[2:]
    shape = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
    buffer_length = int(np.prod(shape))
    data = np.frombuffer(f.read(buffer_length), dtype=np.uint8).reshape(shape)
    return data


def load_idx(path: str) -> np.ndarray:
    """Reads an array in IDX format from disk.

    Parameters
    ----------
    path : str
        Path of the input file. Will uncompress with `gzip` if path ends in '.gz'.

    Returns
    -------
    np.ndarray
        Output array of dtype ``uint8``.

    References
    ----------
    http://yann.lecun.com/exdb/mnist/
    """
    open_fcn = gzip.open if path.endswith('.gz') else open
    with open_fcn(path, 'rb') as f:
        return _load_uint8(f)
    

@typechecked
def computeDistancesDbscan(dbscanInstance:PredictDBSCAN, X:np.ndarray, featureNames:Iterable[str]):

    yPred = dbscanInstance.fit_predict(X=X)

    nOutliers:int = len(yPred[yPred == -1])

    uniqueLabels:Iterable[int] = np.unique(yPred)

    clusterLabels:Iterable[int] = uniqueLabels[uniqueLabels >= 0]

    startClass:int = -1

    allIndexes:np.ndarray = np.arange(X.shape[0])

    startClassIndexes = allIndexes[yPred==startClass]

    resultsArrayBay:np.ndarray = np.empty((nOutliers, len(clusterLabels)))

    resultsArrayOurs:np.ndarray = np.empty((nOutliers, len(clusterLabels)))

    core_points, core_labels, outliers, outlier_mask, sortedIndices = find_outliers_and_clusters(dbscanInstance, X, yPred)    

    counterfactualsBaycon = {outlierIndex:{int(label):[] for label in clusterLabels} for outlierIndex in range(nOutliers)}

    counterfactualsOurs = {outlierIndex:{int(label):[] for label in clusterLabels} for outlierIndex in range(nOutliers)}

    originalPointsBaycon = []

    originalPointsOurs = []

    for outlierIndex in tqdm(range(nOutliers)):

        selected_outlier, selected_outlier_index = select_outlier(outliers, outlier_mask, outlierIndex)

        #print(outliers[outlierIndex])

        counterfactuals, unique_nearby_clusters = analyze_outlier(selected_outlier, core_points, core_labels, dbscanInstance)

        
        for i in range(len(counterfactuals)):

            counterfactualsOurs[outlierIndex][i] = list(counterfactuals[i])

        originalPointsOurs.append(np.array(outliers[outlierIndex]))

        #print(counterfactuals[0])

        counterfactualsDict:Dict[int, Iterable[float]] = {int(cluster):count for (cluster, count) in zip(unique_nearby_clusters, counterfactuals)}


        for clusterLabel in tqdm(clusterLabels, desc='Baycon loop'):

            targetClass:int = int(clusterLabel)

            targetClassIndexes = allIndexes[yPred==targetClass]

            t = Target(target_type="classification", target_feature="class", target_value=targetClass)

            output = executeCustom(
            target=t,
            initial_instance_index=startClassIndexes[outlierIndex],
            model=dbscanInstance, X=X, Y=yPred,
            featureNames=featureNames,
            actionable_features=None
            )

            
            
            if len(output['counterfactuals']) == 0:

                resultsArrayBay[outlierIndex, clusterLabel] = np.nan

                counterfactualsBaycon[outlierIndex][clusterLabel] = None

                

            else:
                

                counterfactuals:np.ndarray = np.array(output['counterfactuals'])

                

                

                distancesEculidean:Iterable[float] = np.sqrt(np.sum((np.array(output['counterfactuals']) - output['initial_instance'])**2, axis = 1))

                minimumDistance:float = np.min(distancesEculidean)

            

                resultsArrayBay[outlierIndex, clusterLabel] = minimumDistance

                counterfactualsBaycon[outlierIndex][clusterLabel] = list(counterfactuals[np.argmin(minimumDistance)])

            originalPointsBaycon.append(np.array(output['initial_instance']))
            

            #if clusterLabel == 0:

            #    print(counterfactualsDict[clusterLabel], output['initial_instance'])

            distanceClosestCounterfactual:float = np.sqrt(np.sum((np.array(counterfactualsDict[clusterLabel] - np.array(output['initial_instance'])) ** 2)))

            #distanceDifference:float = (minimumDistance - distanceClosestCounterfactual) / distanceClosestCounterfactual

            

            resultsArrayOurs[outlierIndex, clusterLabel] = distanceClosestCounterfactual


    return resultsArrayBay, resultsArrayOurs, counterfactualsBaycon, counterfactualsOurs, np.array(originalPointsBaycon), np.array(originalPointsOurs)


@typechecked
def computeCFBinary(dbscanInstance:PredictDBSCAN, X:np.ndarray, featureNames:Iterable[str]):

    assert dbscanInstance.binary, "We hsould have a binary 'classifier' to compute diversity"

    allIndexes:np.ndarray = np.arange(X.shape[0])

    yPred = dbscanInstance.fit_predict(X=X)

    #print('First pass', np.unique(yPred, return_counts=True))
    
    startClassIndexes = allIndexes[yPred==-1]

    nOutliers:int = len(yPred[yPred == -1])

    yPredBaycon = yPred.copy()

    yPredBaycon[yPredBaycon > 0] = 0

    core_points, core_labels, outliers, outlier_mask, sortedIndices = find_outliers_and_clusters(dbscanInstance, X, yPred)    

    for outlierIndex in tqdm(range(nOutliers)):

        selected_outlier, _ = select_outlier(outliers, outlier_mask, outlierIndex)

        #print(outliers[outlierIndex])

        counterfactualsOurs, _ = analyze_outlier(selected_outlier, core_points, core_labels, dbscanInstance)

        t = Target(target_type="classification", target_feature="class", target_value=0)

        output = executeCustom(
            target=t,
            initial_instance_index=startClassIndexes[outlierIndex],
            model=dbscanInstance, X=X, Y=yPredBaycon,
            featureNames=featureNames,
            actionable_features=None
            )
        
        counterfactualsBaycon = output['counterfactuals']

    return counterfactualsOurs, counterfactualsBaycon


@typechecked
def diversityMetric(counterfactuals) -> float:

    distanceMatrix:np.ndarray = pairwise_distances(counterfactuals)

    upperTri = distanceMatrix[np.triu_indices_from(distanceMatrix, k=1)]

    return np.mean(upperTri)


def sparsity(cf_examples, x_orig):
    """
    cf_examples: np.ndarray of shape (k, d)
    x_orig:      np.ndarray of shape (d,)
    Returns the fraction of features (between 0 and 1) that change on average,
    compared to the original instance x_orig.
    """

    k, d = np.array(cf_examples).shape

    #For each of the k CFs, count number of differing features
    diffs_per_cf = np.sum(np.array(cf_examples) != x_orig, axis=1)  # shape (k,)

    # Average over CFs, then divide by total # of features d
    # This yields a number in [0, 1], representing fraction of changed features
    sparsity_value = np.mean(diffs_per_cf) / d
    return sparsity_value


def runTimeBaycon(dbscanInstance:PredictDBSCAN, X:np.ndarray, featureNames:Iterable[str]):

    yPred = dbscanInstance.fit_predict(X=X)

    nOutliers:int = len(yPred[yPred == -1])

    for outlierIndex in tqdm(range(nOutliers)):

        t = Target(target_type="classification", target_feature="class", target_value=targetClass)




def loadDbscanParameters(parametersPath, datasetName, x):

    if os.path.exists(parametersPath):

        with open(parametersPath, 'r') as f:

            allParams:Dict[str, Tuple[float, float]] = json.load(f)

    else:

        allParams:Dict[str, Tuple[float, float]] = {}


    if datasetName in list(allParams.keys()):

        bestParams:Dict[str, Tuple[float, float]] = allParams[datasetName]

    else:

        epsValues:List[float] = [0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 2.5, 3, 3.5]

        minSamplesValues:List[float] = [3, 5, 10, 15, 20, 30]

        bestParams, _, _, _, _ = dbscan_opt(encoding=x, eps_values=epsValues, min_samples_values=minSamplesValues)

        allParams[datasetName] = bestParams

        with open(parametersPath, 'w') as f:

            json.dump(allParams, f, indent=2)

    
    return allParams


def loadResults(dfPath):

    if os.path.exists(dfPath):

        df:pd.DataFrame = pd.read_csv(
            dfPath
        )

    else:

        df:pd.DataFrame = pd.DataFrame(
            columns=['name', 'indexPoint', 'targetCluster', 'proximityDb', 'proximityBaycon', 'proximityDice', 'proximityFace','validity', 'diversity']
        )

    return df



def executeBaycon(xData, labels, model, 
                  indexStartPoint, targetClass, categoricalFeatures=None, 
                  featureNames = None, plainCounterfactuals=False, actionableFeatures=None):

    t = Target(target_type="classification", target_feature="class", target_value=targetClass)

    if featureNames is None:

        featureNames = np.arange(xData.shape[1])

    output = executeCustom(
                target=t,
                initial_instance_index=indexStartPoint,
                model=model, X=xData, Y=labels,
                featureNames=featureNames,
                actionable_features=actionableFeatures,
                categorical_features=categoricalFeatures
                )
    
    #print('OUTPUT', output)

    if plainCounterfactuals:

        return output
    
    else:

        counterfactuals = np.array(output['counterfactuals']) 

        predictions = np.array(output['predictions'])

        counterfactuals = counterfactuals[predictions == 1]


        if len(counterfactuals) > 0:

            distances = np.linalg.norm(counterfactuals - xData[indexStartPoint], axis=1)

            closest_idx = np.argmin(distances)

            closest_cf = counterfactuals[closest_idx]

            return closest_cf

        else:

            return None
        

    
def executeDice(df:pd.DataFrame, model, queryIndex, totalCfs:int = 1, targetClass = 'opposite'):

    X = df.drop(columns='target')
    
    data_dice = dice_ml.Data(dataframe=df, continuous_features=X.select_dtypes(include='number').columns.tolist(), outcome_name='target')

    model_dice = dice_ml.Model(model=model, backend="sklearn")

    explainer = Dice(data_dice, model_dice, method='random')

    cf = explainer.generate_counterfactuals(X.iloc[[queryIndex]], total_CFs=totalCfs, desired_class=int(targetClass))

    return cf


def computeDiversityGraph(graph: nx.Graph, nodes):
    """
    Compute the sum of pairwise shortest path distances between nodes.
    
    Args:
        graph: NetworkX graph
        nodes: Collection of nodes to compute distances between
    
    Returns:
        Sum of all pairwise distances, or float('inf') if any pair is disconnected
    """
    #total_distance = 0
    allDistances = []
    nodes_list = list(nodes)
    
    # Iterate over all unique pairs of nodes
    for i in range(len(nodes_list)):
        for j in range(i + 1, len(nodes_list)):
            try:
                distance = nx.shortest_path_length(graph, nodes_list[i], nodes_list[j], weight='weight')
                allDistances.append(distance)
                
            except nx.NetworkXNoPath:
                # Nodes are disconnected
                return float('inf')
    
    
    return np.mean(1/np.array(allDistances))

def computeDiversityGraphDpp(graph: nx.Graph, nodes):
    """
    Compute the sum of pairwise shortest path distances between nodes.
    
    Args:
        graph: NetworkX graph
        nodes: Collection of nodes to compute distances between
    
    Returns:
        Sum of all pairwise distances, or float('inf') if any pair is disconnected
    """
    #total_distance = 0
    
    nodes_list = list(nodes)
    finalMatrix = np.empty((len(nodes_list), len(nodes_list)))
    # Iterate over all unique pairs of nodes
    
    for i in range(len(nodes_list)):
        finalMatrix[i, i] = 1
        for j in range(i + 1, len(nodes_list)):
            try:
                distance = nx.shortest_path_length(graph, nodes_list[i], nodes_list[j], weight='weight')
                finalMatrix[i,j] = 1/(1 + distance)

                finalMatrix[j,i] = 1/(1 + distance)
                
            except nx.NetworkXNoPath:
                
                # Nodes are disconnected
                return float('inf')
    
    
    return np.linalg.det(finalMatrix)

def computeDiversityGraph_(graph: nx.Graph, nodes):
    """
    More efficient version using shortest path lengths computed in batch.
    """
    allDistances=[]
    #total_distance = 0
    
    # Compute shortest paths from each node to all others
    for source in nodes:
        try:
            lengths = nx.single_source_dijkstra_path_length(graph, source, weight='weight')
            # Sum distances to other nodes in our set (avoid double counting)
            for target in nodes:
                if target > source and target in lengths:  # lexicographic comparison to avoid duplicates
                    #total_distance += lengths[target]

                    allDistances.append(lengths[target])
        except nx.NetworkXError:
            continue
    
    return np.mean(allDistances)


@typechecked
def computeDiversity(points):

    distances = pdist(points, metric='euclidean')

    return np.mean(distances)

@typechecked
def createGraph(neighborhoods, neighborhoodsDist, clusterLabels, X, targetCluster:int|np.int64|np.int32):
    """Create weighted density connectivity graph using DBSCAN's computed neighborhoods"""

    assert targetCluster != -1, "Noise Points are not density connected"

    allDistances = []
    
    G = nx.Graph()
    
    # Get indices of points in the target cluster
    clusterPoints = [i for i in range(len(clusterLabels)) if clusterLabels[i] == targetCluster]
    
    # Add only nodes from the target cluster
    for i in clusterPoints:
        G.add_node(i, pos=X[i], cluster=clusterLabels[i])
        
    # Use the neighborhoods computed by DBSCAN
    for i in clusterPoints:
        if clusterLabels[i] == targetCluster:
            
            neighborIndices = neighborhoods[i]

            neighborDistances = neighborhoodsDist[i]
            
            for j, distance in zip(neighborIndices, neighborDistances):
                if (i != j and clusterLabels[j] == targetCluster):
                    
                    # Add edge with distance as weight (avoid duplicate edges)
                    if not G.has_edge(i, j):
                        
                        G.add_edge(i, j, weight=distance)

                        allDistances.append(distance)
    
    return G, np.mean(allDistances)

@typechecked
def addPointsToGraph(G, newPoints, epsilon: float, targetCluster):
    """
    Add new points to existing graph and connect them to existing nodes if distance < epsilon
    
    Parameters:
    - G: existing networkx graph
    - newPoints: array of new points to add (shape: n_new_points x n_features)
    - epsilon: distance threshold for creating edges
    - X_original: original points array (if None, extracts from graph node positions)
    - connectNewPoints: whether to connect new points to each other as well
    
    Returns:
    - updated graph G
    - list of new distances added
    """
    from scipy.spatial.distance import cdist
    
    newDistances = []
    
    newGraph = deepcopy(G)
    # Get original points from graph if not provided
    
    existingNodes = list(newGraph.nodes())
    X_original = np.array([newGraph.nodes[node]['pos'] for node in existingNodes])
    existingIndices = existingNodes
    
    
    # Get the next available node index
    nextNodeIndex = max(newGraph.nodes()) + 1 if newGraph.nodes() else 0
    
    # Add new points as nodes
    newNodeIndices = []
    for i, point in enumerate(newPoints):
        newNodeIndex = nextNodeIndex + i
        newNodeIndices.append(newNodeIndex)
        # Assume new points don't have a cluster label or assign them a special value
        newGraph.add_node(newNodeIndex, pos=point, cluster=targetCluster)  # -2 for new points
    
    # Connect new points to existing points if distance < epsilon
    if len(existingIndices) > 0:
        # Compute distances between new points and existing points
        distances = cdist(newPoints, X_original)
        
        for i, newNodeIdx in enumerate(newNodeIndices):
            for j, existingNodeIdx in enumerate(existingIndices):
                distance = distances[i, j]
                if distance <= epsilon + 1e-2:
                    newGraph.add_edge(newNodeIdx, existingNodeIdx, weight=distance)
                    newDistances.append(distance)
    
    
    return newGraph, newNodeIndices, newDistances

@typechecked
def addPointsToGraphExdbscan(G, newPoints, targetCluster, greedySet):
    """
    Add new points to existing graph and connect them to existing nodes if distance < epsilon
    
    Parameters:
    - G: existing networkx graph
    - newPoints: array of new points to add (shape: n_new_points x n_features)
    - epsilon: distance threshold for creating edges
    - connectNewPoints: whether to connect new points to each other as well
    
    Returns:
    - updated graph G
    - list of new distances added
    """
    from scipy.spatial.distance import cdist
    
    newDistances = []
    
    newGraph = deepcopy(G)

    # Get the next available node index
    nextNodeIndex = max(newGraph.nodes()) + 1 if newGraph.nodes() else 0
    
    # Add new points as nodes
    newNodeIndices = []
    for i, point in enumerate(newPoints):
        newNodeIndex = nextNodeIndex + i
        newNodeIndices.append(newNodeIndex)
        # Assume new points don't have a cluster label or assign them a special value
        newGraph.add_node(newNodeIndex, pos=point, cluster=targetCluster)
    
    # Connect new points to existing points if distance < epsilon
    for newNodeIndex, node in zip(newNodeIndices, greedySet):
        # Compute distances between new points and existing points
        nodePosition = newGraph.nodes[node]['pos']

        
        distance = np.linalg.norm(point-nodePosition)

        newGraph.add_edge(newNodeIndex, node, weight=distance)
    
    return newGraph, newNodeIndices, newDistances


def physicalOptimizationOld(G: nx.Graph, startVertex=None, nPoints=None, alpha=1.0):
    """
    Greedy algorithm to select particles that minimize total energy in a system with:
    - Electrostatic repulsion between ALL selected particles
    - Spring attraction between each particle and the reference point only
    - Distances computed using Dijkstra's shortest path algorithm
    
    Parameters:
    - G: NetworkX graph where nodes are particles and edge weights are distances
    - startVertex: The reference particle (spring attachment point)
    - nPoints: Maximum number of particles to select (including reference)
    - spring_constant: Spring constant for attraction to reference point
    - coulomb_constant: Coulomb constant for electrostatic repulsion
    
    Returns:
    - List of selected particle vertices
    """
    
    if startVertex is None or startVertex not in G.nodes():
        raise ValueError("Invalid start vertex")
    
    if nPoints is None or nPoints < 1:
        return [startVertex]
    
    # Initialize with the reference point
    selected = [startVertex]
    available = set(G.nodes()) - {startVertex}


    
    # Precompute shortest distances from reference vertex to all other vertices
    try:
        distances_from_ref = nx.single_source_dijkstra_path_length(G, startVertex)
    except nx.NetworkXError:
        raise ValueError("Graph contains negative edge weights or other issues")
    
    while len(selected) < nPoints and available:
        best_candidate = None
        best_energy_contribution = float('inf')
        
        for candidate in available:
            # Check if candidate is reachable from reference point
            if candidate not in distances_from_ref:
                continue  # Skip unreachable candidates
            
            distance_to_ref = distances_from_ref[candidate]
            if distance_to_ref <= 0:
                continue  # Skip invalid distances
            
            # Calculate energy contribution if we add this candidate
            total_energy_contribution = 0
            
            
            # 1. Spring energy: Only between candidate and reference point
            #spring_energy = 0.5 * spring_constant * distance_to_ref**2
            spring_energy = alpha * distance_to_ref**2
            total_energy_contribution += spring_energy
            
            # 2. Electrostatic repulsion: Between candidate and ALL previously selected particles
            valid_candidate = True
            for selected_particle in selected:
                try:
                    # Compute shortest path distance using Dijkstra
                    distance = nx.shortest_path_length(G, candidate, selected_particle, weight='weight')
                    
                    if distance <= 0:
                        valid_candidate = False
                        break
                    
                    # Electrostatic repulsion energy (positive)
                    #electrostatic_energy = 1 / distance
                    electrostatic_energy = (1-alpha) / distance
                    total_energy_contribution += electrostatic_energy
                    
                except nx.NetworkXNoPath:
                    # If no path exists, treat as infinite repulsion (skip this candidate)
                    valid_candidate = False
                    break
                except nx.NetworkXError:
                    valid_candidate = False
                    break
            
            if not valid_candidate:
                continue
            
            
            # Select particle that minimizes total energy contribution
            if total_energy_contribution < best_energy_contribution:
                best_energy_contribution = total_energy_contribution
                best_candidate = candidate
        
        # Add the best candidate if found
        if best_candidate is not None:
            selected.append(best_candidate)
            available.remove(best_candidate)
        else:
            # No more valid candidates
            break
    
    return selected


def physicalOptimization(G: nx.Graph, pointToExplain, carachteristicDistance, singleCounterfactual, 
                         startVertex=None, nPoints=None, alpha=0.5, normalization=False):
    """
    Greedy algorithm to select particles that minimize total energy in a system with:
    - Electrostatic repulsion between ALL selected particles
    - Spring attraction between each particle and the reference point only
    - Distances computed using Dijkstra's shortest path algorithm
    
    Parameters:
    - G: NetworkX graph where nodes are particles and edge weights are distances
    - startVertex: The reference particle (spring attachment point)
    - nPoints: Maximum number of particles to select (including reference)
    - spring_constant: Spring constant for attraction to reference point
    - coulomb_constant: Coulomb constant for electrostatic repulsion
    
    Returns:
    - List of selected particle vertices
    """

    
    if startVertex is None or startVertex not in G.nodes():
        raise ValueError("Invalid start vertex")
    
    if nPoints is None or nPoints < 1:
        return [startVertex]
    
    # Initialize with the reference point
    selected = [startVertex]
    available = set(G.nodes()) - {startVertex}


    
    # Precompute shortest distances from reference vertex to all other vertices
    try:
        distances_from_ref = nx.single_source_dijkstra_path_length(G, startVertex)
    except nx.NetworkXError:
        raise ValueError("Graph contains negative edge weights or other issues")
    
    scaleDistance = np.linalg.norm(pointToExplain - singleCounterfactual)
    
    while len(selected) < nPoints and available:
        best_candidate = None
        best_energy_contribution = float('inf')

        

        for candidate in available:
            
            # Check if candidate is reachable from reference point
            if candidate not in distances_from_ref:
                #continue  # Skip unreachable candidates
                pass
            
            #distance_to_ref = distances_from_ref[candidate]
            distance_to_ref = np.linalg.norm(pointToExplain - G.nodes[candidate]['pos'])
            if distance_to_ref <= 0:
                continue  # Skip invalid distances
            
            # Calculate energy contribution if we add this candidate
            #total_energy_contribution = 0
            totalSpringContribution = 0

            totalElectrostaticContribution = 0

            allDistancesElec = []
            
            # 1. Spring energy: Only between candidate and reference point
            #spring_energy = 0.5 * spring_constant * distance_to_ref**2
            spring_energy =  distance_to_ref**2
            totalSpringContribution += spring_energy
            
            # 2. Electrostatic repulsion: Between candidate and ALL previously selected particles
            valid_candidate = True
            for selected_particle in selected:
                try:
                    # Compute shortest path distance using Dijkstra
                    distance = nx.shortest_path_length(G, candidate, selected_particle, weight='weight')
                    
                    if distance <= 0:
                        valid_candidate = False
                        break
                    
                    # Electrostatic repulsion energy (positive)
                    #electrostatic_energy = 1 / distance
                    electrostatic_energy = (1 / distance)
                    allDistancesElec.append(distance)
                    if normalization:
                        totalElectrostaticContribution +=  (scaleDistance/carachteristicDistance) * electrostatic_energy#
                    else:
                        totalElectrostaticContribution += electrostatic_energy
                    
                except nx.NetworkXNoPath:
                    # If no path exists, treat as infinite repulsion (skip this candidate)
                    valid_candidate = False
                    break
                except nx.NetworkXError:
                    valid_candidate = False
                    break
            
            if not valid_candidate:
                continue
            
            total_energy_contribution =  alpha * totalSpringContribution + (1-alpha) * (totalElectrostaticContribution)
            # Select particle that minimizes total energy contribution
            if total_energy_contribution < best_energy_contribution:
                best_energy_contribution = total_energy_contribution
                best_candidate = candidate
        
        # Add the best candidate if found
        if best_candidate is not None:
            selected.append(best_candidate)
            available.remove(best_candidate)
        else:
            # No more valid candidates
            break
    
    return selected




def greedyPermutation(G: nx.Graph, startVertex=None, nPoints=None, alpha=0.5, normalize=True):
    """
    Greedy permutation (farthest-first) with optional proximity-to-reference tradeoff.

    Args:
        G: NetworkX graph (with 'weight' on edges)
        startVertex: reference node (n1)
        nPoints: number of nodes to select
        lam: tradeoff parameter between diversity and proximity
             - lam = 0: pure diversity (standard farthest-first)
             - lam > 0: balances closeness to startVertex
        normalize: if True, normalize both diversity and proximity terms to [0,1]

    Returns:
        permutation: list of selected vertices
    """
    

    if len(G) == 0:
        return []

    vertices = list(G.nodes())
    n = len(G.nodes()) if nPoints is None else nPoints

    # Initialize distances to infinity
    distances = {v: float('inf') for v in vertices}

    # Pick starting vertex
    if startVertex not in G.nodes():
        startVertex = vertices[0]

    permutation = [startVertex]

    # Precompute distances from reference (for proximity term)
    dist_from_ref = {v: float('inf') for v in vertices}
    _modifiedDijkstra(G, startVertex, dist_from_ref)

    # Initial Dijkstra update from startVertex
    distances[startVertex] = 0
    _modifiedDijkstra(G, startVertex, distances)

    # Normalization constants
    max_ref = np.median([d for d in dist_from_ref.values() if d < float('inf')])
    if normalize and max_ref == 0:  # avoid division by zero
        max_ref = 1.0

    for i in range(1, n):
        def score(v):
            diversity = distances[v]
            proximity = dist_from_ref[v]

            if normalize:
                # Normalize to [0,1]
                div_norm = diversity / np.median(list(distances.values()))
                prox_norm = proximity / max_ref
                return alpha * div_norm + (1-alpha) / (prox_norm**2)
            else:
                return diversity - alpha * proximity

        # Pick the vertex maximizing the score
        next_vertex = max((v for v in vertices if v not in permutation),
                          key=score)

        permutation.append(next_vertex)

        # Update distances via Dijkstra from the new vertex
        if i < n - 1:
            distances[next_vertex] = 0
            _modifiedDijkstra(G, next_vertex, distances)

    return permutation


def _modifiedDijkstra(G: nx.Graph, start: int, distances: Dict) -> None:
    """
    Modified Dijkstra's algorithm as described in Figure 2.1
    
    Args:
        G: NetworkX graph
        start: Starting vertex (πi in the paper)
        distances: Dictionary of current distances (lv in the paper)
                  Will be modified in-place
    """
    # Priority queue: (distance, vertex)
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current in visited:
            continue
            
        visited.add(current)
        
        # Update distances to neighbors
        for neighbor in G.neighbors(current):
            if neighbor in visited:
                continue
                
            # Get edge weight
            edge_weight = G[current][neighbor].get('weight', 1)
            new_distance = current_dist + edge_weight
            
            # Only update if we found a shorter path
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(pq, (new_distance, neighbor))

def moveToEpsilonDistanceVectorised(start_point, displacement, epsilon):

    displacement_norm = np.linalg.norm(displacement)

    unitVectorDisplacement = displacement / displacement_norm

    diff = -displacement

    A = np.sum(unitVectorDisplacement * unitVectorDisplacement, axis=1)
    B = 2 * np.sum(diff * unitVectorDisplacement, axis=1)
    C = np.sum(diff * diff, axis=1) - epsilon**2

    discriminant = B**2 - 4*A*C

    sqrt_discriminant = np.sqrt(discriminant)
    t1 = (-B - sqrt_discriminant) / (2*A)
    t2 = (-B + sqrt_discriminant) / (2*A)

    allResults = np.empty((len(t1), displacement.shape[1]))

    for i, (sol1, sol2) in enumerate(zip(t1, t2)):

        if np.abs(sol1) < np.abs(sol2):

            sol = sol1

        else:

            sol = sol2

        final_position = start_point + sol * unitVectorDisplacement[i]

        allResults[i] = final_position


    return allResults


def moveToEpsilonDistance(start_point, direction, target_point, epsilon):
    """
    Move a point along a direction until it's epsilon distance from target_point.
    Works in any number of dimensions.
    
    Args:
        start_point: array-like - starting position [x1, x2, ..., xn]
        direction: array-like - direction vector [dx1, dx2, ..., dxn]
        target_point: array-like - point to stay epsilon away from
        epsilon: float - desired distance from target_point
    
    Returns:
        numpy array - final position, or None if no valid solution
    """
    start = np.array(start_point, dtype=float)
    dir_vec = np.array(direction, dtype=float)
    target = np.array(target_point, dtype=float)
    
    # Vector from target to start
    diff = start - target
    
    # Quadratic equation coefficients: At² + Bt + C = 0
    A = np.dot(dir_vec, dir_vec)
    B = 2 * np.dot(diff, dir_vec)
    C = np.dot(diff, diff) - epsilon**2
    
    # Handle case where direction vector is zero
    if A == 0:
        current_distance = np.linalg.norm(diff)
        if abs(current_distance - epsilon) < 1e-10:
            return start
        else:
            return None
    
    # Calculate discriminant
    discriminant = B**2 - 4*A*C
    
    # No real solutions - the ray doesn't intersect the hypersphere
    if discriminant < 0:
        return None
    
    # Calculate the two solutions
    sqrt_discriminant = np.sqrt(discriminant)
    t1 = (-B - sqrt_discriminant) / (2*A)
    t2 = (-B + sqrt_discriminant) / (2*A)
    
    # Choose the appropriate solution (first positive intersection)
    valid_t = None
    if np.abs(t1) < np.abs(t2):

        valid_t = t1
    
    else:

        valid_t = t2
    
    # Calculate final position
    final_position = start + valid_t * dir_vec
    return final_position


def createGraphActionable(neighborhoods, neighborhoodsDist, clusterLabels, X, targetCluster: int, 
                         nonActionableFeatures: List[int], pointToExplain: np.ndarray, eps: float):
    """
    Create weighted density connectivity graph considering only actionable features
    """
    assert targetCluster != -1, "Noise Points are not density connected"

    # Get actionable features mask
    actionableFeatures = np.arange(X.shape[1])[~np.isin(np.arange(X.shape[1]), nonActionableFeatures)]
    
    # Filter point to explain for non-actionable features
    pointToExplainFiltered = pointToExplain[nonActionableFeatures].reshape(1, -1)
    
    allDistances = []
    G = nx.Graph()
    
    # Get indices of points in the target cluster
    clusterPoints = [i for i in range(len(clusterLabels)) if clusterLabels[i] == targetCluster]
    
    # Filter cluster points based on non-actionable feature constraints
    validClusterPoints = []
    for i in clusterPoints:
        # Check if point is within eps distance in non-actionable space
        pointFiltered = X[i, nonActionableFeatures].reshape(1, -1)
        distance_filtered = np.linalg.norm(pointFiltered - pointToExplainFiltered)
        
        if distance_filtered <= eps:
            validClusterPoints.append(i)
            G.add_node(i, pos=X[i], cluster=clusterLabels[i])
    
    # Create edges only between valid points
    for i in validClusterPoints:
        neighborIndices = neighborhoods[i]
        neighborDistances = neighborhoodsDist[i]
        
        for j, distance in zip(neighborIndices, neighborDistances):
            if (i != j and j in validClusterPoints and clusterLabels[j] == targetCluster):
                # Add edge with distance as weight (avoid duplicate edges)
                if not G.has_edge(i, j):
                    G.add_edge(i, j, weight=distance)
                    allDistances.append(distance)
    
    # Calculate characteristic distance in actionable space only
    if len(validClusterPoints) > 1:
        actionablePoints = X[validClusterPoints][:, actionableFeatures]
        characteristicDistance = np.mean(np.linalg.norm(
            actionablePoints[:, None, :] - actionablePoints[None, :, :], axis=2
        ))
    else:
        characteristicDistance = np.mean(allDistances) if allDistances else 1.0
    
    return G, characteristicDistance


def physicalOptimizationActionable(G: nx.Graph, pointToExplain, carachteristicDistance, singleCounterfactual, 
                                  startVertex=None, nPoints=None, alpha=0.5, normalization=False,
                                  nonActionableFeatures: List[int] = None):
    """
    Physical optimization that considers only actionable features for distance calculations
    """
    if startVertex is None or startVertex not in G.nodes():
        raise ValueError("Invalid start vertex")
    
    if nPoints is None or nPoints < 1:
        return [startVertex]
    
    # Initialize with the reference point
    selected = [startVertex]
    available = set(G.nodes()) - {startVertex}
    
    # Precompute shortest distances from reference vertex to all other vertices
    try:
        distances_from_ref = nx.single_source_dijkstra_path_length(G, startVertex)
    except nx.NetworkXError:
        raise ValueError("Graph contains negative edge weights or other issues")
    
    # Calculate scale distance considering only actionable features
    if nonActionableFeatures is not None:
        actionableFeatures = np.arange(len(pointToExplain))[~np.isin(np.arange(len(pointToExplain)), nonActionableFeatures)]
        scaleDistance = np.linalg.norm((pointToExplain - singleCounterfactual)[actionableFeatures])
    else:
        scaleDistance = np.linalg.norm(pointToExplain - singleCounterfactual)
    
    while len(selected) < nPoints and available:
        best_candidate = None
        best_energy_contribution = float('inf')
        
        for candidate in available:
            # Calculate distance in actionable space only
            if nonActionableFeatures is not None:
                candidate_pos = G.nodes[candidate]['pos'][actionableFeatures]
                point_pos = pointToExplain[actionableFeatures]
                distance_to_ref = np.linalg.norm(candidate_pos - point_pos)
            else:
                distance_to_ref = np.linalg.norm(pointToExplain - G.nodes[candidate]['pos'])
            
            if distance_to_ref <= 0:
                continue  # Skip invalid distances
            
            # Calculate energy contribution
            totalSpringContribution = distance_to_ref**2
            totalElectrostaticContribution = 0
            
            # Electrostatic repulsion between candidate and all previously selected particles
            valid_candidate = True
            for selected_particle in selected:
                try:
                    # Compute shortest path distance using Dijkstra
                    distance = nx.shortest_path_length(G, candidate, selected_particle, weight='weight')
                    
                    if distance <= 0:
                        valid_candidate = False
                        break
                    
                    electrostatic_energy = (1 / distance)
                    if normalization:
                        totalElectrostaticContribution += (scaleDistance/carachteristicDistance) * electrostatic_energy
                    else:
                        totalElectrostaticContribution += electrostatic_energy
                    
                except nx.NetworkXNoPath:
                    valid_candidate = False
                    break
                except nx.NetworkXError:
                    valid_candidate = False
                    break
            
            if not valid_candidate:
                continue
            
            total_energy_contribution = alpha * totalSpringContribution + (1-alpha) * totalElectrostaticContribution
            
            # Select particle that minimizes total energy contribution
            if total_energy_contribution < best_energy_contribution:
                best_energy_contribution = total_energy_contribution
                best_candidate = candidate
        
        # Add the best candidate if found
        if best_candidate is not None:
            selected.append(best_candidate)
            available.remove(best_candidate)
        else:
            # No more valid candidates
            break
    
    return selected