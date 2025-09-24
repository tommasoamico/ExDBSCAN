from typing import Union, List, Optional
from typeguard import typechecked
import numpy as np
import warnings
from sklearn.neighbors import NearestNeighbors
from LOFcounterfactuals.modules.gowerNN import GowerNearestNeighbors
import gower
import pandas as pd


def gower_distance(point1, point2, num_max, num_ranges):
    """
    Compute the Gower distance between two points using the gower package.

    Parameters:
    - point1: array-like or pandas Series (n,)
    - point2: array-like or pandas Series (n,)

    Returns:
    - float: Gower distance
    """
    #print(type(point1), type(point2))
    df = pd.concat([point1, point2])
    return gower.gower_matrix_custom(
        df, num_max=num_max, num_ranges=num_ranges
        )[0, 1]


class customLOF:

    @typechecked
    def __init__(self, minPts:Union[int, np.int64, np.int32]=20, threshold:Union[int, float] = 1.5,
                 metric='minkowski') -> None:
        
        self.minPts:int = minPts

        self.threshold:Union[int, float] = threshold
        
        self.offset:Optional[float] = None

        self.isFitted:bool = False

        self.metric=metric





    @typechecked
    def __getOffset(self, scores:np.ndarray) -> float:

        if self.threshold > 1:

            return self.threshold
        
        else:

            offset:float = np.percentile(scores, (1-self.threshold)*100)

        return offset


    
    @typechecked
    def fit(self, X:Union[np.ndarray, pd.DataFrame]) -> np.ndarray:

        '''if self.metric.lower() == 'gower':

            knn:GowerNearestNeighbors = GowerNearestNeighbors(
                n_neighbors=self.minPts + 1,
                num_max=self.num_max,
                num_ranges=self.num_ranges)

            knn.fit(X=X)

        else:'''


        
        knn:NearestNeighbors = NearestNeighbors(n_neighbors=self.minPts + 1,
                                            metric=self.metric)

        knn.fit(X=X)

        #print('X', X)

        self.knnDistances, self.knnNeighbours = knn.kneighbors(X)
        
        
        #print('DATA', X)

        #print('Neighbors Indeces', self.knnNeighbours[-1][1:])

        self.knnDistances = self.knnDistances[:, 1:]
        
        self.knnNeighbours = self.knnNeighbours[:, 1:]

        #print('DISTANCES HERE', self.knnDistances[193], X.iloc[[193]], X.iloc[self.knnNeighbours[193]],
         #     gower_distance(X.iloc[[193]], X.iloc[[48]]))

        #dist_k = self.knnDistances[neighbors_indices, self.minPts - 1]

        #print('POINTS NOW', X[self.knnNeighbours][self.knnNeighbours, self.minPts - 1])
        
        lrd = self._local_reachability_density(self.knnDistances, self.knnNeighbours)

        lrd_ratios_array = (
            lrd[self.knnNeighbours] / lrd[:, np.newaxis]
        )
        #print('Indeces', self.knnNeighbours[-1])
        #print('knn point', X[158], np.linalg.norm(X[158] - X[-1]))

        #print('knnPoint', self.metric(X[569], X[99]), X[569], X[99])

        
        #print('THIS POINT', np.linalg.norm(X[347]-X[344]), np.linalg.norm(X[345]-X[344]))
        #print('MEAN', lrd[self.knnNeighbours[217]])

        #print('DISTANCES', self.knnNeighbours[self.knnNeighbours[-1]][-7])#, self.knnNeighbours[-1])

        #print('DISTANCES', self.knnDistances[self.knnNeighbours[-1]][-3], self.knnNeighbours[-3])

        #print(X)

        #print('Single POINT', X[108])
        #print(lrd)

        #print('NUMERATOR', lrd[self.knnNeighbours][8], self.knnNeighbours[8])

        #print('denominator', lrd[:, np.newaxis][8], lrd[:, np.newaxis][6])

        #print('NUMERATOR', lrd[self.knnNeighbours][6], self.knnNeighbours[6])

        #print('denominator', lrd[:, np.newaxis][6], lrd[:, np.newaxis][8])

        #print('NUMERATOR', lrd_ratios_array, axis = 1)[6]

        lofs = np.mean(lrd_ratios_array, axis = 1)

        self.isFitted:bool = True

        self.offset:float = self.__getOffset(scores=np.array(lofs))

        #print('LOF', lofs[-1])

        return np.array(lofs)
    

    ''' @typechecked
        def predict(self, scores:np.ndarray) -> np.ndarray:

            predictVector:np.ndarray = np.ones(scores.shape[0])

            predictVector[scores > self.offset] = -1

            return predictVector'''

    def predict(self, X):
        
        lofs = self.fit(X)

        return lofs[lofs>self.threshold].astype(int)







        

    def _local_reachability_density(self, distances_X, neighbors_indices):
        """The local reachability density (LRD)

        The LRD of a sample is the inverse of the average reachability
        distance of its k-nearest neighbors.

        Parameters
        ----------
        distances_X : ndarray of shape (n_queries, self.n_neighbors)
            Distances to the neighbors (in the training samples `self._fit_X`)
            of each query point to compute the LRD.

        neighbors_indices : ndarray of shape (n_queries, self.n_neighbors)
            Neighbors indices (of each query point) among training samples
            self._fit_X.

        Returns
        -------
        local_reachability_density : ndarray of shape (n_queries,)
            The local reachability density of each sample.
        """
        

        dist_k = self.knnDistances[neighbors_indices, self.minPts - 1]

        #print('neighbours', dist_k[-1])

        #print('points', self)

        #print(dist_k[208])

        #print(self.knnDistances[208])

        #print('LRDS', distances_X[6], dist_k[6], np.maximum(distances_X, dist_k)[6])

        #print('LRDS', distances_X[8], dist_k[8], np.maximum(distances_X, dist_k)[8])

        
        #reach_dist_array = np.maximum(distances_X, dist_k)

        reach_dist_array = np.maximum(distances_X, dist_k)

        #print('Distances k', dist_k[neighbors_indices[-1]][-2], neighbors_indices[-1][-2])

        #print('Indeces', self.knnNeighbours[neighbors_indices, self.minPts - 1][neighbors_indices[-1]][-1])

        
        #print('INDECES RNN', neighbors_indices[-1])
        #reach_dist_array = distances_X

        # 1e-10 to avoid `nan' when nb of duplicates > n_neighbors_:
        return 1.0 / (np.mean(reach_dist_array, axis=1) + 1e-10)