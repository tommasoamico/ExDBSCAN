from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.utils.validation import validate_data, _check_sample_weight
import warnings
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster._dbscan_inner import dbscan_inner


class PredictDBSCAN(DBSCAN):

    def __init__(self, eps = 0.5, *, min_samples = 5,
                  metric = "euclidean", metric_params = None, 
                  algorithm = "auto", leaf_size = 30, p = None,
                  n_jobs = None, binary=False):
        
        super().__init__(eps, min_samples=min_samples, metric=metric, metric_params=metric_params, algorithm=algorithm, leaf_size=leaf_size, p=p, n_jobs=n_jobs)

        self.binary:bool = binary



    def predict(self, X):

        if not isinstance(X, np.ndarray):

            X = np.array(X)
        
        #assert hasattr(self, 'components_'), 'The instance is not yet fitted'

        nr_samples = X.shape[0]

        y_new = np.ones(shape=nr_samples, dtype=int) * -1

        for i in range(nr_samples):
            diff = self.components_ - X[i, :]  # NumPy broadcasting

            dist = np.linalg.norm(diff, axis=1)  # Euclidean distance

            shortest_dist_idx = np.argmin(dist)

            if dist[shortest_dist_idx] < self.eps:

                if self.binary:
                    y_new[i] = 0

                else:
                    y_new[i] = self.labels_[self.core_sample_indices_[shortest_dist_idx]]

        return y_new
    

    def predict_proba(self, X):

        y_new = self.predict(X)

        allClasses = np.unique(self.labels_)

        nSamples = len(y_new)

        finalArray = np.zeros((nSamples, len(allClasses)))

        for i in range(nSamples):

            finalArray[i, np.sum(y_new[i] < allClasses)] = 1

        return finalArray
    
    def fitCustom(self, X, y=None, sample_weight=None):
        """Perform DBSCAN clustering from features, or distance matrix.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), or \
            (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``metric='precomputed'``. If a sparse matrix is provided, it will
            be converted into a sparse ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with a
            negative weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.

        Returns
        -------
        self : object
            Returns a fitted instance of self.
        """
        X = validate_data(self, X, accept_sparse="csr")

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        # Calculate neighborhood for all samples. This leaves the original
        # point in, which needs to be considered later (i.e. point i is in the
        # neighborhood of point i. While True, its useless information)
        if self.metric == "precomputed" and sparse.issparse(X):
            # set the diagonal to explicit values, as a point is its own
            # neighbor
            X = X.copy()  # copy to avoid in-place modification
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", sparse.SparseEfficiencyWarning)
                X.setdiag(X.diagonal())

        neighbors_model = NearestNeighbors(
            radius=self.eps,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            metric_params=self.metric_params,
            p=self.p,
            n_jobs=self.n_jobs,
        )
        neighbors_model.fit(X)
        # This has worst case O(n^2) memory complexity
        neighborhoods_dist, neighborhoods = neighbors_model.radius_neighbors(X, return_distance=True)

        if sample_weight is None:
            n_neighbors = np.array([len(neighbors) for neighbors in neighborhoods])
        else:
            n_neighbors = np.array(
                [np.sum(sample_weight[neighbors]) for neighbors in neighborhoods]
            )

        # Initially, all samples are noise.
        labels = np.full(X.shape[0], -1, dtype=np.intp)

        # A list of all core samples found.
        core_samples = np.asarray(n_neighbors >= self.min_samples, dtype=np.uint8)
        dbscan_inner(core_samples, neighborhoods, labels)

        self.core_sample_indices_ = np.where(core_samples)[0]
        self.labels_ = labels

        if len(self.core_sample_indices_):
            # fix for scipy sparse indexing issue
            self.components_ = X[self.core_sample_indices_].copy()
        else:
            # no core samples
            self.components_ = np.empty((0, X.shape[1]))
        return self, neighborhoods, neighborhoods_dist


    

