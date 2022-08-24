import faiss
import numpy as np
from cuml import HDBSCAN as gpu_hdbscan
from hdbscan.hdbscan_ import HDBSCAN as cpu_hdbscan

from typing import Dict, Tuple


class HDBSCAN(gpu_hdbscan):
    """
    Custom HDBSCAN class to bridge cuML's GPU implementation of HDBSCAN
    and the scikit-contrib CPU version.

    Parameters
    ----------
    Matches cuML's HDBSCAN implementation
    https://docs.rapids.ai/api/cuml/stable/api.html#cuml.cluster.HDBSCAN

    Notes
    -----
    As of RAPIDS 21.10, cuML's HDBSCAN does not implement the
    approximate_predict function for transforming new data.
    So we supercede cuML's fit() function with a custom version
    """

    def __init__(self, *args, **kwargs):
        # Initialise gpu_hdbscan
        super().__init__(*args, **kwargs)

    def fit(self, X: np.ndarray, prediction_data: bool = True) -> None:
        """
        Fit model and generate prediction data

        Parameters
        ----------
        X : np.ndarray
            Input vectors
        prediction_data : bool
            Generate the data needed to make transformations
        """
        # Run gpu_hdbscan's fit method
        super().fit(X, y=None)

        if prediction_data is not True:
            return

        # cuML's implementation is missing some key variables, re-add here
        self.condensed_tree_.cluster_selection_method = \
            self.cluster_selection_method
        self._raw_data = X
        self._metric_kwargs = {}

        # Generate prediction data using cpu HDBSCAN's function
        cpu_hdbscan.generate_prediction_data(self)

        # Strange naming mismatch
        self.prediction_data_ = self._prediction_data

        # Build tree index
        self.tree_row_indices = self.build_condensed_tree_idx(
            self.condensed_tree_._raw_tree,
            X.shape[0])

        self.cluster_tree_row_indices = \
            self.build_cluster_tree_idx(
                self.prediction_data_.cluster_tree,
            )

        self.build_faiss_index()

        self.prediction_data_.core_distances = \
            np.array(self.prediction_data_.core_distances)

    def transform(self, points: np.ndarray) -> np.ndarray:
        """
        Predict clusters for new datapoints

        Parameters
        ----------
        points : np.ndarray
            Input vectors

        Returns
        -------
        labels : np.ndarray
            Predicted cluster for each vector

        Notes
        -----
        Must have trained with predicton_data set to True
        """

        if self.prediction_data_ is None:
            raise Exception('Prediction data not generated')

        labels = self.approximate_predict(points)

        return labels

    @staticmethod
    def build_condensed_tree_idx(tree: np.recarray,
                                 n: int) -> np.ndarray:
        """
        Building an index of the condensed tree

        Parameters
        ----------
        tree : numpy.recarray
            Tree to generate index for
        n : int
            Number of datapoints in set, accounts for n not being part of the tree

        Returns
        -------
        tree_row_indices : np.ndarray
            Array with the index being each datapoint,
            values being the location in the tree

        Notes
        -----
        The default prediction will traverse the tree every time to find the
        node pertaining to a given datapoint. So if you trained on 500k
        datapoints, it will search the 500k+clusters nodes until it finds
        the node it's looking for. This builds an index to avoid that, so
        to find the node for datapoint #123, index[123] will give the
        location in the tree directly.
        """
        tree_children = tree['child']
        # N is not in the tree, which can mess with the index
        tree_children = np.append(tree_children, n)

        # Initialise index to cover the whole tree, +1 for n
        array_range = np.array(range(tree_children.max() + 1))

        tree_row_indices = np.empty(len(tree_children) + 1, int)
        tree_row_indices[tree_children] = array_range

        return tree_row_indices

    @staticmethod
    def build_cluster_tree_idx(tree: np.recarray) -> Dict[int, int]:
        """
        Building an index of the cluster tree

        Parameters
        ----------
        tree : numpy.recarray
            Tree to build the index for

        Returns
        -------
        output_dict : Dict[int, int]
            Dictionary with cluster : index_in_tree format

        Notes
        -----
        Same idea as build_condensed_tree_idx, but for the cluster tree
        instead (nodes+clusters in condensed tree, just clusters in
        cluster_tree). Uses a dictionary instead as the scale is lower
        and the tree is not as extensive as the condensed one.
        """
        inverted_dict = dict(enumerate(tree['child']))

        output_dict = {cluster: idx for idx, cluster in inverted_dict.items()}

        return output_dict

    def approximate_predict(self,
                            points_to_predict: np.ndarray,
                            prediction_neighbours: int = 5) -> np.ndarray:
        """
        Function for predicting new data with a trained HDBSCAN model

        Parameters
        ----------
        points_to_predict : np.ndarray
            Vectors to predict on
        prediction_neighbours : int
            How many neighbours to use for prediction

        Returns
        -------
        labels : np.ndarray
            Array of labels denoting the cluster of each point

        Notes
        -----
        Based heavily on scikit-contrib's version

        Main changes are:
            Use FAISS index search instead of KDTree querying
            Use saved index of tree nodes to speed up tree querying
            Remove unnecessary computations
        """
        points_to_predict = np.asarray(points_to_predict).astype(np.float32)

        labels = np.empty(points_to_predict.shape[0], dtype=np.int32)

        # Calculate nearest neighbours for each point
        neighbour_distances, neighbour_indices = self.faiss_index.search(
            points_to_predict,
            k=prediction_neighbours)

        # Find cluster for each point
        for i in range(points_to_predict.shape[0]):
            label = self.find_cluster(
                neighbour_indices[i],
                neighbour_distances[i]
            )
            labels[i] = label

        return labels

    def find_cluster(self,
                     neighbour_indices: np.ndarray,
                     neighbour_distances: np.ndarray) -> int:
        """
        Find the cluster of a point, given information about the
        points closest to it in the trained space

        Parameters
        ----------
        neighbour_indices: np.ndarray
            Indices of neighbours (in the training data)
        neighbour_distances: np.ndarray
            Distances to those neighbours

        Returns
        -------
        cluster_label: int
            Label of the cluster/topic to attribute it to

        """
        tree_root = self.prediction_data_.cluster_tree['parent'].min()

        nearest_neighbour, lambda_ = self._find_neighbour_and_lambda(
            neighbour_indices=neighbour_indices,
            neighbour_distances=neighbour_distances
        )

        neighbour_tree_row = self.condensed_tree_._raw_tree[
            self.tree_row_indices[nearest_neighbour]]

        # Start with neighbour
        potential_cluster = neighbour_tree_row['parent']

        # Go down tree to find best cluster fit
        if neighbour_tree_row['lambda_val'] > lambda_:
            # Find appropriate cluster based on lambda of new point
            while (potential_cluster > tree_root
                   and self.prediction_data_.cluster_tree['lambda_val'][self.cluster_tree_row_indices[potential_cluster]] >= lambda_):  # noqa: E501

                potential_cluster = \
                    self.prediction_data_.cluster_tree['parent'][
                        self.cluster_tree_row_indices[potential_cluster]]

        if potential_cluster in self.prediction_data_.cluster_map:
            cluster_label = \
                self.prediction_data_.cluster_map[potential_cluster]
        else:
            cluster_label = -1

        return cluster_label

    def _find_neighbour_and_lambda(self,
                                   neighbour_indices: np.ndarray,
                                   neighbour_distances: np.ndarray) -> Tuple[int, float]:
        """
        Find the nearest mutual reachability neighbour of a point, and compute
        the associated lambda value for the point, given the mutual reachability
        distance to a nearest neighbour.

        Parameters
        ----------
        neighbour_indices : array
            An array of raw distance based nearest neighbour indices.
        neighbour_distances : array
            An array of raw distances to the nearest neighbours.

        Returns
        -------
        neighbour: int
            The index into the full raw data set of the nearest mutual
            reachability distance neighbour of the point.
        lambda_: float
            The lambda value at which this point joins with neighbour
        """
        neighbour_core_distances = \
            self.prediction_data_.core_distances[neighbour_indices]

        mr_distances = np.vstack((
            neighbour_core_distances,
            neighbour_distances
        )).max(axis=0)

        nn_index = mr_distances.argmin()

        nearest_neighbour = neighbour_indices[nn_index]

        lambda_ = 1. / mr_distances[nn_index] \
            if mr_distances[nn_index] > 0.0 else np.finfo(np.double).max

        return nearest_neighbour, lambda_

    def build_faiss_index(self):
        """
        Set up FAISS index for faster KNN search
        """
        self.faiss_index = faiss.GpuIndexFlatL2(
            faiss.StandardGpuResources(),
            self._raw_data.shape[1]
        )

        self.faiss_index.add(self._raw_data.astype(np.float32))
