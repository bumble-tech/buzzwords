[<< Back](index)

# Custom Model: What did we Change?

While the base algorithm is the same, we had to make a significant amount of changes to speed up the process for our use cases. Once you work on a bigger scale, the limitations of the base libraries become apparent so this was necessary

## Sentence Embedding

No change, runs using Pytorch so is automatically run on GPU where available

## UMAP

Instead of using the standard UMAP-learn library, we use cuML's UMAP implementation. cuML is a part of Nvidia's [RAPIDS.ai](https://rapids.ai/about.html) library that includes UMAP as standard. The version has been implemented for a while now and is well supported. As such, we can simply plug in cuML's version where the cpu version was previously. 

## HDBSCAN

As of October 2021 (Update Apr 2022 - Not even the nightly build of cuML has the predict feature yet, so we're still using this for the near future), cuML began to offer HDBSCAN, though the recency means this is only supported to a limited extent. The fit() function is implemented, meaning we can use cuML's HDBSCAN in Buzzword's fit() function. However, the approximate_predict() function is not yet implemented. As a workaround, we must use the base HDBscan's approximate_predict() function in conjunction with cuML's HDBSCAN module. A lot of work had to be done to speed this up and fit it in:

* A custom HDBSCAN class was created with cuML's (gpu_hdbscan) as a base class so as to avoid changing too much
* Within this, the fit() function is superseded by a custom version which first uses gpu_hdbscan.fit(), then runs a few necessary steps to allow for predictions
	* self.condensed_tree_.cluster_selection_method is missing in gpu_hdbscan's condensed_tree, set it
	* self.\_raw_data and self.\_metric_kwargs are missing from the base class, set them too
	* Run cpu_hdbscan's generate_prediction_data to build the necessary data to speed up predictions
	* self.prediction_data\_ is referred to as self.\_prediction_data in gpu_hdbscan, assign the latter to the former
	* Generates indices for the condensed tree and the cluster tree†
	* Set up FAISS index with the training data‡
* The code from cpu_hdbscan's [approximate_predict](https://github.com/scikit-learn-contrib/hdbscan/blob/54da636c2bf4ffcde0c99ae6f8eabc131f1999d2/hdbscan/prediction.py#L331) is cleaned up and used with some changes:
	* Separate the k in the knn search from min_cluster_size to allow for faster lookups: The base method uses a KDTree query to check the k nearest-neighbours to the prediction point, default is to use 2\*min\_cluster\_size (used when training to determine the number of points needed for a group to be considered a cluster, at high amounts of data it's good to increase this in order to limit the amount of clusters you end up) which can get very costly at high numbers of min_cluster_size. So this is now a separate input to approximate_predict(), allowing for high values for min_cluster_size without affecting prediction times
	* ‡ Replace the above KDTree query with a [FAISS index search](https://github.com/facebookresearch/faiss/wiki/) (GPU powered)
	* Removes an unnecessary calculation when figuring out the mutual reachability neighbour. Previously it would get the argmin distance of an n\*3 matrix consisting of the core distances of the n neighbours (from the tree query), the core distance of the prediction point repeated n times (yes, the exact same number n times) and the distances between the prediction point and the n nearest neighbours. The 2nd row was unnecessary and thus removed
	* † Replace the get_tree_row_with_child cython function with a numpy array tree_row_indices. get_tree_row_with_child would previously iterate through each element in the condensed tree (tree with a node for every training point + one for each cluster) 'child' column until it found the one it needed (e.g. if it wanted the tree row for point 23, it would iterate through every node in the tree until it found the one attributed to 23, it would then return tree[23]). This is obviously not efficient, cython or not. The replacement simply takes the location of each node and uses the index of the array as a reference point. Meaning tree_row_indices[23] will return the row of the tree referring to point 23. So rather than looping through the tree until we find 23, we simply call tree[tree_row_indices[23]] and the row is returned at much greater speeds
	* † A similar thing is done for the cluster tree to build an array cluster_tree_row_indices. This only has nodes for each of the clusters, but beforehand this tree would constantly be referred to with cluster_tree['lambda_val'][cluster_tree['child'] == potential_cluster]. Rather than run these checks every time, it's slightly faster to use a similar index. So the equivalent here would be cluster_tree[cluster_tree_row_indices[potential_cluster]]
	* Doesn't calculate any probabilities, they're mostly useless and just a heavy load on computation
* This custom approximate_predict is used as the transform() function in the superclass
So the end result is a HDBSCAN implementation that runs (mostly) on GPU with fit() and transform() functions, allowing it to also be slotted in where the cpu implementation of HDBSCAN was previously

## cTF-IDF 

No change to the actual algorithm; this is decently fast and only run when training, not predicting. We do add an optional lemmatiser in an attempt to improve the keyword selections. The code was just significantly cleaned up

We also add the option to use KeyBERT as a keyword extraction backend

## Misc

* Added the ability to get a list of 'similar topics' to a given input. Uses the embeddings of each topic (calculated during the cTF-IDF) and embeds the input. Then calculates the cosine similarity between the input and all topics, returning the closest n topics. Is used as a fast alternative to transform(), where you specifically want more than one option.

* Model parameters can be input for all steps as a single dict for customisability, but there is also a model_parameters.yaml file that can be used

* To reduce outliers, we introduced *Matryoshka models*. This is where you recursively train subsequent UMAP/HDBSCAN models on the outliers of the previous UMAP/HDBSCAN model. This helps to reduce outliers

* We added support for CLIP image embeddings as well - allowing you to train topic models on images