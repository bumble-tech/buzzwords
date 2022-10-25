[<< Back](index)

# Custom Model: What did we Change?

While the base algorithm is the same, we had to make a significant amount of changes to speed up the process for our use cases. Once you work on a bigger scale, the limitations of the base libraries become apparent so this was necessary

## Sentence Embedding

No change, runs using Pytorch so is automatically run on GPU where available

## UMAP

Instead of using the standard UMAP-learn library, we use cuML's UMAP implementation. cuML is a part of Nvidia's [RAPIDS.ai](https://rapids.ai/about.html) library that includes UMAP as standard. The version has been implemented for a while now and is well supported. As such, we can simply plug in cuML's version where the cpu version was previously. As of October 2022, we are also able to use cuML's HDBSCAN (previously we were using a custom version which was capable of inference using FAISS and a mix of cuML's and scikit-contrib's HDBSCAN implementations)

## cTF-IDF 

No change to the actual algorithm; this is decently fast and only run when training, not predicting. We do add an optional lemmatiser in an attempt to improve the keyword selections. The code was just significantly cleaned up

We also add the option to use KeyBERT as a keyword extraction backend

## Misc

* Added the ability to get a list of 'similar topics' to a given input. Uses the embeddings of each topic (calculated during the cTF-IDF) and embeds the input. Then calculates the cosine similarity between the input and all topics, returning the closest n topics. Is used as a fast alternative to transform(), where you specifically want more than one option.

* Model parameters can be input for all steps as a single dict for customisability, but there is also a model_parameters.yaml file that can be used

* To reduce outliers, we introduced *Matryoshka models*. This is where you recursively train subsequent UMAP/HDBSCAN models on the outliers of the previous UMAP/HDBSCAN model. This helps to reduce outliers

* We added support for CLIP image embeddings as well - allowing you to train topic models on images