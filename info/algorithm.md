[<< Back](index)

# Base Algorithm - How it Works

The model takes a collection of text documents as input and figures out a number of topics that are being discussed in the collection. Each document is attributed to a topic and specific keywords are picked out that are most relevant for each topic, to help with interpretability.

There are 4 main stages to the BERTopic/Top2Vec algorithm:

* Embedding the documents into a vector (using the SentenceTransformers library)
* Reducing the dimensionality of these vectors (using UMAP)
* Clustering those vectors into similar groups (using HDBSCAN)
* Gathering the topic keywords for each cluster (using a class-based TF/IDF model)

With this, each document is assigned one of the topics and the keywords for that topic are given to help give an idea of the content of each topic.

***

## Sentence Embedding

The sentences are embedded as vectors using a pretrained model in the [Sentence Transformers](https://www.sbert.net/index.html) library. Generally we use either [paraphrase-MiniLM-L3-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2/tree/main) or [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) as they're both very fast, but any pretrained model can be used. See a list of some pretrained models [here](https://www.sbert.net/docs/pretrained_models.html#multi-lingual-models)

## UMAP

[UMAP](https://umap-learn.readthedocs.io/en/latest/) (Uniform Manifold Approximation and Projection) is a powerful way of reducing the dimensionality of vectors. For a more in-depth idea of the workings of UMAP, please refer to the [paper](https://arxiv.org/abs/1802.03426)

## HDBSCAN

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) is a way of clustering based on the density of the points. HDBSCAN generally outperforms more naive clustering methods and also offers the option for points to be labelled as 'outliers' (topic: -1), meaning you don't force points into clusters they have no part of. More can be found in the [paper](http://pdf.xuebalib.com:1262/2ac1mJln8ATx.pdf)

## cTF-IDF

The class-based TF-IDF deals with the interpretability of each topic. Each topic is treated as a document in the index and the entirety of datapoints in each topic are appended together to act as this document (e.g. all messages for 'topic 0' are used as one datapoint for the TF-IDF, same for all messages in 'topic 1' and so on). This will then give the keywords that are most relevant to each topic, in the context of the whole corpus. So topic 0 may be given 'dogs pup puppies pups pets' if the texts for topic 0 revolve mostly around people having dogs.

The topics are then given as output, along with the keywords associated with each topic. The end result is a topic for every text datapoint, and an informative list of words describing the content of each topic

