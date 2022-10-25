[<< Back](..)

# v0.3.0


**<span style="color:purple">Buzzwords</span>_(params_dict: Dict[str, Dict[str, str]] = None)_**


Model capable of gathering topics from a collection of text or image documents


#### Parameters
* params_dict : <b><i>Dict[str, Dict[str, str]], optional</i></b>  Custom parameters for the model, use to override the defaults. Has the
    format `{ model_type1: {parameter1: value, parameter2: value }, .. }` with the
    following model types:
    * ***Embedding*** -
        Parameters from your chosen `model_type`, e.g. [SentenceTransformers](https://www.sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer) or [CLIP](https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/clip.py#L94)
    * ***UMAP*** -
        Parameters from [UMAP](https://docs.rapids.ai/api/cuml/stable/api.html#umap)
    * ***HDBSCAN*** -
        Parameters from [HDBSCAN](https://docs.rapids.ai/api/cuml/stable/api.html#cuml.cluster.HDBSCAN)
    * ***Keywords***
        * min_df : **int**
            min_df value for CountVectoriser
        * num_words : **int**
            Number of keywords to return per topic
        * num_word_candidates : **int**
            Number of top td-idf candidates to consider as possible keywords
    * ***Buzzwords***
        * lemmatise_sentences : **bool**
            Whether to lemmatise sentences before getting keywords
        * embedding_batch_size : **int**
            Batch size to embed sentences with
        * matryoshka_decay: **int**
            The speed at which HDBSCAN.min_cluster_size decreases on each recursion
        * get_keywords: **bool**
            Whether or not to gather the keywords
        * keyword_model: **str**
            Which keyword model to use - 'ctfidf' or 'keybert'
        * model_type: **str**
            Type of encoding model to run, e.g. 'sentencetransformers' or 'clip'

#### Attributes
* model_parameters : <b><i>Dict[str, Dict[str, str]]</i></b>  Values for each model (params_dict parameter alters this)
* embedding_model : <b><i>Union[SentenceTransformer, clip_encoder.CLIPEncoder]</i></b>  Chosen embedding model
* umap_model : <b><i>cuml.manifold.umap.UMAP</i></b>  cuML's UMAP model
* hdbscan_model : <b><i>cuml.cluster.hdbscan.HDBSCAN</i></b>  Custom HDBSCAN model
* keyword_model : <b><i>keywords.Keywords</i></b>  Chosen model for keyword gathering
* topic_embeddings : <b><i>np.ndarray</i></b>  The top n (num_word_candidates) words concatenated and embedded for each topic
* topic_descriptions : <b><i>Dict[int, str]</i></b>  {Topic number:Topic Keywords} for each topic

#### Examples
```python
from buzzwords import Buzzwords
params_dict = {'UMAP': {'n_neighbors': 20}}
model = Buzzwords(params_dict)
```

Run with default params, overriding UMAP's n_neighbors value

```python
model = Buzzwords()
docs = df['text_column']
topics = model.fit_transform(docs)
topic_keywords = [model.topic_descriptions[topic] for topic in topics]
```

Basic model training

```python
model = Buzzwords()
train_df = df.iloc[:50000]
pred_df = df.iloc[50000:]
topics = model.fit_transform(train_df.text_col)
topics.extend(model.transform(pred_df.text_col.reset_index(drop=True)))
```

Train a model on a batch of data, predict on the rest

```python
keyword = 'covid vaccine corona'
closest_topics = model.get_closest_topic(keyword, n=5)
```

Get 5 topics (from a trained model) similar to a given phrase

```python
model = Buzzwords()
model.load('saved_model/')
```

Load a saved model from disk

***


**<span style="color:purple">fit</span>_(docs: List[str], recursions: int = 1) -> None_**


Fit model based on given data


#### Parameters
* docs : <b><i>List[str]</i></b>  Text documents to get topics for
* recursions : <b><i>int</i></b>  Number of times to recurse the model. See Notes

#### Notes
Also accepts numpy arrays/pandas series as input
Make sure to reset_index(drop=True) if you use a Series

*recursions* is used as input for `matryoshka_iteration()`, the outlier reduction
method. When it's set to 1, the model is run once on the input data, which can leave a
significant number of outliers. To alleviate this, you can recurse the fit and run
another fit_transform on the outliers themselves. This will consider the outliers a
separate set of data and train a new model to cluster them, repeating recursions
times. The format of the output is the same, except as num_recursions increases, the
amount of outliers in the final dataset decreases.

***


**<span style="color:purple">fit&#95;transform</span>_(docs: List[str], recursions: int = 1) -> List[int]_**


Fit model based on given data and return the transformations


#### Parameters
* docs : <b><i>List[str]</i></b>  Text documents to get topics for
* recursions : <b><i>int</i></b>  Number of times to recurse the model. See Notes

#### Returns
* topics : <b><i>List[int]</i></b>  Topics for each document

#### Notes
Also accepts numpy arrays/pandas series as input
Make sure to reset_index(drop=True) if you use a Series

*recursions* is used as input for `matryoshka_iteration()`, the outlier reduction
method. When it's set to 1, the model is run once on the input data, which can leave a
significant number of outliers. To alleviate this, you can recurse the fit and run
another fit_transform on the outliers themselves. This will consider the outliers a
separate set of data and train a new model to cluster them, repeating recursions
times. The format of the output is the same, except as num_recursions increases, the
amount of outliers in the final dataset decreases.

***


**<span style="color:purple">get&#95;closest&#95;topic</span>_(word: str, n: int = 5) -> List[Tuple[int, str]]_**


Return the top n closest topics a new document may go in


#### Parameters
* word : <b><i>str</i></b>  Keyword to gather closest topics for
* n : <b><i>int</i></b>  Number of closest topics to show

#### Returns
* closest_topics : <b><i>List[Tuple[int, str]]</i></b>  List with topic_index:topic_keywords of the n closest topics

#### Notes
This differs from transform() as it returns multiple choices based
on the full-size topic embedding, rather than use UMAP/HDBSCAN
to return a single pred.

Generating probabilities for HDBSCAN cluster selections is very
inefficient, so this is a simpler alternative if we want to
return multiple possible topics

***


**<span style="color:purple">load</span>_(destination: str) -> None_**


Load model from local filesystem


#### Parameters
* destination : <b><i>str</i></b>  Location of locally saved model

***


**<span style="color:purple">matryoshka&#95;iteration</span>_(embeddings: <built-in function array>, recursions: int, min_cluster_size: int = None, highest_topic: int = -1, topics: <built-in function array> = None) -> <built-in function array>_**


Iterate through a training loop of umap/hdbscan, recursing on outliers each time


#### Parameters
* embeddings : <b><i>np.array</i></b>  Vector embeddings to cluster
* recursions : <b><i>int</i></b>  Number of times to recursively run this function
* min_cluster_size : <b><i>int</i></b>  HDBSCAN.min_cluster_size to use for this iteration
* highest_topic : <b><i>int</i></b>  Highest topic number from previous recursion
* topics : <b><i>np.array</i></b>  Topic list from previous recursion

#### Returns
* topics : <b><i>np.array</i></b>  Every topic for the input data - 1 per datapoint in the input

#### Notes
This is a recursive function for adding more granularity to a model. It's used to train a
new UMAP/HDBSCAN model on the outliers of each previous recursion. e.g. if you run a model
and the first recursion has 40% outliers with 100 topics, the next recursion would be run
only on the 40% outliers and would start from topic 101. This keeps going _recursions_
times, reducing the number of outliers and increasing the number of topics each time.

The final output will then be a stack of models which will cascade down when transforming
new data. So for a given datapoint, the simplified process goes like:

```
for model in model_stack:
    topic = model.predict(datapoint)

    if topic is not an outlier:
        break
```

The key to getting a good distribution of topics is in the `matryoshka_decay` parameter.
This will reduce the minimum cluster size multiplicatively on each recursion, meaning you
get a smooth transition from large models to smaller models. To illustrate this, imagine
you set a minimum cluster size of 400 for a training dataset of size 500k - the third
recursion training set will be much smaller than 500k, so it doesn't necessarily make
sense to keep the min cluster size at 400 (it will lead to a very skewed topic
distribution and outliers are often dealt with poorly). By multiplying 400 by a matryoshka
decay of 0.8, it means that the second recursion has a min cluster size of 400*0.8=320
and then the third has 320*0.8=256 and so on

***


**<span style="color:purple">merge&#95;topic</span>_(source_topic: int, destination_topic: int) -> None_**


Merge two similar topics into one. This is useful when you want
to perform surgery on your topic model and curate the topics found


#### Parameters
* source_topic : <b><i>int</i></b>  Topic to add to destination_topic
* destination_topic : <b><i>int</i></b>  Topic to add source_topic to

***


**<span style="color:purple">save</span>_(destination: str) -> None_**


Save model to local filesystem


#### Parameters
* destination : <b><i>str</i></b>  Location to dump model to

***


**<span style="color:purple">transform</span>_(docs: List[str]) -> <built-in function array>_**


Predict topics with trained model on new data


#### Parameters
* docs : <b><i>List[str]</i></b>  New data to predict topics for

#### Returns
* topics : <b><i>numpy.ndarray[int]</i></b>  Topics for each document

#### Notes
Also accepts numpy arrays/pandas series as input
Make sure to reset_index(drop=True) if you use a series

***


**<span style="color:purple">transform&#95;iteration</span>_(embeddings: <built-in function array>, umap_models: List[cuml.manifold.umap.UMAP], hdbscan_models: List[cuml.cluster.hdbscan.hdbscan.HDBSCAN], topics: <built-in function array> = None) -> <built-in function array>_**


Iterate through UMAP and HDBSCAN models to reduce outliers


#### Parameters
* embeddings : <b><i>np.array</i></b>  Vector embeddings to get clusters for
* umap_models : <b><i>List[UMAP]</i></b>  Trained UMAP models to iterate through
* hdbscan_models : <b><i>List[HDBSCAN]</i></b>  Trained HDBSCAN models to iterate through
* topics : <b><i>np.array</i></b>  List of topics from previous recursion

#### Returns
* topics : <b><i>np.array</i></b>  Every topic for the input data - 1 per datapoint in the input

#### Notes
See matryoshka_iteration() Notes

***


