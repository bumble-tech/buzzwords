import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import cupy
import numpy as np
import pandas as pd
import yaml
from cuml import UMAP
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .models.clip_encoder import CLIPEncoder
from .models.custom_hdbscan import HDBSCAN
from .models.keywords import Keywords
from .models.lemmatiser import CustomWordNetLemmatizer
from .utils.generic_utils import HideAllOutput


class Buzzwords():
    """
    Model capable of gathering topics from a collection of text or image documents

    Parameters
    ----------
    params_dict : Dict[str, Dict[str, str]], optional
        Custom parameters for the model, use to override the defaults. Has the
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

    Attributes
    ----------
    model_parameters : Dict[str, Dict[str, str]]
        Values for each model (params_dict parameter alters this)
    embedding_model : Union[SentenceTransformer, clip_encoder.CLIPEncoder]
        Chosen embedding model
    umap_model : cuml.manifold.umap.UMAP
        cuML's UMAP model
    hdbscan_model : custom_hdbscan.HDBSCAN
        Custom HDBSCAN model
    keyword_model : keywords.Keywords
        Chosen model for keyword gathering
    topic_embeddings : np.ndarray
        The top n (num_word_candidates) words concatenated and embedded for each topic
    topic_descriptions : Dict[int, str]
        {Topic number:Topic Keywords} for each topic

    Examples
    --------
    >>> from buzzwords import Buzzwords
    >>> params_dict = {'UMAP': {'n_neighbors': 20}}
    >>> model = Buzzwords(params_dict)

    Run with default params, overriding UMAP's n_neighbors value

    >>> model = Buzzwords()
    >>> docs = df['text_column']
    >>> topics = model.fit_transform(docs)
    >>> topic_keywords = [model.topic_descriptions[topic] for topic in topics]

    Basic model training

    >>> model = Buzzwords()
    >>> train_df = df.iloc[:50000]
    >>> pred_df = df.iloc[50000:]
    >>> topics = model.fit_transform(train_df.text_col)
    >>> topics.extend(model.transform(pred_df.text_col.reset_index(drop=True)))

    Train a model on a batch of data, predict on the rest

    >>> keyword = 'covid vaccine corona'
    >>> closest_topics = model.get_closest_topic(keyword, n=5)

    Get 5 topics (from a trained model) similar to a given phrase

    >>> model = Buzzwords()
    >>> model.load('saved_model/')

    Load a saved model from disk

    """  # noqa: E501

    def __init__(self,
                 params_dict: Dict[str, Dict[str, str]] = None):
        # Load default parameters from config file
        model_parameters_file = \
            Path(__file__).parent / 'model_parameters.yaml'

        with open(model_parameters_file, 'r') as file:
            self.model_parameters = yaml.safe_load(file)

        # Update parameters if any were given when initialising
        if params_dict is not None:
            self.model_parameters = {
                model_type: dict(params, **params_dict.get(model_type, {}))
                for model_type, params in self.model_parameters.items()
            }

        bw_parameters = self.model_parameters['Buzzwords']
        self.lemmatise = bw_parameters['lemmatise_sentences']
        self.embedding_batch_size = bw_parameters['embedding_batch_size']
        self.get_keywords = bw_parameters['get_keywords']
        self.matryoshka_decay = bw_parameters['matryoshka_decay']
        self.keyword_backend = bw_parameters['keyword_backend']
        self.prediction_data = bw_parameters['prediction_data']

        model_type = bw_parameters['model_type']

        embedding_params = self.model_parameters['Embedding']

        if model_type == 'sentencetransformers':
            embedding_model = SentenceTransformer(**embedding_params)
        elif model_type == 'clip':
            embedding_model = CLIPEncoder(**embedding_params)

        if self.get_keywords:
            keyword_parameters = self.model_parameters['Keywords']

            self.keyword_model = Keywords(
                embedding_model,
                **keyword_parameters
            )

        self.embedding_model = embedding_model

        if self.lemmatise:
            self.lemmatiser = CustomWordNetLemmatizer()

        # Each topic has an embedding of its own
        self.topic_embeddings = None

        self.topic_descriptions = None
        self.topic_alterations = {}

    def fit(self, docs: List[str], recursions: int = 1) -> None:
        """
        Fit model based on given data

        Parameters
        ----------
        docs : List[str]
            Text documents to get topics for
        recursions : int
            Number of times to recurse the model. See Notes

        Notes
        -----
        Also accepts numpy arrays/pandas series as input
        Make sure to reset_index(drop=True) if you use a Series

        *recursions* is used as input for `matryoshka_iteration()`, the outlier reduction
        method. When it's set to 1, the model is run once on the input data, which can leave a
        significant number of outliers. To alleviate this, you can recurse the fit and run
        another fit_transform on the outliers themselves. This will consider the outliers a
        separate set of data and train a new model to cluster them, repeating recursions
        times. The format of the output is the same, except as num_recursions increases, the
        amount of outliers in the final dataset decreases.
        """
        _ = self.fit_transform(
            docs=docs,
            recursions=recursions
        )

    def fit_transform(self, docs: List[str], recursions: int = 1) -> List[int]:
        """
        Fit model based on given data and return the transformations

        Parameters
        ----------
        docs : List[str]
            Text documents to get topics for
        recursions : int
            Number of times to recurse the model. See Notes

        Returns
        -------
        topics : List[int]
            Topics for each document

        Notes
        -----
        Also accepts numpy arrays/pandas series as input
        Make sure to reset_index(drop=True) if you use a Series

        *recursions* is used as input for `matryoshka_iteration()`, the outlier reduction
        method. When it's set to 1, the model is run once on the input data, which can leave a
        significant number of outliers. To alleviate this, you can recurse the fit and run
        another fit_transform on the outliers themselves. This will consider the outliers a
        separate set of data and train a new model to cluster them, repeating recursions
        times. The format of the output is the same, except as num_recursions increases, the
        amount of outliers in the final dataset decreases.
        """
        # Progress bar
        self.pbar = tqdm(
            total=1 + (2 * recursions) + self.get_keywords,
            desc="Training: "
        )

        # Encode text to vectors
        embeddings = self.embedding_model.encode(
            docs,
            batch_size=self.embedding_batch_size,
            show_progress_bar=False
        )

        self.pbar.update(1)

        self.umap_models = []
        self.hdbscan_models = []

        # Run UMAP -> HDBSCAN _recursions_ times
        topics = self.matryoshka_iteration(
            embeddings=embeddings,
            recursions=recursions,
            min_cluster_size=int(self.model_parameters['HDBSCAN']['min_cluster_size'])
        )

        # Lemmatise words to avoid similar words in top n keywords
        if self.lemmatise:
            docs = [
                self.lemmatiser.wordnet_lemmatise_sentence(doc) for doc in docs
            ]

        # Get keywords from the topics
        if self.get_keywords:
            df = pd.DataFrame({
                'text': docs,
                'topic': topics
            })

            # Get keywords using tf-idf
            self.topic_descriptions, self.topic_embeddings = \
                self.keyword_model.get_topic_keywords(
                    df,
                    backend=self.keyword_backend
                )
            self.pbar.update(1)

            # Topic index -1 indicates outliers
            self.topic_descriptions[-1] = 'Outliers'

            topics = df['topic'].tolist()

        # Can't serialise without setting it to None
        self.pbar.close()
        self.pbar = None

        return topics

    def matryoshka_iteration(self,
                             embeddings: np.array,
                             recursions: int,
                             min_cluster_size: int = None,
                             highest_topic: int = -1,
                             topics: np.array = None) -> np.array:
        """
        Iterate through a training loop of umap/hdbscan, recursing on outliers each time

        Parameters
        ----------
        embeddings : np.array
            Vector embeddings to cluster
        recursions : int
            Number of times to recursively run this function
        min_cluster_size : int
            HDBSCAN.min_cluster_size to use for this iteration
        highest_topic : int
            Highest topic number from previous recursion
        topics : np.array
            Topic list from previous recursion

        Returns
        -------
        topics : np.array
            Every topic for the input data - 1 per datapoint in the input

        Notes
        -----
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
        """
        # When we reach the bottom of the recursive loop, go back to the top
        if recursions == 0:
            return topics

        # Reduce dimensionality of vector embeddings
        self.umap_models.append(UMAP(**self.model_parameters['UMAP']))
        umap_embeddings = self.umap_models[-1].fit_transform(embeddings)

        self.pbar.update(1)

        # Cluster topics
        updated_params = self.model_parameters['HDBSCAN'].copy()

        if min_cluster_size is not None:
            # Apply updated min_cluster_size
            updated_params['min_cluster_size'] = min_cluster_size
        else:
            min_cluster_size = updated_params['min_cluster_size']

        self.hdbscan_models.append(HDBSCAN(**updated_params))

        # HideAllOutput hides the Cython output from HDBSCAN
        with HideAllOutput():
            self.hdbscan_models[-1].fit(
                umap_embeddings,
                prediction_data=self.prediction_data
            )
        self.pbar.update(1)

        self.hdbscan_models[-1].min_topic = highest_topic + 1

        """
        Without using copy() here, we run into a VERY interesting bug wherein the labels
        update as the topics do. In theory the first layer should have a max value < the
        final topic list max value. Without copy(), the first layer's labels_ will update
        to perfectly mirror the FINAL topic list (even though it should know nothing after
        the first recursion). Doesn't break anything, just counter-intuitive so we use copy()

        Probably just a result of poor memory usage on our part, turns out we're flawed :')
        """
        if type(self.hdbscan_models[-1].labels_) == cupy._core.core.ndarray:
            topics = self.hdbscan_models[-1].labels_.get().copy()
        else:
            topics = np.array(self.hdbscan_models[-1].labels_).copy()

        # -1 indicates outliers
        outlier_topics = topics == -1

        # We need to start the topics from the prev recursion's max topic value + 1
        topics[~outlier_topics] = topics[~outlier_topics] + highest_topic + 1

        # Go down one level recursively
        if outlier_topics.any():
            topics[outlier_topics] = self.matryoshka_iteration(
                embeddings=embeddings[outlier_topics],
                recursions=recursions - 1,
                min_cluster_size=int(min_cluster_size * self.matryoshka_decay),
                highest_topic=topics.max(),
                topics=topics[outlier_topics]
            )

        return topics

    def transform(self, docs: List[str]) -> np.array:
        """
        Predict topics with trained model on new data

        Parameters
        ----------
        docs : List[str]
            New data to predict topics for

        Returns
        -------
        topics : numpy.ndarray[int]
            Topics for each document

        Notes
        -----
        Also accepts numpy arrays/pandas series as input
        Make sure to reset_index(drop=True) if you use a series
        """
        if self.prediction_data is False:
            raise Exception('`prediction_data` is set to false for this model')

        embeddings = self.embedding_model.encode(
            docs,
            batch_size=self.embedding_batch_size,
            show_progress_bar=False
        )

        topics = self.transform_iteration(
            embeddings,
            self.umap_models,
            self.hdbscan_models
        )

        topics = np.array(
            [self.topic_alterations.get(topic, topic) for topic in topics]
        )

        return topics

    def transform_iteration(self,
                            embeddings: np.array,
                            umap_models: List[UMAP],
                            hdbscan_models: List[HDBSCAN],
                            topics: np.array = None) -> np.array:
        """
        Iterate through UMAP and HDBSCAN models to reduce outliers

        Parameters
        ----------
        embeddings : np.array
            Vector embeddings to get clusters for
        umap_models : List[UMAP]
            Trained UMAP models to iterate through
        hdbscan_models : List[HDBSCAN]
            Trained HDBSCAN models to iterate through
        topics : np.array
            List of topics from previous recursion

        Returns
        -------
        topics : np.array
            Every topic for the input data - 1 per datapoint in the input

        Notes
        -----
        See matryoshka_iteration() Notes

        """
        # When we reach the bottom, go back to the top
        if not umap_models:
            return topics

        # Get cluster of data
        umap_embeddings = umap_models[0].transform(embeddings)
        topics = hdbscan_models[0].approximate_predict(umap_embeddings)

        outlier_topics = topics == -1

        # Account for models covering different topic ranges (model 2 doesn't start from topic 0)
        topics[~outlier_topics] = topics[~outlier_topics] + hdbscan_models[0].min_topic

        # Next recursion
        topics[outlier_topics] = self.transform_iteration(
            embeddings[outlier_topics],
            umap_models[1:],
            hdbscan_models[1:],
            topics[outlier_topics]
        )

        return topics

    def get_closest_topic(self,
                          word: str,
                          n: int = 5) -> List[Tuple[int, str]]:
        """
        Return the top n closest topics a new document may go in

        Parameters
        ----------
        word : str
            Keyword to gather closest topics for
        n : int
            Number of closest topics to show

        Returns
        -------
        closest_topics : List[Tuple[int, str]]
            List with topic_index:topic_keywords of the n closest topics

        Notes
        -----
        This differs from transform() as it returns multiple choices based
        on the full-size topic embedding, rather than use UMAP/HDBSCAN
        to return a single pred.

        Generating probabilities for HDBSCAN cluster selections is very
        inefficient, so this is a simpler alternative if we want to
        return multiple possible topics

        """
        if self.topic_embeddings is None:
            raise Exception('You must train the model first')

        # Embed input
        test_embed = self.embedding_model.encode(word)

        similarities = []

        # Calculate similarities for each topic embedding
        for topic in self.topic_embeddings.keys():
            similarity = cosine_similarity(
                test_embed.reshape(1, -1),
                self.topic_embeddings[topic].reshape(1, -1)
            )

            similarities.append(np.array([topic, similarity]))

        similarities = np.array(similarities)

        similarities = similarities[
            similarities[:, 1].argsort()][::-1][:n]

        closest_topics = [(n, self.topic_descriptions[n])
                          for n in similarities[:, 0]]

        return closest_topics

    def merge_topic(self,
                    source_topic: int,
                    destination_topic: int) -> None:
        """
        Merge two similar topics into one. This is useful when you want
        to perform surgery on your topic model and curate the topics found

        Parameters
        ----------
        source_topic : int
            Topic to add to destination_topic
        destination_topic : int
            Topic to add source_topic to

        """
        self.topic_alterations[source_topic] = destination_topic

    def save(self, destination: str) -> None:
        """
        Save model to local filesystem

        Parameters
        ----------
        destination : str
            Location to dump model to
        """
        if self.prediction_data is False:
            raise Exception('`prediction_data` is set to false for this model')

        destination = Path(destination)

        # If file directory was given instead of filename
        if not destination.suffix:
            destination = destination / 'model.buzz'

        # Drop FAISS indices so we can serialise
        for index, hdbscan_model in enumerate(self.hdbscan_models):
            self.hdbscan_models[index].faiss_index = None

        with open(destination, 'wb') as file:
            pickle.dump(self.__dict__, file)

        # Recreate FAISS indices
        for index, hdbscan_model in enumerate(self.hdbscan_models):
            self.hdbscan_models[index].build_faiss_index()

    def load(self, destination: str) -> None:
        """
        Load model from local filesystem

        Parameters
        ----------
        destination : str
            Location of locally saved model
        """
        destination = Path(destination)

        # If file directory was given instead of filename
        if not destination.suffix:
            destination = destination / 'model.buzz'

        with open(destination, 'rb') as file:
            self.__dict__ = pickle.load(file)

        # Recreate FAISS indices
        for i in range(len(self.hdbscan_models)):
            self.hdbscan_models[i].build_faiss_index()
