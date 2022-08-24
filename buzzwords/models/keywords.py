import numpy as np
import pandas as pd
import scipy.sparse as sp
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.utils import check_array
from sklearn.metrics.pairwise import cosine_similarity

from typing import List, Dict, Tuple, Any


class Keywords():
    """
    Class for parsing keywords from the topics - with two backend types supported:
    * A Class-based TF-IDF procedure based on BERTopic
    * KeyBERT

    Parameters
    ----------
    embedding_model : sentence_transformers.SentenceTransformer.SentenceTransformer
        Sentence embedding model to use
    min_df : int
        min_df value to use for CountVectorizer
    num_words : int
        Number of keywords to return per topic
    num_word_candidates : int
        Number of top ranking words from TF/IDF to consider as possible keywords

    Attributes
    ----------
    vectoriser_model : sklearn.feature_extraction.text.CountVectorizer
        CountVectorizer model

    Notes
    -----
    C-TF-IDF can best be explained as a TF-IDF formula adopted for multiple classes
    by joining all documents per class. Thus, each class is converted to a single document
    instead of set of documents. Then, the frequency of words **t** are extracted for
    each class **i** and divided by the total number of words **w**.
    Next, the total, unjoined, number of documents across all classes **m** is divided by the total
    sum of word **i** across all classes.

    Adapted from github.com/MaartenGr/BERTopic/blob/master/bertopic/_ctfidf.py
    """

    def __init__(self,
                 embedding_model: SentenceTransformer,
                 min_df: int = 1,
                 num_words: int = 4,
                 num_word_candidates: int = 30,
                 *args, **kwargs):
        self.vectoriser_model = CountVectorizer(
            min_df=min_df,
            stop_words='english')

        self.embedding_model = embedding_model

        """
        So this is a fun one

        We had some troubles where Buzzwords on an A100 would crash when it tried to encode
        text for keywords after clustering. It would encode the training set with no issues,
        then return a CUDA ordinal error at the keywords step

        This genuinely fixed it lol
        """
        self.embedding_model.encode(
            ['how the fuck does this solve the problem'],
            show_progress_bar=False
        )

        self.num_words = num_words
        self.num_word_candidates = num_word_candidates

    def get_topic_keywords(self,
                           documents: pd.DataFrame,
                           backend: str = 'keybert',
                           text_col_name: str = 'text',
                           topic_col_name: str = 'topic'
                           ) -> Tuple[Dict[int, str], Dict[int, np.ndarray]]:
        """
        Pull topic keywords from a dataframe of text and their corresponding topic

        Parameters
        ----------
        documents : pd.DataFrame
            Dataframe with 'topic' column and a text column
        backend : str
            Keyword backend to use - 'keybert' or 'ctfidf'
        text_col_name : str
            Name of 'documents' column with the text
        topic_col_name : str
            Name of 'documents' column with the topics

        Returns
        -------
        topic_names : Dict[int, str]
            Dictionary with keywords for each topic (e.g. {0: 'cat_dog_farm_animal', 1: ..})
        topic_embeddings : Dict[int, np.ndarray]
            Dictionary of topic : topic_embedding

        """
        documents[text_col_name] = documents[text_col_name].astype(str)

        # Concatenate all documents in each topic
        grouped_documents = (
            documents
            .groupby([topic_col_name], as_index=False)
            .agg({text_col_name: ' '.join})
        )

        message_clusters = grouped_documents[text_col_name]

        if backend == 'ctfidf':
            c_tf_idf, words = self._c_tf_idf(
                message_clusters,
                self.vectoriser_model)

            sizes = (
                documents
                .groupby([topic_col_name])
                .count()
                .sort_values(text_col_name, ascending=False)
                .reset_index()
            )

            topic_sizes = dict(zip(sizes.topic, sizes[text_col_name]))

            # Get keywords for each topic
            topics, topic_embeddings = self.extract_words_per_topic(
                words,
                c_tf_idf,
                topic_sizes
            )

            topic_names = {
                key: " ".join(
                    [word[0] for word in values[:self.num_words]]
                ) for key, values in topics.items()
            }
        elif backend == 'keybert':
            topic_names = {}

            keyword_model = KeyBERT()

            for topic in grouped_documents[topic_col_name]:
                docs = grouped_documents[grouped_documents[topic_col_name] == topic][text_col_name]

                topic_keywords = keyword_model.extract_keywords(
                    docs.iloc[0],
                    top_n=self.num_words
                )

                topic_names[topic] = ' '.join([
                    string[0]
                    for string in topic_keywords if string[1] > 0.2]
                )
            topic_names[-1] = 'Outliers'

            topic_embeddings = [self.embedding_model.encode(doc, show_progress_bar=False)
                                for doc in message_clusters]
        else:
            raise Exception(f'Keyword backend not supported - {backend}')

        return topic_names, topic_embeddings

    @staticmethod
    def _c_tf_idf(message_clusters: pd.Series,
                  vectoriser_model: CountVectorizer
                  ) -> Tuple[sp.csr.csr_matrix, List[str]]:
        """
        Run a class-based tf-idf model on the clusters of messages

        Parameters
        ----------
        message_clusters : pd.Series
            Strings of every message in a given cluster, concatenated
        vectoriser_model: sklearn.feature_extraction.text.CountVectorizer
            Vectoriser model to be used for the messages

        Returns
        -------
        X: sp.csr.csr_matrix
            CSR matrix of the tf-idf model
        words: List[str]
            List of words in the corpus
        """
        vectoriser_model.fit(message_clusters)
        words = vectoriser_model.get_feature_names_out()
        X = vectoriser_model.transform(message_clusters)

        X = check_array(X, accept_sparse=('csr', 'csc'))
        if not sp.issparse(X):
            X = sp.csr_matrix(X)

        df = np.squeeze(np.asarray(X.sum(axis=0)))
        avg_nr_samples = int(X.sum(axis=1).mean())
        idf = np.log(avg_nr_samples / df)
        _idf_diag = sp.diags(idf,
                             offsets=0,
                             shape=(X.shape[1], X.shape[1]),
                             format='csr',
                             dtype=np.float64)

        X = normalize(X, axis=1, norm='l1', copy=False)
        X = X * _idf_diag

        return X, words

    def extract_words_per_topic(self,
                                words: List[str],
                                c_tf_idf: sp.csr.csr_matrix,
                                topic_sizes: Dict[int, int]
                                ) -> Tuple[Dict[int, List[Tuple[Any, Any]]], Dict[int, np.ndarray]]:  # noqa: E501
        """
        Get the top keywords for each topic

        Parameters
        ----------
        words: List[str]
            Full corpus of words
        c_tf_idf: sp.csr.csr_matrix
            c-tf-idf model
        topic_sizes: Dict[int, int]
            Dict denoting the size of each topic

        Returns
        -------
        topics : Dict[int, List[Tuple[Any, Any]]]
            Dictionary of topic:topic_keywords
        topic_embeddings : Dict[int, np.ndarray]
            Dictionary of topic : topic_embedding
        """
        c_tf_idf = c_tf_idf.toarray()

        labels = sorted(list(topic_sizes.keys()))

        # Get top words per topic based on c-TF-IDF score
        indices = c_tf_idf.argsort()[:, -self.num_word_candidates:]
        topics = {label: [(words[j], c_tf_idf[i][j])
                          for j in indices[i]][::-1]
                  for i, label in enumerate(labels)}

        topic_embeddings = {}

        for topic, topic_words in topics.items():
            words = [word[0] for word in topic_words]
            word_embeddings = self.embedding_model.encode(
                words, show_progress_bar=False)
            topic_embedding = \
                self.embedding_model.encode(
                    " ".join(words),
                    show_progress_bar=False).reshape(1, -1)

            topic_words = self.mmr(topic_embedding, word_embeddings, words,
                                   top_n=self.num_words)
            topics[topic] = [
                (word, value) for word, value in topics[topic]
                if word in topic_words
            ]

            topic_embeddings[topic] = topic_embedding

        return topics, topic_embeddings

    @staticmethod
    def mmr(doc_embedding: np.ndarray,
            word_embeddings: np.ndarray,
            words: List[str],
            top_n: int = 5) -> List[str]:
        """
        Calculate Maximal Marginal Relevance (MMR) between candidate keywords and the document

        Parameters
        ----------
        doc_embedding: np.ndarray
            The document embeddings
        word_embeddings: np.ndarray
            The embeddings of the selected candidate keywords/phrases
        words: List[str]
            The selected candidate keywords/keyphrases
        top_n: int
            The number of keywords/keyphrases to return

        Returns
        -------
        mmr_output : List[str]
            The selected keywords/keyphrases

        Notes
        -----
        MMR considers the similarity of keywords/keyphrases with the
        document, along with the similarity of already selected
        keywords and keyphrases. This results in a selection of keywords
        that maximize their within diversity with respect to the document.

        Taken from BERTopic
        """

        # Extract similarity within words, and between words and the document
        word_doc_similarity = cosine_similarity(
            word_embeddings,
            doc_embedding
        )

        word_similarity = cosine_similarity(word_embeddings)

        # Initialize candidates and already choose best keyword/keyphras
        keywords_idx = [np.argmax(word_doc_similarity)]
        candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

        for _ in range(top_n - 1):
            # Extract similarities within candidates and
            # between candidates and selected keywords/phrases
            candidate_similarities = word_doc_similarity[candidates_idx, :]
            target_similarities = np.max(
                word_similarity[candidates_idx][:, keywords_idx], axis=1)

            # Calculate MMR
            mmr = 1 * candidate_similarities \
                - 1 * target_similarities.reshape(-1, 1)
            mmr_idx = candidates_idx[np.argmax(mmr)]

            # Update keywords & candidates
            keywords_idx.append(mmr_idx)
            candidates_idx.remove(mmr_idx)
        mmr_output = [words[idx] for idx in keywords_idx]

        return mmr_output
