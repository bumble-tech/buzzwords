import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords


class CustomWordNetLemmatizer():
    """
    Class for lemmatising sentences using nltk and Wordnet

    Attributes
    ----------
    lemmatiser : nltk.stem.wordnet.WordNetLemmatizer
        Actual lemmatiser object
    stopwords : Set[str]
        Set of stopwords used
    """

    def __init__(self):
        self.lemmatiser = WordNetLemmatizer()

        # For POS, helps provide context for lemmatiser
        self.tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }

        self.stopwords = set(stopwords.words('english'))

    def wordnet_lemmatise_sentence(self, sentence: str) -> str:
        """
        Lemmatise all words in a given sentence

        Parameters
        ----------
        sentence : str
            string to be lemmatised

        Returns
        -------
        rejoined_sentence : str
            Sentence with all words lemmatised
        """

        # Split sentence into tokens
        tokenised_sentence = nltk.word_tokenize(sentence.lower())

        # Lemmatise each token
        lemmatised_sentence = [
            self.lemmatiser.lemmatize(
                word,
                self.get_wordnet_pos(word)
            ) for word in tokenised_sentence if word not in self.stopwords
        ]

        # Join list of words back together
        rejoined_sentence = ' '.join(lemmatised_sentence)

        return rejoined_sentence

    def get_wordnet_pos(self, word: str) -> str:
        """
        Map Part-Of-Speech tag to first character lemmatize() accepts

        Parameters
        ----------
        word : str
            word for which we need the POS

        Returns
        -------
        pos : str
            POS tag for the word
        """
        tag = nltk.pos_tag([word])[0][1][0].upper()

        pos = self.tag_dict.get(tag, wordnet.NOUN)

        return pos
