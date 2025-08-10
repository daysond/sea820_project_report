from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk


def  preprocess_text(text: str, method='lemma'):
    """
    Preprocesses a given text string by applying common NLP cleaning steps.

    Steps:
    1. Lowercases the input text.
    2. Tokenizes the text into words.
    3. Removes English stopwords and non-alphanumeric tokens (e.g., punctuation).
    4. Applies either lemmatization or stemming to the remaining words.
    5. Joins the processed words back into a single string.

    Parameters:
    ----------
    text : str
        The input text to preprocess.
    method : str, optional (default='lemma')
        The method used to reduce words:
        - 'lemma' for lemmatization using WordNetLemmatizer.
        - Any other value will trigger stemming using PorterStemmer.

    Returns:
    -------
    str
        The cleaned and normalized text.
    """
    # lowercasing
    text = text.lower()

    # tokenize
    words = nltk.word_tokenize(text)

    # stopword and punctuation removal
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words and word.isalnum()]

    # lemmatization / stemming
    if method == 'lemma':
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
    else:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    
    return " ".join(words)