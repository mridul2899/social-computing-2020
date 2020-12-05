import nltk
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd
import csv

lemmatizer = WordNetLemmatizer()

def read_csv(path):
    """
    Returns a dataframe after reading a tsv file as mentioned in path
    Ensures that all lines are read
    Takes one argument:
        path - path of the tsv file
    """

    df = pd.read_csv(path, sep='\t', index_col=0, quoting=csv.QUOTE_NONE, encoding='utf-8')
    return df

def punctuation_lower(series):
    """
    Removes punctuation from Pandas Series text and converts it to lowercase
    Takes one argument:
        series - Pandas series containing text
    Returns the series after preprocessing
    """

    series = series.apply(lambda x : x.translate(str.maketrans('', '', string.punctuation)).lower())
    return series

def lemmatize(series):
    series = series.apply(lambda x : ' '.join([lemmatizer.lemmatize(j) for j in x.split()]))
    return series