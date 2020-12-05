import string
import pandas as pd
import csv
import os

def make_temp_file(df, name):
    """
    Makes a temporary file for training fasttext model
    Takes two arguments:
        df - training dataframe, must have text and labels
        name - name of the temp file to be created
    """

    file = open(name, 'w')
    for index, line in df.iterrows():
        file.write(f'__label__{line.hateful} {line.text}\n')
    file.close()

def del_temp_file(name):
    """
    Deletes the temporary file used for training fasttext model
    Takes one argument:
        name - name of the temp file to be created
    """

    os.remove(name)

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