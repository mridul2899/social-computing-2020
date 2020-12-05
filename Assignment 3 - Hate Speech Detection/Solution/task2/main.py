import os
from preprocess import read_csv, punctuation_lower, lemmatize
import spacy
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

kfold = KFold(5, True, 1)
nlp = spacy.load("en_core_web_md")

def xgb(x_train, y_train, x_test):
    """
    Fits XGBoost classifier on training data and predicts labels on test data
    Takes three arguments:
        x_train - numpy array of training data embeddings
        y_train - numpy array of training data labels
        x_test - numpy array of test data embeddings
    Returns the predicted labels
    """

    model = XGBClassifier()
    # y_train = np.reshape(y_train, (len(y_train), 1))
    # data = np.concatenate((x_train, y_train), axis=1)
    # for train, test in kfold.split(data):
    #     # print("reached here")
    #     x_tr = data[train, :-1]
    #     y_tr = data[train, -1]
    #     x_va = data[test, :-1]
    #     y_va = data[test, -1]

    #     model.fit(x_tr, y_tr)
    #     y_pred = model.predict(x_va)
    #     predictions = [round(value) for value in y_pred]
    #     f1 = f1_score(y_va, predictions)
    #     print(f1)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    y_predict = [round(value) for value in y_predict]
    return y_predict

if __name__ == '__main__':
    # make a directory for saving predictions if not already made
    try:
        os.mkdir('../predictions')
    except:
        pass

    # read the tsv files and load them as Pandas dataframes
    df_train = read_csv('../data/train.tsv')
    df_test = read_csv('../data/test.tsv')

    # preprocess text - remove punctuation, convert into lowercase
    df_train['text'] = punctuation_lower(df_train['text'])
    df_test['text'] = punctuation_lower(df_test['text'])

    df_train['text'] = lemmatize(df_train['text'])
    df_test['text'] = lemmatize(df_test['text'])

    # save SpaCy vector embeddings for the text in training data in x_train
    vectors = df_train['text'].apply(nlp)
    x_train = vectors.apply(lambda x : x.vector)
    x_train = x_train.to_numpy()
    x_train = np.vstack(x_train).astype(np.float)

    # save SpaCy vector embeddings for the text in test data in x_test
    vectors = df_test['text'].apply(nlp)
    x_test = vectors.apply(lambda x : x.vector)
    x_test = x_test.to_numpy()
    x_test = np.vstack(x_test).astype(np.float)

    y_train = np.array(df_train['hateful'])

    y_predict = xgb(x_train, y_train, x_test)

    pd.DataFrame({'hateful': y_predict}, index=df_test.index).to_csv('../predictions/T2.csv')
