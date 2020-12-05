from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

def rf_results(x_train, y_train, x_test):
    """
    Fits Random Forest classifier on training data, predicts on test data
    Takes three arguments:
        x_train - numpy array of training data embeddings
        y_train - numpy array of training data labels
        x_test - numpy array of test data embeddings
    Returns the predicted labels
    """

    # define the Random Forest Classifier model with 10 trees
    clf = RandomForestClassifier(n_estimators=10)

    # train the RF model on training data
    clf.fit(x_train, y_train)

    # predict labels on test data using trained classifier
    y_predict = clf.predict(x_test)

    # return predicted labels
    return y_predict

def predict_rf(df_train, df_test):
    """
    Predicts labels on test data set by RF classifier on tf-idf embeddings
    Takes two arguments:
        df_train - training dataframe, must have text and labels
        df_test - testing dataframe, must have ids and text
    Saves the labels for test data in the file RF.csv
    Returns y_train - a numpy array of labels
    """

    # define tf-idf vectorizer with min and max document frequency parameters
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=5)

    # compute tf-idf vectors for training data
    vectors = vectorizer.fit_transform(df_train['text'])
    x_train = np.array(vectors.todense())

    # compute tf-idf vectors for test data
    vectors_test = vectorizer.transform(df_test['text'])
    x_test = np.array(vectors_test.todense())

    # create a numpy array of training data labels
    y_train = np.array(df_train['hateful'])

    # get predicted labels for test data set
    y_predict = rf_results(x_train, y_train, x_test)

    # create a dataframe from the predicted labels and save to csv file
    pd.DataFrame({'hateful': y_predict}, index=df_test.index).to_csv('../predictions/RF.csv')

    # return numpy array of training labels for further usage
    return y_train