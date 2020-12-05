from sklearn.svm import SVC
import spacy
import numpy as np
import pandas as pd

# define the Spacy English model to use for generating embeddings
nlp = spacy.load("en_core_web_md")

def svm_results(x_train, y_train, x_test):
    """
    Fits SVM classifier on training data and predicts labels on test data
    Takes three arguments:
        x_train - numpy array of training data embeddings
        y_train - numpy array of training data labels
        x_test - numpy array of test data embeddings
    Returns the predicted labels
    """

    # define the Support Vector Classification model, default kernel = 'rbf'
    clf = SVC()

    # train the SVM model on training data
    clf.fit(x_train, y_train)

    # predict labels on test data using trained classifier
    y_predict = clf.predict(x_test)

    # return predicted labels
    return y_predict


def predict_svm(df_train, df_test, y_train):
    """
    Predicts labels on test data set by SVM classifier on word2vec embeddings
    Takes three arguments:
        df_train - training dataframe, must have text
        df_test - testing dataframe, must have ids and text
        y_train - numpy array of training data labels
    Saves the labels for test data in the file SVM.csv
    """

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

    # get predicted labels for test data set
    y_predict = svm_results(x_train, y_train, x_test)

    # create a dataframe from the predicted labels and save to csv file
    pd.DataFrame({'hateful': y_predict}, index=df_test.index).to_csv('../predictions/SVM.csv')