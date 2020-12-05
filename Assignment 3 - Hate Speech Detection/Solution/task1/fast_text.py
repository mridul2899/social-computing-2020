import fasttext
import pandas as pd
from preprocess import make_temp_file, del_temp_file

def predict_ft(df_train, df_test):
    """
    Predicts labels on the test data set by computing fasttext vectors
    Takes two arguments:
        df_train - training dataframe, must have preprocessed text and labels
        df_test - testing dataframe, must have ids and text
    Saves the labels for test data in file FT.csv
    """

    # make a temporary file to store labels and text for fasttext
    temp_name = 'temp.train'
    make_temp_file(df_train, temp_name)

    # train the fasttext model on training data set
    model = fasttext.train_supervised(input=temp_name, verbose=False)

    # create a list of text for predicting its labels
    x_test = list(df_test['text'])

    # get labels for the the test data
    labels = model.predict(x_test)[0]

    # make dataframe with proper format which has ids and labels
    df_final = pd.DataFrame({'hateful': labels}, index=df_test.index)
    df_final['hateful'] = df_final['hateful'].apply(lambda x : int(x[0][-1]))

    # save the dataframe to csv file
    df_final.to_csv('../predictions/FT.csv')

    # delete temporary files
    del_temp_file(temp_name)