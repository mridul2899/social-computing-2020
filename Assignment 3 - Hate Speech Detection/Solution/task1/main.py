import os
from preprocess import read_csv, punctuation_lower
from tfidf import predict_rf
from word2vec import predict_svm
from fast_text import predict_ft

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

    # compute tf-idf vectors and implement random forest to predict labels
    y_train = predict_rf(df_train, df_test)

    # compute word2vec vectors and implement SVM to predict labels
    predict_svm(df_train, df_test, y_train)

    # compute fasttext vectors and use them to predict labels
    predict_ft(df_train, df_test)