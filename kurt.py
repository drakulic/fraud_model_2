import pandas as pd
import numpy as np
import random
import dill as pickle
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# from brian_fraud_detection import *
from brian_train_test_split_df import *

string_type_categorical_columns = ['country', 'currency', 'email_domain',
                                   'listed', 'payout_type', 'venue_country',
                                   'venue_state']

def make_target_variable(df):
    account_class_dict = {'premium':0,
                          'fraudster_event':1,
                          'fraudster':1,
                          'spammer_limited':0,
                          'spammer_warn':0,
                          'tos_warn':0,
                          'spammer_noinvite':0,
                          'tos_lock':0,
                          'locked':0,
                          'fraudster_att':1,
                          'spammer_web':0,
                          'spammer':0}
    df['fraud'] = df.acct_type.apply(lambda acct_type: account_class_dict[acct_type])
    df.drop(['acct_type'], axis=1, inplace=True)
    return df

def read_data():
    return pd.read_json('data/data.json')

def add_fraud(df):
    df['fraud'] = random.sample(xrange(100000), len(df))

def get_fraud_percentage(df):
    return float(np.sum(df.iloc[:,1])) / len(df)

def strip_html(row, column):
    # Strip HTML from 'column'
    return  BeautifulSoup(row, "lxml").get_text()

def replace_nan(df):
    df.fillna(-999, inplace = True)

def category_average_encoding(train_df, test_df, column):
    # Create dictionary of average encoding
    category_df = train_df[[column, 'fraud']]
    category_dict = category_df.groupby(column).aggregate(get_fraud_percentage).to_dict()['fraud']

    # Add NaN average to dictionary
    nan_indices = pd.isnull(category_df[column])
    if np.sum(nan_indices) != 0:
        category_dict[np.nan] = np.sum(category_df.loc[nan_indices, 'fraud']) / \
                                float(np.sum(nan_indices))

    # Encode columns
    train_df[column + 'encoded'] = train_df[column].map(category_dict)
    train_df[column + 'encoded'].fillna(-999, inplace = True)

    test_df[column + 'encoded'] = test_df[column].map(category_dict)
    test_df[column + 'encoded'].fillna(-999, inplace = True)
    with open('{}dict.obj'.format(column), 'w+') as f:
        pickle.dump(category_dict, f)
    return train_df, test_df


# Custom tokenizer
def tokenize(doc):
    '''
    INPUT: string
    OUTPUT: list of strings

    Tokenize and stem/lemmatize the document.
    '''
    # Tokenize
    words = word_tokenize(doc)

    # Stem
    porter = PorterStemmer()
    words = [porter.stem(word) for word in words]

    return words

def train_naive_bayes(df, text_variable = None, target_variable = None):
    """
    Input:
        df -- DataFrame of data
        text_variable -- string of column with text we are training naive Bayes
                         model on
        target_variable -- string of column of target variable
    Output:
        nb -- sklearn MultinomialNB object model
        vectorizer -- sklearn Vectorizer object
        df -- DataFrame with NaiveBayes prediction column added
    """

    # Get indices that have values
    indices = np.argwhere((df[text_variable] != '') &
                          (df[text_variable] != -999) &
                          (df[text_variable] != np.nan) &
                          (pd.notnull(df[text_variable]))).flatten()

    # Get documents
    docs = df.loc[indices, text_variable]

    print docs


    # Strip HTML
    docs = docs.apply(func = strip_html, args = (text_variable,))

    # Get target variables
    y = df.loc[indices, target_variable].values

    # Vectorize
    X, vectorizer = vectorize_docs(docs)

    # Fit and predict NaiveBayes
    nb = MultinomialNB(alpha = 1.0)
    nb.fit(X, y)

    # Predict NaiveBayes and add as new column to 'df'
    df['NB' + text_variable] = pd.Series(index = indices, data = nb.predict_proba(X)[:,1])
    print df[['NB' + text_variable, target_variable]]

    # Pickle models
    model_name = text_variable + 'NaiveBayesModel'
    with open('{}.obj'.format(model_name), 'w+') as f:
        pickle.dump(nb, f)

    model_name = text_variable + 'Vectorizer'
    with open('{}.obj'.format(model_name), 'w+') as f:
        pickle.dump(vectorizer, f)

    return nb, vectorizer, df

def add_naive_bayes_prediction(df, text_variable = None, target_variable = None):
    # Read model and vectorizer
    model_name = text_variable + 'NaiveBayesModel'
    with open('models/{}.obj'.format(model_name), 'r') as f:
        model = pickle.load(f)

    model_name = text_variable + 'Vectorizer'
    with open('models/{}.obj'.format(model_name), 'r') as f:
        vectorizer = pickle.load(f)

    # Get indices that have values
    indices = np.argwhere((df[text_variable] != '') &
                          (df[text_variable] != -999) &
                          (df[text_variable] != np.nan) &
                          (pd.notnull(df[text_variable]))).flatten()

    # Get documents
    docs = df.loc[indices, text_variable]

    # Strip HTML
    docs = docs.apply(func = strip_html, args = (text_variable,))

    # Vectorize documents
    X = vectorizer.transform(docs)

    # Predict NaiveBayes and add as new column to 'df'
    df['NB' + text_variable] = pd.Series(index = indices, data = nb.predict_proba(X)[:,1])

    return df


def vectorize_docs(docs):
    porter = PorterStemmer()
    vectorizer = CountVectorizer(stop_words = stopwords.words('english'),
                                 preprocessor = porter.stem,
                                 tokenizer = word_tokenize)
    return vectorizer.fit_transform(docs), vectorizer

if __name__ == "__main__":
    # Read
    train_df, test_df = load_data_and_make_train_test_split('data/data.json', 0.2)

    # Create target variable
    train_df = make_target_variable(train_df)
    test_df = make_target_variable(test_df)

    # Train Naive Bayes
    # nb_model, vectorizer, train_df = train_naive_bayes(train_df,
    #                                              text_variable = 'org_desc',
    #                                              target_variable = 'fraud')
    #
    # nb_model, vectorizer, train_df = train_naive_bayes(train_df,
    #                                              text_variable = 'description',
    #                                              target_variable = 'fraud')
    #
    # test_df = add_naive_bayes_prediction(test_df,
    #                                      text_variable = 'description',
    #                                      target_variable = 'fraud')
    #
    # test_df = add_naive_bayes_prediction(test_df,
    #                                      text_variable = 'description',
    #                                      target_variable = 'fraud')

    # Replace NaN with -999
    replace_nan(df)

    # Categorical encode
    for column in string_type_categorical_columns:
        train_df, test_df = category_average_encoding(train_df, test_df, column)



    # Predict Naive Bayes
    # predict_naive_bayes(df,
    #                     text_variable = 'org_desc',
    #                     target_variable = 'fraud')
