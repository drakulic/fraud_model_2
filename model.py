import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import scale
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_score
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from sklearn.preprocessing import StandardScaler
from unidecode import unidecode
import scipy.stats as scs
from sklearn.cluster import DBSCAN
import dill as pickle
from sklearn.ensemble.partial_dependence import plot_partial_dependence, partial_dependence
import matplotlib.pyplot as plt
import random
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB



string_type_categorical_columns = ['country', 'currency', 'email_domain',
                                   'listed', 'payout_type', 'venue_country',
                                   'venue_state']


def load_data_and_make_train_test_split(path, percentage):
    df = pd.read_json(path)
    df.columns = [c.lower().replace(' ', '_').replace('?', '').replace("'", '') for c in df.columns]
    mask = np.random.choice(df.shape[0], size=int(df.shape[0]*percentage), replace=False)
    dftest = df.loc[mask]
    newmask = set(range(df.shape[0])) - set(mask)
    dftrain = df.loc[newmask]
    dftest = dftest.reset_index()
    dftrain = dftrain.reset_index()
    return dftrain, dftest


def process_dataframe(df, dftest):
    df = df.fillna(-999)
    df, dftest = transform_variables(df, dftest)
    df = drop_unnecessary_features(df)
    return df, dftest


def add_org_history_as_a_feature(df):
    org_history_dict = df.org_name.value_counts().to_dict()
    df['org_history'] = df.org_name.apply(lambda organization: org_history_dict[organization])
    with open('{}dict.obj'.format('org_history'), 'w+') as f:
        pickle.dump(org_history_dict, f)
    return df


def transform_variables(df, dftest):
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
    df = parse_ticket_types_information(df)
    for column in string_type_categorical_columns:
        df, dftest = category_average_encoding(df, dftest, column)
    df = add_org_history_as_a_feature(df)
    return df, dftest


def parse_ticket_types_information(df):
    average_ticket_price = []
    minimum_ticket_price = []
    maximum_ticket_price = []
    total_sales_to_date = []
    maximum_potential_revenue = []
    percentage_sold_to_date = []
    percentage_revenue_to_date = []
    for event_ticket_types in df.ticket_types:
        if len(event_ticket_types) > 0:
            cost = np.array([])
            qty_sold = np.array([])
            qty_total = np.array([])
            for ticket_type_info in event_ticket_types:
                cost = np.append(cost, ticket_type_info['cost'])
                qty_sold = np.append(qty_sold, ticket_type_info['quantity_sold'])
                qty_total = np.append(qty_total, ticket_type_info['quantity_total'])
            average_ticket_price.append(np.mean(cost))
            minimum_ticket_price.append(np.min(cost))
            maximum_ticket_price.append(np.max(cost))
            total_sales_to_date.append(np.sum(cost*qty_sold))
            maximum_potential_revenue.append(np.sum(cost*qty_total))
            percentage_sold_to_date.append(np.sum(qty_sold)/np.sum(qty_total))
            percentage_revenue_to_date.append(np.sum(cost*qty_sold)/np.sum(cost*qty_total))
        else:
            average_ticket_price.append(-999)
            minimum_ticket_price.append(-999)
            maximum_ticket_price.append(-999)
            total_sales_to_date.append(-999)
            maximum_potential_revenue.append(-999)
            percentage_sold_to_date.append(-999)
            percentage_revenue_to_date.append(-999)
    df['average_ticket_price'] = average_ticket_price
    df['minimum_ticket_price'] = minimum_ticket_price
    df['maximum_ticket_price'] = maximum_ticket_price
    df['total_sales_to_date'] = total_sales_to_date
    df['maximum_potential_revenue'] = maximum_potential_revenue
    df['percentage_sold_to_date'] = percentage_sold_to_date
    df['percentage_revenue_to_date'] = percentage_revenue_to_date
    return df


def drop_unnecessary_features(df):
    # the line below removes acct_type which is where the 'fraud' column came from
    # the line below REMOVES all non-numeric columns temporarily in order to build a basic model
    df = df[['body_length', 'channels', 'delivery_method', 'fb_published',
              'gts', 'has_header', 'has_analytics', 'has_logo', 'name_length',
              'org_facebook', 'org_twitter', 'show_map', 'user_age',
              'average_ticket_price', 'minimum_ticket_price',
              'maximum_ticket_price', 'maximum_potential_revenue', 'countryencoded',
              'currencyencoded', 'email_domainencoded', 'listedencoded', 'payout_typeencoded',
              'venue_countryencoded', 'venue_stateencoded', 'org_desc', 'description',
              'org_history', 'fraud']]
    return df


def get_x(df):
    X = df.values
    return df, X


def get_y(df, target):
    y = df.pop(target).values
    return y


def gridsearch(X_train, y_train):
    paramgrid = {'n_estimators': [11, 12, 13],
                 'max_features': ['auto'],
                 'min_samples_split': [1, 2, 3],
                 'bootstrap': [True]}
    gridsearch = GridSearchCV(RandomForestClassifier(n_jobs=-1),
                              paramgrid,
                              n_jobs=-1,
                              verbose=10,
                              cv=10)
    gridsearch.fit(X_train, y_train)
    best_model = gridsearch.best_estimator_
    print(best_model)
    print(gridsearch.best_score_)
    return best_model, gridsearch


def fit_model(best_model, X_train, X_test, y_train, y_test):
    clf = best_model
    clf.fit(X_train, y_train)
    print("this is the score for the best model")
    print(clf.score(X_test, y_test))
    return clf


def evaluate_model(best_model, X_train, X_test, y_train, y_test):
    clf = best_model
    print(np.mean(cross_val_score(clf, X_train, y_train, cv=10, n_jobs=-1, verbose=10)))
    clf.fit(X_train, y_train)
    return clf


def view_classification_report(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


def view_feature_importances(df, model):
    columns = df.columns
    features = model.feature_importances_
    featimps = []
    for column, feature in zip(columns, features):
        featimps.append([column, feature])
    print(pd.DataFrame(featimps, columns=['Features',
                       'Importances']).sort_values(by='Importances',
                                                   ascending=False))


def balance_classes(method, X, y):
    sm = method
    X, y = sm.fit_sample(X, y)
    return X, y


def cluster_based_oversampling(X_train, y_train):
    db = DBSCAN()
    db.fit(X_train)
    X_list = []
    y_list = []
    for cluster in np.unique(db.labels_):
        X_samp = X_train[db.labels_ == cluster]
        y_samp = y_train[db.labels_ == cluster]
        X_samp, y_samp = balance_classes(SMOTE(), X_train, y_train)
        X_list.append(X_samp)
        y_list.append(y_samp)
    return np.vstack(X_list), np.hstack(y_list)


def save_model(model, filename):
    with open(filename, 'w') as f:
        pickle.dump(model, f)


def load_model(filename):
    with open(filename) as f_un:
        model = pickle.load(f_un)
    return model


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
    # print docs
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
    # print df[['NB' + text_variable, target_variable]]
    # Pickle models
    model_name = text_variable + 'NaiveBayesModel'
    with open('{}.obj'.format(model_name), 'w+') as f:
        pickle.dump(nb, f)
    model_name = text_variable + 'Vectorizer'
    with open('{}.obj'.format(model_name), 'w+') as f:
        pickle.dump(vectorizer, f)
    return nb, vectorizer, df


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


def get_fraud_percentage(df):
    return float(np.sum(df.iloc[:,1])) / len(df)


def strip_html(row, column):
    # Strip HTML from 'column'
    return BeautifulSoup(row, "lxml").get_text()


def vectorize_docs(docs):
    porter = PorterStemmer()
    vectorizer = CountVectorizer(stop_words = stopwords.words('english'),
                                 preprocessor = porter.stem,
                                 tokenizer = word_tokenize)
    return vectorizer.fit_transform(docs), vectorizer


if __name__ == "__main__":
    dftrain, dftest = load_data_and_make_train_test_split('data/data.json', .2)
    dftrain, dftest = process_dataframe(dftrain, dftest)
    nb_model, vectorizer, dftrain = train_naive_bayes(dftrain,
                                                 text_variable = 'org_desc',
                                                 target_variable = 'fraud')
    nb_model, vectorizer, dftrain = train_naive_bayes(dftrain,
                                                 text_variable = 'description',
                                                 target_variable = 'fraud')
    dftrain = dftrain.fillna(-999)
    dftrain.drop(['org_desc', 'description'], axis=1, inplace=True)
    y = get_y(dftrain, 'fraud')
    dftrain, X = get_x(dftrain)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    # X_train, y_train = balance_classes(RandomOverSampler(random_state=42), X_train, y_train)
    X_train, y_train = cluster_based_oversampling(X_train, y_train)
    # model, gridsearch = gridsearch(X_train, y_train)
    # print("best gridsearch score is above")
    model = RandomForestClassifier()
    model = fit_model(model, X_train, X_test, y_train, y_test)
    model = evaluate_model(model, X_train, X_test, y_train, y_test)
    print("score on data it's never seen")
    print(model.score(X_test, y_test))
    view_classification_report(model, X_test, y_test)
    # model = load_model('rf_model_with_text_features.pkl')
    view_feature_importances(dftrain, model)
    # print("fit on all training data")
    # model = fit_model(model, X, X_test, y, y_test)
    save_model(model, 'vanilla_rf_model.pkl')
    # predictions = make_predictions(model)
    dftrain.to_json('dftrain.json')
    dftest.to_json('dftest.json')
