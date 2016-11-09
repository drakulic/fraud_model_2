import pandas as pd
import numpy as np
import dill as pickle
from model import strip_html, parse_ticket_types_information, get_y, get_x
from sklearn.metrics import classification_report
from pymongo import MongoClient


with open("descriptionNaiveBayesModel.obj") as a:
    descNBmodel = pickle.load(a)
with open("descriptionVectorizer.obj") as b:
    descNBvec = pickle.load(b)
with open("org_descNaiveBayesModel.obj") as c:
    orgdNBmodel = pickle.load(c)
with open("org_descVectorizer.obj") as d:
    orgdNBvec = pickle.load(d)
with open("vanilla_rf_model.pkl") as e:
    rfmodel = pickle.load(e)
with open("countrydict.obj") as f:
    countrydict = pickle.load(f)
with open("currencydict.obj") as g:
    currencydict = pickle.load(g)
with open("email_domaindict.obj") as h:
    email_domaindict = pickle.load(h)
with open("listeddict.obj") as i:
    listeddict = pickle.load(i)
with open("org_historydict.obj") as j:
    org_historydict = pickle.load(j)
with open("payout_typedict.obj") as k:
    payouttypedict = pickle.load(k)
with open("venue_countrydict.obj") as l:
    venue_countrydict = pickle.load(l)
with open("venue_statedict.obj") as m:
    venue_statedict = pickle.load(m)


def load_single_row():
    df = pd.read_json('dftest.json')
    return df.loc[np.random.choice(df.shape[0], 1)]

def process_observation(observation):
    observation = observation.fillna(-999)
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
    observation['fraud'] = observation.acct_type.apply(lambda acct_type: account_class_dict[acct_type])
    observation.drop(['acct_type'], axis=1, inplace=True)
    observation = parse_ticket_types_information(observation)
    try:
        observation['countryencoded'] = observation.country.apply(lambda country: countrydict[country])
    except:
        observation['countryencoded'] = -999
    try:
        observation['currencyencoded'] = observation.currency.apply(lambda currency: currencydict[currency])
    except:
        observation['currencyencoded'] = -999
    try:
        observation['email_domainencoded'] = observation.email_domain.apply(lambda email_domain: email_domaindict[email_domain])
    except:
        observation['email_domainencoded'] = -999
    try:
        observation['listedencoded'] = observation.listed.apply(lambda listed: listeddict[listed])
    except:
        observation['listedencoded'] = -999
    try:
        observation['payout_typeencoded'] = observation.payout_type.apply(lambda payout_type: payouttypedict[payout_type])
    except:
        observation['payout_typeencoded'] = -999
    try:
        observation['venue_countryencoded'] = observation.venue_country.apply(lambda venue_country: venue_countrydict[venue_country])
    except:
        observation['venue_countryencoded'] = -999
    try:
        observation['venue_stateencoded'] = observation.venue_state.apply(lambda venue_state: venue_statedict[venue_state])
    except:
        observation['venue_stateencoded'] = -999
    try:
        observation['org_history'] = observation.org_name.apply(lambda organization: org_historydict[organization])
    except:
        observation['org_history'] = 0
    # observation['org_history'] = 1
    observation = observation[['body_length', 'channels', 'delivery_method', 'fb_published',
              'gts', 'has_header', 'has_analytics', 'has_logo', 'name_length',
              'org_facebook', 'org_twitter', 'show_map', 'user_age',
              'average_ticket_price', 'minimum_ticket_price',
              'maximum_ticket_price', 'maximum_potential_revenue', 'countryencoded',
              'currencyencoded', 'email_domainencoded', 'listedencoded', 'payout_typeencoded',
              'venue_countryencoded', 'venue_stateencoded', 'org_desc', 'description',
              'org_history', 'fraud']]
    return observation


def predict(X, rfmodel):
    return rfmodel.predict(X)


def process_description(observation):
    text = observation.description.apply(func=strip_html, args=('description',))
    vector = descNBvec.transform(text)
    observation['NBdescription'] = descNBmodel.predict_proba(vector)[0][1]
    return observation


def process_org_desc(observation):
    text = observation.description.apply(func=strip_html, args=('org_desc',))
    vector = orgdNBvec.transform(text)
    observation['NBorg_desc'] = orgdNBmodel.predict_proba(vector)[0][1]
    return observation


def process_text_features(observation):
    observation = process_description(observation)
    observation = process_org_desc(observation)
    observation = observation.fillna(-999)
    observation.drop(['org_desc', 'description'], axis=1, inplace=True)
    return observation


def fill_none_with_negative_999(item):
    if item == '':
        return -999
    elif item == None:
        return -999
    else:
        return item


def correct_categorical_string_type_columns(observation):
    string_type_categorical_columns = ['country', 'currency', 'email_domain',
                                       'listed', 'payout_type', 'venue_country',
                                       'venue_state']
    for column in string_type_categorical_columns:
        observation[column] = observation[column].apply(fill_none_with_negative_999)
    return observation


def store_information_into_mongo(observation, prediction):
    client = MongoClient()
    db = client['fraud_prediction']
    tab = db['predictions']
    observation['prediction'] = prediction
    n = observation.shape[0]
    for row in xrange(n):
        tab.insert_one(observation.loc[row].to_dict())



if __name__ == "__main__":
    df = load_single_row()
    # df = pd.read_json('dftest.json')
    observation = df.copy()
    try:
        observation.drop(['countryencoded', 'currencyencoded',
                 'email_domainencoded', 'listedencoded', 'payout_typeencoded',
                 'venue_countryencoded', 'venue_stateencoded'], axis=1, inplace=True)
    except:
        pass
    observation = observation.reset_index()
    # observation.drop('level_0', axis=1, inplace=True)
    observation = correct_categorical_string_type_columns(observation)
    observation = process_observation(observation)
    observation = process_text_features(observation)
    y = get_y(observation, 'fraud')
    observation, X = get_x(observation)
    prediction = predict(X, rfmodel)
    # print classification_report(y, prediction)
    # store_information_into_mongo(df, prediction)
