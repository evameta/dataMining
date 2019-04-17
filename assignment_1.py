"""
Assignment 1
"""
import pandas as pd
from datetime import timedelta

from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


LOOK_BACK = 8


def load_data():
    """
    Load raw data from the file into a pandas dataframe

    :return: dataframe containing raw data
    """
    headers = ['row', 'id', 'time', 'variable', 'value']
    data_types = [int, str, str, str, float]

    with open('data.csv') as file_:
        data = pd.read_csv(file_, skiprows=1, names=headers, parse_dates=['time'], dtype=dict(zip(headers, data_types)))

    data.set_index(keys=['row'], drop=True, inplace=True)
    data['date'] = data['time'].map(lambda x: x.date())

    return data


def mood_dataframe():
    """
    Initialised empty dataframe with the necessary column names, depending on how far we are planning to look back

    :return: empty dataframe with correct column names
    """
    raw_columns = ['id', 'date', 'output']

    for day in range(LOOK_BACK):
        raw_columns += 'mean_{n};min_{n};max_{n}'.format(n=day).split(';')

    return pd.DataFrame(columns=raw_columns), raw_columns


def get_values(data, key_id, key_date, delta):

    key = (key_id, key_date - timedelta(days=delta))

    try:
        return data.get_group(key)['value'], True
    except KeyError:
        return None, False


def prepare_mood_data(raw_data):
    """
    Prepares the raw data for mood processing in a way that facilitates further actions. We move the data from rows
    into separate columns.

    :param raw_data:    dataframe containing the raw data for mood processing
    :return:            dataframe with pivoted data
    """
    try:
        with open('mood.csv') as file_:
            return pd.read_csv(file_, parse_dates=['date'])
    except IOError:
        pass

    mood = raw_data.groupby(by=['id', 'date'])
    mood_final, columns = mood_dataframe()

    for key, values in mood:

        results = list(key)
        key_id, key_date = key

        tomorrow, available = get_values(mood, key_id, key_date, -1)
        if available:
            results.append(tomorrow.mean())
        else:
            continue

        for day in range(LOOK_BACK):

            values, available = get_values(mood, key_id, key_date, day)

            if available:
                results += [values.mean(), values.min(), values.max()]
            else:
                results = None
                break

        if results:
            mood_final = mood_final.append(dict(zip(columns, results)), ignore_index=True)

    mood_final.to_csv('mood.csv', index=False)
    return mood_final


def describe_period(data, period):
    """
    Add columns to the dataframe describing the min, max, and mean behaviour of a relative period of time,
    i.e. if period = 5, we look at all the days between today and today - 5 days.

    :param data:      dataframe containing the raw data for the relevant period
    :param period:    integer representing the amount of days we look back on
    :return:          input dataframe with additional columns describing mean, min, and max behaviour for the period
                      of time in question.
    """
    mean_cols = ['mean_{}'.format(n) for n in range(period)]
    min_cols = ['min_{}'.format(n) for n in range(period)]
    max_cols = ['max_{}'.format(n) for n in range(period)]

    data['mean_0{}'.format(period)] = data[mean_cols].mean(axis=1)
    data['min_0{}'.format(period)] = data[min_cols].mean(axis=1)
    data['max_0{}'.format(period)] = data[max_cols].mean(axis=1)

    data['diff_0{}'.format(period)] = data['mean_{}'.format(period)] - data['mean_0']

    return data


def enrich_data(data):
    """

    :param data:
    :return:
    """
    mood = prepare_mood_data(data.loc[data['variable'] == 'mood'])

    for period in [1, 2, 3, 5, 7]:
        mood = describe_period(mood, period)

    mood = pd.concat([pd.get_dummies(mood['date'].apply(lambda x: x.weekday()), prefix='weekday'), mood], axis=1)

    X = mood.drop(['output', 'id', 'date'], axis=1)
    X = X.fillna(0)
    y = mood['output'].astype(int)

    return X, y


def preprocess(method='plain'):
    """

    :param method:
    :return:
    """
    data = load_data()
    if method == 'plain':
        return data
    elif method == 'enriched':
        return enrich_data(data)
    else:
        raise RuntimeError('Incorrect pre-processing method defined')


def svm_classify():
    """

    :return:
    """
    X, y = preprocess(method='enriched')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    print(X_train.shape)

    svclassifier = svm.SVC(kernel='rbf', C=1.0, gamma=0.2)
    svclassifier.fit(X_train, y_train)

    y_pred = svclassifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    svm_classify()
