"""
Assignment 1
"""
import pandas as pd
from datetime import timedelta


LOOK_BACK = 8


def load_data():

    headers = ['row', 'id', 'time', 'variable', 'value']
    data_types = [int, str, str, str, float]

    with open('data.csv') as file_:
        data = pd.read_csv(file_, skiprows=1, names=headers, parse_dates=['time'], dtype=dict(zip(headers, data_types)))

    data.set_index(keys=['row'], drop=True, inplace=True)
    data['date'] = data['time'].map(lambda x: x.date())

    return data


def mood_dataframe():

    raw_columns = ['id', 'date']

    for day in range(LOOK_BACK):
        raw_columns += 'mean_{n};min_{n};max_{n}'.format(n=day).split(';')

    return pd.DataFrame(columns=raw_columns), raw_columns


def prepare_mood_data(raw_data):

    try:
        with open('mood.csv') as file_:
            return pd.read_csv(file_)
    except IOError:
        pass

    mood = raw_data.groupby(by=['id', 'date'])
    mood_final, columns = mood_dataframe()

    for key, values in mood:

        results = list(key)
        key_id, key_date = key

        for day in range(LOOK_BACK):

            look_back = (key_id, key_date - timedelta(days=day))

            try:
                values = mood.get_group(look_back)['value']
                results += [values.mean(), values.min(), values.max()]
            except KeyError:
                results += [None, None, None]

        mood_final = mood_final.append(dict(zip(columns, results)), ignore_index=True)

    return mood_final


def enrich_mood_data():

    return


def describe_period(data, period):

    mean_cols = ['mean_{}'.format(n) for n in range(period)]
    min_cols = ['min_{}'.format(n) for n in range(period)]
    max_cols = ['max_{}'.format(n) for n in range(period)]

    data['mean_0{}'.format(period)] = data[mean_cols].mean(axis=1)
    data['min_0{}'.format(period)] = data[min_cols].mean(axis=1)
    data['max_0{}'.format(period)] = data[max_cols].mean(axis=1)

    data['diff_0{}'.format(period)] = data['mean_{}'.format(period)] - data['mean_0']

    return data


def main():

    data = load_data()
    mood = prepare_mood_data(data.loc[data['variable'] == 'mood'])

    for period in [1, 2, 3, 5, 7]:
        mood = describe_period(mood, period)

    print(mood[:30])


if __name__ == '__main__':
    main()
