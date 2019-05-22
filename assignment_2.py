"""
Hello
"""
import csv
import random
import pandas as pd
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import dump_svmlight_file
from operator import add
from sklearn.preprocessing import MinMaxScaler

TRAIN = 'training_set_VU_DM.csv'
SAMPLE = 'training_sample.csv'
LEN_TRAIN = 4958347


def random_sample(k=150):
    """
    Generate a random sample from the training data_set and save to new .csv file
    """
    with open('data/' + TRAIN) as file_, open('data/training_sample.csv', mode='w', newline='') as out_:
        reader = csv.reader(file_, None)
        writer = csv.writer(out_)

        headers = next(reader)
        writer.writerow(headers)

        random_lines = random.sample(range(LEN_TRAIN), k=k)

        for i, line in enumerate(reader):
            if i in random_lines:
                writer.writerow(line)


def count_missing_values(calculate=True, plot=False):
    """
    Counts the
    """
    if calculate:
        with open('data/' + TRAIN) as file_:
            reader = csv.reader(file_)

            headers = next(reader)
            none_values = [0 for _ in headers]

            for i, line in enumerate(reader):
                none_values = list(map(add, none_values, [elem == 'NULL' for elem in line]))

                if i % 10000 == 0:
                    print(i, 'iterations complete!')

        with open('data/none_values.csv', 'w', newline='') as out_:
            writer = csv.writer(out_)
            writer.writerow(headers)
            writer.writerow(none_values)

    else:
        with open('data/none_values.csv') as file_:
            reader = csv.reader(file_)
            headers = next(reader)
            none_values = [int(elem) for elem in next(reader)]

    if plot:
        ordered = sorted(range(len(none_values)), key=lambda ind: -none_values[ind])
        headers = [headers[o] for o in ordered if none_values[o] > 0]
        none_values = [none_values[o] for o in ordered if none_values[o] > 0]
        print(none_values)

        plt.bar(headers, none_values)
        plt.xticks(ticks=[i for i in range(len(headers))], labels=headers, rotation='vertical')
        plt.yticks(ticks=range(0, max(none_values), 500000))
        plt.savefig('none_values.pdf', bbox_inches="tight")

    return


def corr_matrix():
    df = pd.read_csv("data/training_set_VU_DM.csv")
    #df = pd.read_csv("data/training_sample.csv")
    mpl.rcParams.update({'font.size': 5})
    columns = list(df)
    ax = plt.imshow(df.corr(), cmap='hot', interpolation='nearest')
    plt.xticks(ticks=[i-1 for i in range(len(columns))], labels=columns, rotation='vertical')
    plt.yticks(ticks=[i-1 for i in range(len(columns))], labels=columns)
    plt.colorbar().set_label('correlation',fontsize=10)
    plt.savefig('corr_matrix.pdf',  bbox_inches="tight")
    plt.show()



def statistics_data():
    df = pd.read_csv("data/" + TRAIN)
    df.drop(columns=['date_time'])
    df_none = pd.read_csv("data/none_values.csv").drop(columns=['date_time']).transpose()
    std_dev = df._get_numeric_data().std(axis=0)
    print(std_dev)
    mean = df._get_numeric_data().mean(axis=0)
    print(mean)
    stats = {'mean': mean, 'std_dev': std_dev} # 'missing_values': df_none}
    df_stats = pd.DataFrame(stats)
    df_stats = pd.concat([df_stats, df_none], axis=1, sort=False)
    df_stats.rename(columns={0: 'missing_values'}, inplace=True)
    df_stats['missing_values'] = df_stats['missing_values'].div(LEN_TRAIN)
    df_stats.to_csv('data/stats.csv')
    print(df_stats)


def plot_data():
    # Plots:
    #   1. The normalized percentage per search position of clicked and booked properties.
    #   2. The normalized percentage of booked properties per search position for random and ordered properties.

    # 1.
    df = pd.read_csv("data/" + TRAIN)
    pos_clicked = df['position'].where(df['click_bool'] == 1)
    pos_booked = df['position'].where(df['booking_bool'] == 1)
    pos_clicked = pos_clicked.value_counts(normalize=True)
    pos_booked = pos_booked.value_counts(normalize=True)
    fig, ax = plt.subplots(figsize=[15, 5])
    x = np.arange(1, 41)
    ax.bar(x - 0.2, pos_clicked[x], width=0.4, label='Clicked', color='red')
    ax.bar(x + 0.2, pos_booked[x], width=0.4, label='Booked', color='green')
    ax.set_xticks(x)
    ax.set_yticklabels(['{:.1%}'.format(i) for i in np.linspace(0, 0.2, 9)])
    ax.set_ylabel('Percentage clicked/booked')
    ax.set_xlabel('Position')
    plt.legend(fancybox=True, loc=9)
    plt.savefig('click_booked_pos.pdf', bbox_inches='tight')
    plt.show()

    # 2.
    pos_random = df['position'].where(df['booking_bool'] == 1).where(df['random_bool'] == 1)
    pos_ordered = df['position'].where(df['booking_bool'] == 1).where(df['random_bool'] == 0)
    pos_random = pos_random.value_counts(normalize=True)
    pos_random.index = pos_random.index.map(int)
    pos_ordered = pos_ordered.value_counts(normalize=True)
    pos_ordered.index = pos_ordered.index.map(int)
    fig, ax = plt.subplots(figsize=[15, 5])
    x = np.arange(1, 41)
    ax.bar(x - 0.2, pos_random[x], width=0.4, label='Random', color='red')
    ax.bar(x + 0.2, pos_ordered[x], width=0.4, label='Ordered', color='green')
    ax.set_xticks(x)
    ax.set_yticklabels(['{:.1%}'.format(i) for i in np.linspace(0, 0.2, 9)])
    ax.set_xlabel('Position')
    ax.set_ylabel('Percentage random/ordered')
    plt.legend(fancybox=True, loc=9)
    plt.savefig('random_booked_pos.pdf', bbox_inches='tight')
    plt.show()


def clean_data(reload=True):

    if reload:
        random_sample(150)

    df = pd.read_csv("data/" + SAMPLE)

    # The distance between customer and hotel is set to -1.
    df["orig_destination_distance"] = df['orig_destination_distance'].fillna(-1)

    # Set missing competitor data to -1.
    filter_col = [col for col in df if col.startswith('comp')]
    df[filter_col] = df[filter_col].fillna(-1)

    # Replace NaN in prop_review_score and prop_location_score2 with minimum of search. Few have no values still --> 0
    df["prop_review_score"] = df.groupby("srch_id")["prop_review_score"].transform(lambda x: x.fillna(x.min()))
    df["prop_review_score"] = df['prop_review_score'].fillna(0)

    df["prop_location_score2"] = df.groupby("srch_id")["prop_location_score2"].transform(lambda x: x.fillna(x.min()))
    df["prop_location_score2"] = df["prop_location_score2"].fillna(0)

    # User data according to matching/mismatching with historical data.
    df["starrating_diff"] = (df['visitor_hist_starrating'].fillna(0) - df['prop_starrating'].fillna(0)).abs()
    df["usd_diff"] = np.log10((df['visitor_hist_adr_usd'].fillna(0) - df['price_usd'].fillna(0)).abs())
    # Remove column if not useful or missing data.
    df = df.drop(columns=['date_time', 'gross_bookings_usd', 'srch_query_affinity_score', 'visitor_hist_starrating',
                          'visitor_hist_adr_usd'])

    # Normalize price_usd.
    norm_columns = ['price_usd', 'prop_location_score1', 'prop_location_score2']
    df = normalise_columns(df, *norm_columns)
    df = df.replace([np.inf, -np.inf], np.nan)
    print(df.isna().sum())
    df = df.fillna(-1)

    # Define target in training set.
    df['target'] = np.fmax((5 * df['booking_bool']).values, df['click_bool'].values)

    # Save to file.
    df.to_csv('data/cleaned_data.csv')


def normalise_columns(data, *columns):
    """
    Normalises the values in a column of the dataframe passed, indicated by a string containing the name of the
    column(s) selected.
    """
    scalar = MinMaxScaler()
    for column in columns:
        scaled = scalar.fit_transform(data[[column]].values.astype('float'))
        data[column] = pd.Series(scaled[:, 0])

    return data


def bucketing_column(data, column, bucket_splits=None, bucket_size=None):
    """
    Transposes a column containing continuous data into multiple columns, with binary values indicating the buckets
    into which the data from the first column fits. Passing the parameter 'bucket_splits' lets the user manually set
    the ranges into which the splits are to occur (this is just a list of integers). On the other hand, passind the
    parameter bucket_size (simply an integer), lets the function determine the max of the column, and splits the data
    into buckets (starting at 0) of size 'bucket_size'.
    """
    try:
        bucket_splits = sorted(bucket_splits)
    except TypeError:
        column_max = data[column].max()
        bucket_splits = [50 * i for i in range(math.ceil(column_max/bucket_size) + 1)]

    for lower, upper in zip(bucket_splits[:-1], bucket_splits[1:]):
        new_column = '{col}_{lo}_{hi}'.format(col=column, lo=lower, hi=upper)
        data[new_column] = (lower < data[column]) & (data[column] <= upper)

    return data


def svmlight_file():
    df = pd.read_csv('data/' + 'cleaned_data.csv')
    file = 'data/svm_file'
    target_columns = ['booking_bool', 'click_bool']
    X = np.array(df.drop(columns=target_columns))
    y = np.array(df['booking_bool'])
    dump_svmlight_file(X, y, file, multilabel=True)


if __name__ == '__main__':
    clean_data()
    svmlight_file()
