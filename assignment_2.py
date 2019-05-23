"""
Hello
"""
import csv
import logging
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time
from sklearn.datasets import dump_svmlight_file
from operator import add
from sklearn.preprocessing import MinMaxScaler

FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('expedia')
logger.setLevel(logging.DEBUG)

TRAIN = 'train.csv'
SAMPLE = 'sample.csv'
LEN_TRAIN = 4958347

class DataProcessing:

    def __init__(self, data='train'):
        self.type = data
        self.file_name = 'data/in/' + self.type + '.csv'
        self.data = self.load_data()

        self.columns_to_normalise = ['price_usd', 'prop_location_score1', 'prop_location_score2', 'orig_destination_distance']
        self.columns_to_drop = ['date_time', 'gross_bookings_usd', 'srch_query_affinity_score',
                                'visitor_hist_starrating', 'visitor_hist_adr_usd', 'booking_bool', 'click_bool']

    def load_data(self):
        """
        Load data to process form input file
        """
        logger.info('Loading ' + self.type + ' data from file: ' + self.file_name)
        data = pd.read_csv(self.file_name)
        logger.info('Number of rows found in set: {n}'.format(n=len(data)))

        return data

    def preprocess(self):
        """
        Full pre-processing pipeline executed
        """
        logger.info('Starting pre-processing for ' + self.type + ' data.')
        start = time.time()

        self.fill_na_prop_review_score()
        self.fill_na_prop_location_score2()
        self.historical_user_data()

        self.normalise_column(*self.columns_to_normalise)

        self.data = self.data.replace([np.inf, -np.inf], np.nan)
        self.data = self.data.fillna(-1)

        self.make_target_column()
        self.data.replace(0, -1, inplace=True)
        self.drop_columns()

        logger.info('Preprocessing completed in {s:.3f} seconds'.format(s=time.time() - start))

        self.save_to_file()

    def fill_na_prop_review_score(self):
        """
        Fill Na values in prop_review_score column with group minimum, or zero otherwise
        """
        logger.info('Replace NaN in prop_review_score with search minimum, or 0 otherwise.')

        prs = 'prop_review_score'
        self.data[prs] = self.data.groupby('srch_id')[prs].transform(lambda x: x.fillna(x.min()))
        self.data[prs] = self.data[prs].fillna(0)

    def fill_na_prop_location_score2(self):
        """
        Fill Na values in prop_review_score2 column with group minimum, or zero otherwise
        """
        logger.info('Replace NaN in prop_location_score2 with search minimum, or 0 otherwise.')

        prs = 'prop_location_score2'
        self.data[prs] = self.data.groupby('srch_id')[prs].transform(lambda x: x.fillna(x.min()))
        self.data[prs] = self.data[prs].fillna(-1)

    def historical_user_data(self):
        """
        Combine property star rating and price with historical user data
        """
        logger.info('User data according to matching/mismatching with historical data.')

        hist_rating, rating = 'visitor_hist_starrating', 'prop_starrating'
        hist_price, price = 'visitor_hist_adr_usd', 'price_usd'

        self.data['starrating_diff'] = (self.data[hist_rating].fillna(0) - self.data[rating].fillna(0)).abs()
        self.data['usd_diff'] = np.log10((self.data[hist_price].fillna(0) - self.data[price].fillna(0)).abs())

    def normalise_column(self, *columns):
        """
        Normalise columns
        """
        scalar = MinMaxScaler()
        for column in columns:
            scaled = scalar.fit_transform(self.data[[column]].values.astype('float'))
            self.data[column] = pd.Series(scaled[:, 0])

    def make_target_column(self):
        """
        Add target column to dataframe
        """
        logger.info('Adding target column to data.' +
                    (' This is the test data, so a column of zeros.' if self.type == 'test' else ''))

        if self.type == 'test':
            self.data['target'] = 0
        else:
            self.data['target'] = np.fmax((5 * self.data['booking_bool'].values), self.data['click_bool'].values) + 1

    def drop_columns(self):
        """
        Drop unnecessary columns
        """
        logger.info('Removing columns with unnecessary or incomplete data.')

        self.data = self.data.drop(columns=self.columns_to_drop, errors='ignore')

    def bucketing_column(self, column, splits):
        """
        Convert column of continuous data into multiple buckets
        """
        column_max = math.ceil(self.data[column].max()/splits[-1]) * splits[-1]
        splits.append(column_max)

        logger.info('Bucketing column ' + column + ' based on splits: ' + ', '.join([str(i) for i in splits]))

        for lower, upper in zip(splits[:-1], splits[1:]):
            new_column = '{col}_{lo}_{hi}'.format(col=column, lo=lower, hi=upper)
            self.data[new_column] = ((lower < self.data[column]) & (self.data[column] <= upper)).astype('int')

    def save_to_file(self):
        """
        Saving processing data to csv file
        """
        file_name = 'data/out/' + self.type + '.csv'
        logger.info('Saving pre-processed data to ' + file_name)

        self.data.to_csv(file_name, index=False)


def random_sample(k=1000):
    """
    Generate a random sample from the training data_set and save to new .csv file
    """
    with open('data/in/' + TRAIN) as file_, open('data/out/sample.csv', mode='w', newline='') as out_:
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
    df = pd.read_csv('data/in/train.csv')
    #df = pd.read_csv("data/sample.csv")
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
    """
    Plots:
       1. The normalized percentage per search position of clicked and booked properties.
       2. The normalized percentage of booked properties per search position for random and ordered properties.
    """
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


def svmlight_file(data):
    """
    Create .svmlight file
    """
    df = pd.read_csv('data/out/' + data + '.csv')
    input_data = np.array(df.drop(columns=['target', 'srch_id'], errors='ignore'))
    target = np.array(df['target'])
    qid = np.array(df['srch_id'])

    file = 'data/svm/' + data + '.svmlight'
    start = time.time()
    dump_svmlight_file(input_data, target, file, multilabel=False, query_id=qid, zero_based=False)
    logger.info('SVMLIGHT file created in {s} seconds'.format(s=time.time() - start))


def validation_set():
    """
    Create training and validation file
    """
    df = pd.read_csv('data/out/train.csv')
    msk = np.random.rand(len(df)) < 0.9
    train = df[msk]
    val = df[~msk]
    train.to_csv('data/out/train_val.csv')
    val.to_csv('data/out/val.csv')


if __name__ == '__main__':
    #svmlight_file('train')
