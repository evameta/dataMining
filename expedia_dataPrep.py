import logging
import math
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler

FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('dataPrep')
logger.setLevel(logging.DEBUG)

TRAIN = 'training_set_VU_DM.csv'
SAMPLE = 'training_sample.csv'

data_set = pd.read_csv('data/' + SAMPLE)
logger.info('Data loaded: {m} rows, {n} columns'.format(m=data_set.shape[0], n=data_set.shape[1]))
logger.info('Data columns: ' + ', '.join(list(data_set)))


def normalise_columns(data, *columns):
    scaler = MinMaxScaler()
    for column in columns:
        scaled = scaler.fit_transform(data[[column]].values.astype('float'))
        data[column] = pd.Series(scaled[:, 0])

    return data


def bucketing_column(data, column, bucket_splits=None, bucket_size=None):

    try:
        bucket_splits = sorted(bucket_splits)
    except TypeError:
        column_max = data[column].max()
        bucket_splits = [50 * i for i in range(math.ceil(column_max/bucket_size) + 1)]

    for lower, upper in zip(bucket_splits[:-1], bucket_splits[1:]):
        new_column = '{col}_{lo}_{hi}'.format(col=column, lo=lower, hi=upper)
        data[new_column] = (lower < data[column]) & (data[column] <= upper)

    return data

start = time.time()
data_set = bucketing_column(data_set, 'price_usd', bucket_size=50)
bucket = time.time()
logger.info('Bucketing column {col}: {time:.3f} seconds'.format(col='price_usd', time=bucket - start))

data_set = normalise_columns(data_set, 'price_usd')
logger.info('Normalising column {col}: {time:.3f} seconds'.format(col='srch_id', time=time.time() - bucket))
