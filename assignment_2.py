"""
Hello
"""
import csv
import random
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from operator import add

TRAIN = 'training_set_VU_DM.csv'
SAMPLE = 'training_sample.csv'
LEN_TRAIN = 4958347


def random_sample(k=150):
    """
    Random Sample
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


def run(reload=False):

    if reload:
        random_sample()


def count_missing_values(calculate=True, plot=False):

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
    plt.legend(fancybox=True, loc=1)
    plt.savefig('click_booked_pos.pdf', bbox_inches='tight')
    plt.show()

    # plot hist of positions of booked hotels where ranking was random vs. ordered
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
    plt.legend(fancybox=True, loc=1)
    plt.savefig('random_booked_pos.pdf', bbox_inches='tight')
    plt.show()

def clean_data():

    df = pd.read_csv("data/" + SAMPLE)

    # Set missing competitor data to 0.
    filter_col = [col for col in df if col.startswith('comp')]
    df[filter_col] = df[filter_col].fillna(0)

    # Remove column if not useful or missing data.
    df = df.drop(columns=['date_time', 'srch query affinity score'])
    df.groupby()


if __name__ == '__main__':
    clean_data()
