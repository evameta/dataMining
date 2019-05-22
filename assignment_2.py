"""
Hello
"""
import csv
import random
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from operator import add

TRAIN = 'training_set_VU_DM.csv'
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
    plt.xticks(ticks=[i for i in range(len(columns))], labels=columns, rotation='vertical')
    plt.yticks(ticks=[i for i in range(len(columns))], labels=columns)
    plt.colorbar().set_label('correlation',fontsize=10)
    plt.savefig('corr_matrix.pdf',  bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    count_missing_values(calculate=False, plot=True)
