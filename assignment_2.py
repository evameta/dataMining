"""
Hello
"""
import csv
import random
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

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
    corr_matrix()