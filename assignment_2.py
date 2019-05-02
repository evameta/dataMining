"""
Hello
"""
import csv
import random

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


def count_missing_values():

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

    return

if __name__ == '__main__':
    count_missing_values()

