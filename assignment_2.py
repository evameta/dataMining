"""
Hello
"""
import csv
import random

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


random_sample()

