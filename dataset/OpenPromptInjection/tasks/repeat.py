import csv
import os

import datasets


def get_repeat(split='train'):
    # raw_data = Alpaca()
    # data = raw_data.as_dataset(split=split)

    data_file = {"train": "./data/repeat.json"}
    data = datasets.load_dataset("json", data_files=data_file)
    data = data[split]

    return data
