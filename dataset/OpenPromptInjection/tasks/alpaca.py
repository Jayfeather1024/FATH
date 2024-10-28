import csv
import os

import datasets
import random


# _CITATION = "Unknown"

# _DESCRIPTION = "The Stanford Alpaca Dataset."

# _HOMEPAGE = "Unknown"

# _LICENSE = "Unknown"

# _URL = "Unknown"


# class Alpaca(datasets.GeneratorBasedBuilder):
#     """SST-2 dataset."""

#     VERSION = datasets.Version("1.0.0")

#     def _info(self):
#         features = datasets.Features(
#             {
#                 "idx": datasets.Value("int32"),
#                 "sentence": datasets.Value("string"),
#                 "label": datasets.Value("string"),
#             }
#         )
#         return datasets.DatasetInfo(
#             description=_DESCRIPTION,
#             features=features,
#             homepage=_HOMEPAGE,
#             license=_LICENSE,
#             citation=_CITATION,
#         )

#     def _split_generators(self, dl_manager):
#         dl_dir = dl_manager.download_and_extract(_URL)
#         return [
#             datasets.SplitGenerator(
#                 name=datasets.Split.TRAIN,
#                 gen_kwargs={
#                     "file_paths": "/data/jiongxiao_wang/prompt_injection_defense/custom_dataset",
#                     "data_filename": "alpaca_data_with_input.json",
#                 },
#             ),
#         ]

#     def _generate_examples(self, file_paths, data_filename):
#         for file_path in file_paths:
#             filename = os.path.basename(file_path)
#             if filename == data_filename:
#                 print(data_filename)
#                 print(filename)
#                 exit()
#                 with open(file_path, encoding="utf8") as f:
#                     reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
#                     for idx, row in enumerate(reader):
#                         yield idx, {
#                             "idx": row["index"] if "index" in row else idx,
#                             "sentence": row["sentence"],
#                             "label": int(row["label"]) if "label" in row else -1,
#                         }



def get_alpaca(split='train', seed=None):
    # raw_data = Alpaca()
    # data = raw_data.as_dataset(split=split)

    data_file = {"train": "./data/alpaca_data_with_input_train.json", "test": "./data/alpaca_data_with_input_test_icl.json"}
    data = datasets.load_dataset("json", data_files=data_file)
    data = data[split]
    if seed is not None:
        data = data.shuffle(seed=seed)
    return data
