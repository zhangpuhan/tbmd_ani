import pandas as pd
import os
import torch
import neurochem
import aev
import natsort

def locate_files(file_path):
    filenames_a = []
    for _, __, files in os.walk(file_path):
        for filename in files:
            if filename.endswith(".csv"):
                filenames_a.append(file_path + "/" + filename)
    filenames_a = natsort.natsorted(filenames_a)

    return filenames_a


filenames = locate_files("test")
print(filenames)


def concat_all_files(files, range_low, range_high):
    data = pd.read_csv(files[range_low], header=None)
    for i in range(range_low + 1, range_high):
        data = pd.concat([data, pd.read_csv(files[i], header=None)], axis=0, ignore_index=True)
        print(str(i) + " processed")
    return data


train_data = concat_all_files(filenames, 0, len(filenames))
train_data.to_csv("test.csv", header=None, index=False)


