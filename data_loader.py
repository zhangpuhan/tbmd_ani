import torch
import os
import natsort
import pandas as pd

device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage *******************************')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024**3, 1), 'GB')


def read_file_names(path):
    filenames = []
    for _, __, files in os.walk(path):
        for filename in files:
            if filename.endswith(".csv"):
                filenames.append(path + "/" + filename)
    filenames = natsort.natsorted(filenames)
    return filenames


train_x_filename = read_file_names("testdata_0403_50")
train_energy_filename = read_file_names("test_energy_0403_50")


def import_data_x(filename):
    coordinate = []
    force = []
    for i in range(0, len(filename)):
        read_file = torch.tensor(pd.read_csv(filename[i], header=None).values, device=device)
        coordinate_tensor = read_file[:, :3]
        force_tensor = read_file[:, 3:]
        coordinate.append(coordinate_tensor.unsqueeze(0))
        force.append(force_tensor.unsqueeze(0))
        print("file " + str(i) + " done.")

    coordinate_tensor = torch.cat(tuple(coordinate), 0)
    force_tensor = torch.cat(tuple(force), 0)
    return coordinate_tensor, force_tensor


def import_data_energy(filename):
    energy = []
    for i in range(0, len(filename)):
        read_file = torch.tensor(pd.read_csv(filename[i], header=None).values, device=device)
        energy.append(read_file)
        print("energy file " + str(i) + " done.")

    energy_tensor = torch.cat(tuple(energy), 0)

    return energy_tensor


energy_temp = import_data_energy(train_energy_filename)
coordinate_temp, force_temp = import_data_x(train_x_filename)

torch.save(coordinate_temp, "coordinate04032019_50.pt")
torch.save(energy_temp, "energy04032019_50.pt")
torch.save(force_temp, "force04032019_50.pt")
