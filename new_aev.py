import pandas as pd
import os
import torch
import neurochem
import aev
import natsort

torch.set_default_tensor_type('torch.DoubleTensor')

path = os.path.dirname(os.path.realpath(__file__))

print(path)

const_file = os.path.join(path, 'torchani/resources/ani-1x_8x/rHCNO-5.2R_16-3.5A_a4-8.params')

print(const_file)
consts = neurochem.Constants(const_file)

print(consts)

aev_computer = aev.AEVComputer(**consts)

print(aev_computer)


data = pd.read_csv("g0.csv", header=None)
print(data[0])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


coordinate_tensor = torch.tensor(data.values)

force = coordinate_tensor[:, 3:]
coordinate = coordinate_tensor[0:, :3]
print(coordinate.unsqueeze(0))

test_data = aev_computer.forward((consts.species_to_tensor('HHHHHHHHHH').unsqueeze(0),
                                  coordinate.unsqueeze(0).double()))[1]
test_data = torch.cat((test_data.squeeze(0), force), dim=1)
pd.DataFrame(test_data.cpu().numpy()).to_csv("test_data.csv", index=None, header=None)
print(test_data)


def locate_files(file_path):
    filenames_a = []
    for _, __, files in os.walk(file_path):
        for filename in files:
            if filename.endswith(".csv"):
                filenames_a.append(file_path + "/" + filename)
    filenames_a = natsort.natsorted(filenames_a)

    return filenames_a


filenames = locate_files("testdata")


def process_files(filenames_a):

    print(str(len(filenames_a)) + " files need to be processed")
    for i in range(801):
        print("********************************************")
        print("Data file " + filenames_a[i] + " is being processing:")
        print(aev_computer)

        data_a = pd.read_csv(filenames_a[i], header=None)

        coordinate_tensor_a = torch.tensor(data_a.values)

        force_a = coordinate_tensor_a[:, 3:]
        coordinate_a = coordinate_tensor_a[0:, :3]

        test_data_a = aev_computer.forward((consts.species_to_tensor('HHHHHHHHHH').unsqueeze(0),
                                            coordinate_a.unsqueeze(0).double()))[1]
        test_data_a = torch.cat((test_data_a.squeeze(0), force_a), dim=1)
        file_string = "train/" + "ga" + str(i) + ".csv"
        pd.DataFrame(test_data_a.cpu().numpy()).to_csv(file_string, index=None, header=None)


process_files(filenames)
