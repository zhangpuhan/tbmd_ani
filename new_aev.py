import pandas as pd
import os
import torch
import neurochem
import aev
import pyanitool as pya

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

print(aev_computer.forward((consts.species_to_tensor('HHHHHHHHHH').unsqueeze(0), coordinate.unsqueeze(0).double())))
