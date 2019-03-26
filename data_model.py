import torch
import torch.nn.functional as F
import torch.utils.data as Data
from constant import DIRECTIONS, FILE_SIZES, CUT_OFF, ATOM_NUMBER
from fre_functions import f_c, exponential_map

coordinate_temp = torch.load("coordinate03252019.pt").requires_grad_(True)
energy_temp = torch.load("energy03252019.pt").requires_grad_(True)
force_temp = torch.load("force03252019.pt").requires_grad_(True)

torch_data_set = Data.TensorDataset(coordinate_temp, energy_temp, force_temp)
loader = Data.DataLoader(
    dataset=torch_data_set,
    batch_size=64,
    shuffle=True
)

for step, (coordinate, energy, force) in enumerate(loader):
    print(coordinate, energy, force)



