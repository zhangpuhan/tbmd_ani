import torch
import torch.nn.functional as F
import torch.utils.data as Data
from constant import DIRECTIONS, FILE_SIZES, CUT_OFF, ATOM_NUMBER
from fre_functions import f_c, exponential_map
import util
from data_model import Net

model = torch.load("net")
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())