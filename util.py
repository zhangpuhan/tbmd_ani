""" This file contains utility functions """
import torch
from constant import RADIAL_SAMPLE_RUBRIC, ANGULAR_SAMPLE_RUBRIC
import itertools


class GenerateCombinations:
    """ this function generate position combinations """
    def __init__(self):
        self.device = torch.device('cpu')
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def combination_2(self, n_number, n_select):
        nums = [i for i in range(1, n_number)]
        result = []
        self.dfs(nums, 0, [0], result, n_select)
        return torch.tensor(result, device=self.device)

    def dfs(self, nums, index, path, result, n_select):
        if len(path) == n_select + 1:
            result.append(list(path))
            return

        for i in range(index, len(nums)):
            path.append(nums[i])
            self.dfs(nums, i + 1, path, result, n_select)
            path.pop()

    def generate_combination_dic(self, n_number, n_select):
        result = {}
        for i in range(3, n_number + 1):
            result[i] = self.combination_2(i, n_select)

        return result


class GenerateSampleGrid:
    """ this function generates sample grids """
    def __init__(self):
        self.radial_parameters = list(RADIAL_SAMPLE_RUBRIC.values())
        self.angular_parameters = list(ANGULAR_SAMPLE_RUBRIC.values())
        self.device = torch.device('cpu')
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def generate_radial_grid(self):
        result = []
        for eta, rs in itertools.product(self.radial_parameters[0], self.radial_parameters[1]):
            result.append([eta, rs])

        return torch.tensor(result, device=self.device, dtype=torch.float64)

    def generate_angular_grid(self):
        result = []
        for eta, rs, zeta, theta in itertools.product(self.angular_parameters[0], self.angular_parameters[1],
                                                      self.angular_parameters[2], self.angular_parameters[3]):
            result.append([eta, rs, zeta, theta])

        return torch.tensor(result, device=self.device, dtype=torch.float64)





