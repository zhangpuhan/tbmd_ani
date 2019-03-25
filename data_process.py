import read_file
import util
import torch

# compute neighbor parameters:
neighbor_comb = util.GenerateCombinations()
angular_neighbor_combinations = neighbor_comb.generate_combination_dic(200, 2)
radial_neighbor_combinations = neighbor_comb.generate_combination_dic(200, 1)

# computer sample parameters:
sample_comb = util.GenerateSampleGrid()
angular_sample_comb = sample_comb.generate_angular_grid()
radial_sample_comb = sample_comb.generate_radial_grid()

# process snapshots
process_file_init = read_file.Aev("testdata")
train_set, force_result = process_file_init.process_files(radial_sample_comb, angular_sample_comb,
                                                          radial_neighbor_combinations, angular_neighbor_combinations)

energy_extraction = read_file.FileEnergy("test_energy")
energy_set = energy_extraction.process_files()
print(force_result)

for i in range(len(train_set)):
    train_set[i] = torch.reshape(train_set[i], (1, -1))

energy_data = torch.cat(tuple(energy_set), 0)
train_data = torch.cat(tuple(train_set), 0)
torch.save(train_data, "train0325.pt")
torch.save(energy_data, "energy0325.pt")
