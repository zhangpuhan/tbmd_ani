import os
import torch
import neurochem
import aev
import pyanitool as pya

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = os.path.dirname(os.path.realpath(__file__))

print(path)

const_file = os.path.join(path, 'torchani/resources/ani-1x_8x/rHCNO-5.2R_16-3.5A_a4-8.params')

print(const_file)
consts = neurochem.Constants(const_file)

print(consts)

aev_computer = aev.AEVComputer(**consts)

print(aev_computer)



# Set the HDF5 file containing the data
hdf5file = 'ani_gdb_s01.h5'

# Construct the data loader class
adl = pya.anidataloader(hdf5file)

# Print the species of the data set one by one
coordinates = []
species = []
for data in adl:

    # Extract the data
    P = data['path']
    X = data['coordinates']
    coordinates.append(X)
    E = data['energies']
    S = data['species']
    species.append(S)
    sm = data['smiles']

    # Print the data
    print("Path:   ", P)
    print("  Symbols:     ", S)
    print("  Coordinates: ", X)
    print("  Energies:    ", E, "\n")
    print("  Smiles:      ", sm)

print(coordinates[0][0])
print(species[0])
print(torch.tensor([coordinates[0][0]]))
print(consts.species_to_tensor('CHHHH').unsqueeze(0))

print(aev_computer.forward((consts.species_to_tensor('CHHHH').unsqueeze(0), torch.tensor([coordinates[0][0]])))[1].size())
# Closes the H5 data file
adl.cleanup()
