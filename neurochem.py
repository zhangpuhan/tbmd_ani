import collections
import torch
import utils


class Constants(collections.abc.Mapping):
    """NeuroChem constants. Objects of this class can be used as arguments
    to :class:`torchani.AEVComputer`, like ``torchani.AEVComputer(**consts)``.

    Attributes:
        species_to_tensor (:class:`ChemicalSymbolsToInts`): call to convert
            string chemical symbols to 1d long tensor.
    """

    def __init__(self, filename):
        self.filename = filename
        with open(filename) as f:
            for i in f:
                try:
                    line = [x.strip() for x in i.split('=')]
                    name = line[0]
                    value = line[1]
                    if name == 'Rcr' or name == 'Rca':
                        setattr(self, name, float(value))
                    elif name in ['EtaR', 'ShfR', 'Zeta',
                                  'ShfZ', 'EtaA', 'ShfA']:
                        value = [float(x.strip()) for x in value.replace(
                            '[', '').replace(']', '').split(',')]
                        setattr(self, name, torch.tensor(value))
                    elif name == 'Atyp':
                        value = [x.strip() for x in value.replace(
                            '[', '').replace(']', '').split(',')]
                        self.species = value
                except Exception:
                    raise ValueError('unable to parse const file')
        self.num_species = len(self.species)
        self.species_to_tensor = utils.ChemicalSymbolsToInts(self.species)

    def __iter__(self):
        yield 'Rcr'
        yield 'Rca'
        yield 'EtaR'
        yield 'ShfR'
        yield 'EtaA'
        yield 'Zeta'
        yield 'ShfA'
        yield 'ShfZ'
        yield 'num_species'

    def __len__(self):
        return 8

    def __getitem__(self, item):
        return getattr(self, item)


class EnergyShifter(torch.nn.Module):
    """Helper class for adding and subtracting self atomic energies

    This is a subclass of :class:`torch.nn.Module`, so it can be used directly
    in a pipeline as ``[input->AEVComputer->ANIModel->EnergyShifter->output]``.

    Arguments:
        self_energies (:class:`collections.abc.Sequence`): Sequence of floating
            numbers for the self energy of each atom type. The numbers should
            be in order, i.e. ``self_energies[i]`` should be atom type ``i``.
    """

    def __init__(self, self_energies):
        super(EnergyShifter, self).__init__()
        self_energies = torch.tensor(self_energies, dtype=torch.double)
        self.register_buffer('self_energies', self_energies)

    def sae(self, species):
        """Compute self energies for molecules.

        Padding atoms will be automatically excluded.

        Arguments:
            species (:class:`torch.Tensor`): Long tensor in shape
                ``(conformations, atoms)``.

        Returns:
            :class:`torch.Tensor`: 1D vector in shape ``(conformations,)``
            for molecular self energies.
        """
        self_energies = self.self_energies[species]
        self_energies[species == -1] = 0
        return self_energies.sum(dim=1)

    def subtract_from_dataset(self, species, coordinates, properties):
        """Transformer for :class:`torchani.data.BatchedANIDataset` that
        subtract self energies.
        """
        energies = properties['energies']
        device = energies.device
        energies = energies.to(torch.double) - self.sae(species).to(device)
        properties['energies'] = energies
        return species, coordinates, properties

    def forward(self, species_energies):
        """(species, molecular energies)->(species, molecular energies + sae)
        """
        species, energies = species_energies
        sae = self.sae(species).to(energies.dtype).to(energies.device)
        return species, energies + sae


def load_sae(filename):
    """Returns an object of :class:`EnergyShifter` with self energies from
    NeuroChem sae file"""
    self_energies = []
    with open(filename) as f:
        for i in f:
            line = [x.strip() for x in i.split('=')]
            index = int(line[0].split(',')[1].strip())
            value = float(line[1])
            self_energies.append((index, value))
    self_energies = [i for _, i in sorted(self_energies)]
    return EnergyShifter(self_energies)




