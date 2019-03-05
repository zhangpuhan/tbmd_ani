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


