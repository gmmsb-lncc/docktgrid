import abc

import numpy as np
import torch

from docktgrid.molecule import MolecularComplex

__all__ = ["View", "VolumeView", "BasicView"]


class View(metaclass=abc.ABCMeta):
    """Interface for defining voxel channels representations.

    Note that the atoms in the returning boolean matrices should follow the order they
    appear in the PDB file. The atoms from the protein are listed first, followed by
    those from the ligand.
    """

    @abc.abstractmethod
    def get_num_channels(self):
        """Return number of channels defined for the view.

        Returns:
            An integer with the number of channels in each set, i.e.:
            sum(complex_set, protein_set, ligand_set).
        """
        pass

    @abc.abstractmethod
    def get_channels_names(self):
        """Return names of channels defined for the view.

        Returns:
            A list of strings with the names of channels in each set, i.e.:
            concat(complex_set, protein_set, ligand_set), in this order.
        """
        pass

    @abc.abstractmethod
    def get_molecular_complex_channels(self, molecular_complex: MolecularComplex):
        """Set of channels considering all atoms of the protein-ligand complex together.

        Args:
            molecular_complex: MolecularComplex object.

        Returns:
            A boolean torch.Tensor array with shape
            (num_of_channels_defined_for_this_set, n_atoms_complex) or None.

        """
        pass

    @abc.abstractmethod
    def get_protein_channels(self, molecular_complex: MolecularComplex):
        """Set of channels considering protein atoms only.

        Args:
            molecular_complex: MolecularComplex object.

        Returns:
            A boolean torch.Tensor array with shape
            (num_of_channels_defined_for_this_set, n_atoms_complex) or None.

        """
        pass

    @abc.abstractmethod
    def get_ligand_channels(self, molecular_complex: MolecularComplex):
        """Set of channels considering ligand atoms only.

        Args:
            molecular_complex (molecule.Complex): docktgrid Complex object.

        Returns:
            A boolean torch.Tensor array with shape
            (num_of_channels_defined_for_this_set, n_atoms_complex) or None.

        """
        pass

    def __call__(self, molecular_complex: MolecularComplex) -> torch.Tensor:
        """Concatenate all channels in a single tensor.

        Args:
            molecular_complex: MolecularComplex object.

        Returns:
            A boolean torch.Tensor array with shape
            (num_of_channels_defined_for_this_view, n_atoms_complex)

        """
        complex = self.get_molecular_complex_channels(molecular_complex)
        protein = self.get_protein_channels(molecular_complex)
        ligand = self.get_ligand_channels(molecular_complex)
        return torch.cat(
            (
                complex if complex is not None else torch.tensor([], dtype=torch.bool),
                protein if protein is not None else torch.tensor([], dtype=torch.bool),
                ligand if ligand is not None else torch.tensor([], dtype=torch.bool),
            ),
        )


class VolumeView(View):
    """Default volume channel sets.

    This view includes all atoms from either protein, ligand or protein-ligand complex
    in a single channel.
    """

    def get_num_channels(self):
        return sum((1, 1, 1))

    def get_channels_names(self):
        return ["complex_volume", "protein_volume", "ligand_volume"]

    def get_molecular_complex_channels(
        self, molecular_complex: MolecularComplex
    ) -> torch.Tensor:
        return torch.ones((1, molecular_complex.n_atoms), dtype=torch.bool)

    def get_protein_channels(self, molecular_complex: MolecularComplex) -> torch.Tensor:
        vol = torch.zeros((1, molecular_complex.n_atoms), dtype=torch.bool)
        vol[0][: molecular_complex.n_atoms_protein] = True
        return vol

    def get_ligand_channels(self, molecular_complex: MolecularComplex) -> torch.Tensor:
        vol = torch.zeros((1, molecular_complex.n_atoms), dtype=torch.bool)
        vol[0][-molecular_complex.n_atoms_ligand :] = True
        return vol


class BasicView(View):
    """Basic view.

    The `x` below stands for any other chemical element different from CHONS.

    Protein channels (in this order):
        carbon, hydrogen, oxygen, nitrogen, sulfur, x*.
    Ligand channels:
        carbon, hydrogen, oxygen, nitrogen, sulfur, x*.
    """

    def get_num_channels(self):
        return sum((6, 6, 6))

    def get_channels_names(self):
        chs = ["carbon", "hydrogen", "oxygen", "nitrogen", "sulfur", "other"]
        return (
            [f"{ch}_complex" for ch in chs]
            + [f"{ch}_protein" for ch in chs]
            + [f"{ch}_ligand" for ch in chs]
        )

    def get_molecular_complex_channels(
        self, molecular_complex: MolecularComplex
    ) -> torch.Tensor:
        """Set of channels for all atoms."""

        channels = {
            0: ["C"],
            1: ["H"],
            2: ["O"],
            3: ["N"],
            4: ["S"],
            5: ["C", "H", "O", "N", "S"],
        }
        nchs = len(channels)

        # get a list of bools representing each atom in each channel
        symbs = molecular_complex.element_symbols
        chs = np.asarray([np.isin(symbs, channels[c]) for c in range(nchs)])

        # invert bools in last channel, since it represents any atom except CHONS
        np.invert(chs[-1], out=chs[-1])

        return torch.from_numpy(chs)

    def get_ligand_channels(self, molecular_complex: MolecularComplex) -> torch.Tensor:
        """Set of channels for ligand atoms."""
        chs = self.get_molecular_complex_channels(molecular_complex)

        # exclude protein atoms from ligand channels
        chs[..., : -molecular_complex.n_atoms_ligand] = False
        return chs

    def get_protein_channels(self, molecular_complex: MolecularComplex) -> torch.Tensor:
        """Set of channels for protein atoms."""
        chs = self.get_molecular_complex_channels(molecular_complex)

        # exclude ligand atoms from protein channels
        chs[..., -molecular_complex.n_atoms_ligand :] = False
        return chs
