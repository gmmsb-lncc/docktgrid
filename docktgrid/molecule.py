import os

import numpy as np
import torch

from docktgrid.config import DTYPE
from docktgrid.molparser import MolecularData, MolecularParser, Parser
from docktgrid.periodictable import ptable

__all__ = ["MolecularComplex"]


class MolecularComplex:
    """Protein-ligand molecular complex.

    If the files are already parsed, pass them as MolecularData objects.

    Attrs:
        protein_data:
            A `MolecularData` object.
        ligand_data:
            A `MolecularData` object.
        coords:
            A torch.Tensor of shape (3, n_atoms).
        n_atoms:
            An integer with the total number of atoms.
        n_atoms_protein:
            An integer with the number of protein atoms.
        n_atoms_ligand:
            An integer with the number of ligand atoms.
        element_symbols:
            A np.ndarray of shape (n_atoms,), type str.
        vdw_radii:
            A torch.Tensor of shape (n_atoms,).

    """

    def __init__(
        self,
        protein_file: str | MolecularData,
        ligand_file: str | MolecularData,
        molparser: Parser | None = MolecularParser(),
        path="",
    ):
        """Initialize MolecularComplex.

        Args:
            protein_file:
                Path to the protein file or a MolecularData object.
            ligand_file:
                Path to the ligand file or a MolecularData object.
            molparser:
                A `MolecularParser` object.
            path:
                Path to the files.
        """
        if isinstance(protein_file, MolecularData):
            self.protein_data = protein_file
        else:
            self.protein_data: MolecularData = molparser.parse_file(
                os.path.join(path, protein_file), os.path.splitext(protein_file)[1]
            )

        if isinstance(ligand_file, MolecularData):
            self.ligand_data = ligand_file
        else:
            self.ligand_data: MolecularData = molparser.parse_file(
                os.path.join(path, ligand_file), os.path.splitext(ligand_file)[1]
            )

        self.ligand_center = torch.mean(self.ligand_data.coords, 1).to(dtype=DTYPE)
        self.coords = torch.cat((self.protein_data.coords, self.ligand_data.coords), 1)
        self.n_atoms: int = self.coords.shape[1]
        self.n_atoms_protein: int = self.protein_data.coords.shape[1]
        self.n_atoms_ligand: int = self.ligand_data.coords.shape[1]

        self.element_symbols: np.ndarray[str] = np.concatenate(
            (self.protein_data.element_symbols, self.ligand_data.element_symbols)
        )
        self.vdw_radii = self._get_vdw_radii()

    def _get_vdw_radii(self):
        return torch.tensor(
            [ptable[a.title()]["vdw"] for a in self.element_symbols],
            dtype=DTYPE,
        )
