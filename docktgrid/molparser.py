import pickle
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from biopandas import mmcif, mol2, pdb

from .config import DTYPE

__all__ = ["MolecularData", "MolecularParser", "Parser", "extract_binding_pocket"]


@dataclass
class MolecularData:
    """Dataclass for storing molecular info.

    Args:
        molecule_object:
            A biopandas molecule object (can be pdb, mol2 or mmcif)
        coords:
            torch.Tensor of shape (3, n_atoms).
        element_symbols:
            np.ndarray of shape (n_atoms,), type str.
    """

    molecule_object: mol2.PandasMol2 | pdb.PandasPdb | mmcif.PandasMmcif
    coords: torch.Tensor
    element_symbols: np.ndarray


class Parser(Protocol):
    """Interface for implementing molecular parsers."""

    def parse_file(self, mol_file: str, ext: str) -> MolecularData:
        """Parse molecular file and return a MolecularData object.

        Args:
            mol_file (str):
                Full path to the file.
            ext (str):
                File extension (file format).

        Returns:
            A MolecularData object.

        """
        ...


class MolecularParser:
    """Get molecular info using biopandas."""

    def parse_file(self, mol_file: str, ext: str) -> MolecularData:
        """Parse molecular file and return a MolecularData object."""
        self.ppdb = pdb.PandasPdb()
        self.pmol2 = mol2.PandasMol2()

        if ext.lower() in ("pdb", ".pdb"):  # PDB file format
            mol = self.ppdb.read_pdb(mol_file)
            self.df_atom = mol.df["ATOM"]
            self.df_hetatm = mol.df["HETATM"]
            return MolecularData(
                mol, self.get_coords_pdb(), self.get_element_symbols_pdb()
            )
        else:
            raise NotImplementedError(f"File format {ext} not implemented.")

    def get_coords_pdb(self) -> torch.Tensor:
        hetatm_coords = self.df_hetatm[["x_coord", "y_coord", "z_coord"]].values
        atom_coords = self.df_atom[["x_coord", "y_coord", "z_coord"]].values
        coords = np.concatenate((atom_coords, hetatm_coords), axis=0).T
        return torch.tensor(coords, dtype=DTYPE)

    def get_element_symbols_pdb(self) -> list[str]:
        hetatm_symbols = self.df_hetatm["element_symbol"].values
        atom_symbols = self.df_atom["element_symbol"].values
        symbols = np.concatenate((atom_symbols, hetatm_symbols), axis=0)
        return symbols


def extract_binding_pocket(protein_coords, center_point, cutoff_radius):
    """Extract the binding pocket from the protein coordinates.

    Args:
        protein_coords:
            torch.Tensor of shape (3, n_atoms).
        center_point:
            torch.Tensor of shape (3,).
        cutoff_radius:
            float.

    Returns:
        A torch.Tensor of shape (n_included_atoms,) with the indices of the binding
        pocket atoms, excluding those that are outside the cutoff radius.

    """
    dists = torch.norm(protein_coords - center_point[:, None], dim=0)
    return (dists < cutoff_radius).nonzero().squeeze()
