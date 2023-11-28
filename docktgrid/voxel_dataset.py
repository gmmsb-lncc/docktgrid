from typing import Optional

import torch
from torch.utils.data import Dataset

from docktgrid import MolecularComplex, VoxelGrid
from docktgrid.config import DTYPE
from docktgrid.molparser import MolecularData, MolecularParser
from docktgrid.transforms import RandomRotation, Transform

__all__ = ["VoxelDataset"]


class VoxelDataset(Dataset):
    """Dataset for protein-ligand voxel data (generates voxel grids on-the-fly).

    Protein and ligand files must be in a list of strings or a list of MolecularData
    objects and must appear in the same order.
    """

    def __init__(
        self,
        protein_files: list[str] | list[MolecularData],
        ligand_files: list[str] | list[MolecularData],
        labels: list[float],
        voxel: VoxelGrid,
        molparser: MolecularParser = MolecularParser(),
        transform: Optional[list[Transform]] = None,
        root_dir: str = "",
    ):
        assert len(protein_files) == len(ligand_files), "must have the same length!"
        assert len(protein_files) == len(labels), "must have the same length!"

        self.ptn_files = protein_files
        self.lig_files = ligand_files
        self.labels = torch.as_tensor(labels, dtype=DTYPE)
        self.voxel = voxel
        self.molparser = molparser
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        molecule = MolecularComplex(
            self.ptn_files[idx], self.lig_files[idx], self.molparser, self.root_dir
        )
        label = self.labels[idx]

        for transform in self.transform or []:
            if isinstance(transform, RandomRotation):
                transform(molecule.coords, molecule.ligand_center)

        voxs = self.voxel.voxelize(molecule)

        return voxs, label
