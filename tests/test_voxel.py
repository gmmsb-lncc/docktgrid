import time

import torch

from docktgrid.molecule import MolecularComplex
from docktgrid.molparser import MolecularParser
from docktgrid.view import BasicView, VolumeView
from docktgrid.voxel import VoxelGrid


def test_num_channels():
    vox = VoxelGrid(views=[VolumeView()], vox_size=1.0, box_dims=[12.0, 12.0, 12.0])
    assert vox.num_channels == 3

    vox = VoxelGrid([BasicView()], 1.0, [12.0, 12.0, 12.0])
    assert vox.num_channels == 6 * 3

    vox = VoxelGrid([VolumeView(), BasicView()], 1.0, [12.0, 12.0, 12.0])
    assert vox.num_channels == 3 + 6 * 3


def test_voxel_shape():
    vox = VoxelGrid([VolumeView()], 0.5, [12.0, 12.0, 12.0])
    assert vox.shape == (3, 24, 24, 24)


def test_voxel_build_channels_matrix():
    molecule = MolecularComplex(
        "6rnt_protein.pdb", "6rnt_ligand.pdb", MolecularParser(), path="tests/data/"
    )

    vox = VoxelGrid([BasicView()], 0.5, [12.0, 12.0, 12.0])
    channels = vox.get_channels_mask(molecule)

    assert channels.shape == (3 * 6, molecule.n_atoms)


def test_voxel_grid():
    molecule = MolecularComplex(
        "6rnt_protein.pdb", "6rnt_ligand.pdb", MolecularParser(), path="tests/data/"
    )

    vox = VoxelGrid([VolumeView(), BasicView()], 1.0, [24.0, 24.0, 24.0])

    grid = vox.voxelize(molecule)  # compile first?

    stime = time.time()
    grid = vox.voxelize(molecule)
    etime = time.time()
    print(f"<voxelization time: {etime - stime}s>", end=" ", flush=True)

    load_grid = torch.load("tests/data/6rnt_grid.pt")

    assert torch.allclose(grid.detach().cpu(), load_grid, atol=1e-5)
    assert grid.shape == (3 + 3 * 6, 24, 24, 24)


MOLECULE = MolecularComplex(
    "6rnt_protein.pdb", "6rnt_ligand.pdb", MolecularParser(), path="tests/data/"
)
VOXEL = VoxelGrid([VolumeView(), BasicView()], 1.0, [24.0, 24.0, 24.0])


def test_voxelize():
    VOXEL.voxelize(MOLECULE)
