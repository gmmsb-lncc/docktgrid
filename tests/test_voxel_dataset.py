import torch

from docktgrid.transforms import RandomRotation
from docktgrid.view import BasicView
from docktgrid.voxel import VoxelGrid
from docktgrid.voxel_dataset import VoxelDataset


def setup_data(pdbs=["1xap", "2weg", "4bb9", "4qsu", "6std"]):
    pdbs = pdbs
    protein_files = [f"{pdb}_protein.pdb" for pdb in pdbs]
    ligand_files = [f"{pdb}_ligand.pdb" for pdb in pdbs]
    root_dir = "tests/data/dataset"

    voxel = VoxelGrid([BasicView()], 0.5, [12.0, 12.0, 12.0])

    dataset = VoxelDataset(
        protein_files,
        ligand_files,
        labels=list(range(1, len(pdbs) + 1)),
        voxel=voxel,
        root_dir=root_dir,
    )

    return dataset


def test_voxel_dataset_iterator():
    dataset = setup_data()
    for i, (grid, label) in enumerate(dataset):
        assert grid.shape == (3 * 6, 24, 24, 24)
        assert label == i + 1


def test_voxel_dataset_len():
    dataset = setup_data()
    assert len(dataset) == 5


def test_voxel_dataset_with_rotation_transform():
    torch.manual_seed(42)
    dataset = setup_data(["1xap", "1xap"])
    dataset.transform = [RandomRotation()]

    grid1, _ = dataset[0]
    grid2, _ = dataset[1]

    assert torch.any(grid1)
    assert torch.any(grid2)
    assert not torch.allclose(grid1, grid2)


def test_voxel_dataset_without_rotation():
    dataset = setup_data(["1xap", "1xap"])

    grid1, _ = dataset[0]
    grid2, _ = dataset[1]

    assert torch.any(grid1)
    assert torch.any(grid2)
    assert torch.allclose(grid1, grid2)
