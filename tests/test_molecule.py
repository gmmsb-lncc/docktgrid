import torch

from docktgrid.molecule import MolecularComplex
from docktgrid.molparser import MolecularParser


def test_coords_have_correct_values():
    x, y, z = 0, 1, 2  # spatial dimensions
    c = MolecularComplex(
        "6rnt_protein.pdb",
        "6rnt_ligand.pdb",
        molparser=MolecularParser(),
        path="tests/data",
    )

    assert c.coords[x][953].cpu() == torch.tensor(19.6670)
    assert c.coords[y][0].cpu() == torch.tensor(-5.6270)
    assert c.coords[z][-1].cpu() == torch.tensor(-2.9060)


def test_ligand_center_have_correct_values():
    c = MolecularComplex(
        "6rnt_protein.pdb",
        "6rnt_ligand.pdb",
        molparser=MolecularParser(),
        path="tests/data",
    )
    assert torch.allclose(
        c.ligand_center,
        torch.tensor([21.7565, 11.7188, -0.8388]),
        atol=1e-4,
    )


def test_n_atoms():
    c = MolecularComplex(
        "6rnt_protein.pdb",
        "6rnt_ligand.pdb",
        molparser=MolecularParser(),
        path="tests/data",
    )

    assert c.n_atoms == 988
    assert c.n_atoms_protein == 953
    assert c.n_atoms_ligand == 35


def test_vdw_radii_are_correct():
    c = MolecularComplex(
        "6rnt_protein.pdb",
        "6rnt_ligand.pdb",
        molparser=MolecularParser(),
        path="tests/data",
    )

    assert c.vdw_radii[953].cpu() == torch.tensor(1.8000)
    assert c.vdw_radii[0].cpu() == torch.tensor(1.5500)
    assert c.vdw_radii[-1].cpu() == torch.tensor(1.1000)
