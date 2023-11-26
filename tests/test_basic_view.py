import torch

from docktgrid.molecule import MolecularComplex
from docktgrid.molparser import MolecularParser
from docktgrid.view import BasicView


def test_basic_view_num_channels():
    view = BasicView()
    assert view.get_num_channels() == 6 * 3


def test_total_set():
    c = MolecularComplex(
        "6rnt_protein.pdb",
        "6rnt_ligand.pdb",
        molparser=MolecularParser(),
        path="tests/data/",
    )
    view = BasicView()
    total_set = view.get_molecular_complex_channels(c)
    n_atoms = 953 + 35

    assert total_set.shape == (6, n_atoms)
    assert total_set[0][299]
    assert total_set[1][292]
    assert total_set[3][0]
    assert total_set[4][939]
    assert total_set[4][12]
    assert total_set[5][952]
    assert total_set[5][953]  # ligand
    assert not total_set[0][292]
    assert not total_set[5][949]
    assert not total_set[5][0]
    assert not total_set[1][939]


def test_ligand_set():
    c = MolecularComplex(
        "6rnt_protein.pdb",
        "6rnt_ligand.pdb",
        molparser=MolecularParser(),
        path="tests/data/",
    )
    view = BasicView()
    ligand_set = view.get_ligand_channels(c)
    n_atoms = 953 + 35

    assert ligand_set.shape == (6, n_atoms)
    assert ligand_set[0][953 + 8]
    assert ligand_set[1][953 + 23]
    assert ligand_set[2][953 + 7]
    assert ligand_set[3][953 + 21]
    assert not any(ligand_set[4])  # no sulfur atoms
    assert ligand_set[5][953]
    assert not torch.any(ligand_set[:, :953])  # only protein atoms


def test_protein_set():
    c = MolecularComplex(
        "6rnt_protein.pdb",
        "6rnt_ligand.pdb",
        molparser=MolecularParser(),
        path="tests/data/",
    )
    view = BasicView()
    protein_set = view.get_protein_channels(c)

    assert not torch.any(protein_set[:, 953:])  # only ligand atoms
    assert protein_set[0][299]
    assert protein_set[1][292]
    assert protein_set[3][0]
    assert protein_set[4][939]
    assert protein_set[4][12]
    assert protein_set[5][952]
    assert not protein_set[0][292]
    assert not protein_set[5][949]
    assert not protein_set[5][0]
    assert not protein_set[1][939]
