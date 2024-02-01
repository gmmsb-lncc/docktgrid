import os

import torch

from docktgrid.molparser import MolecularParser


def test_get_coords():
    file = "tests/data/6rnt_protein.pdb"
    molparser = MolecularParser()
    coords = molparser.parse_file(mol_file=file, ext=os.path.splitext(file)[1]).coords

    assert torch.equal(coords[:, 0].cpu(), torch.tensor([6.905, -5.627, 16.26]))


def test_get_element_symbols():
    file = "tests/data/6rnt_protein.pdb"
    molparser = MolecularParser()
    names = molparser.parse_file(
        mol_file=file, ext=os.path.splitext(file)[1]
    ).element_symbols

    assert names[-1] == "Ca"


def test_get_coords_mol2():
    file = "tests/data/6rnt_ligand.mol2"
    molparser = MolecularParser()
    coords = molparser.parse_file(mol_file=file, ext=os.path.splitext(file)[1]).coords

    assert torch.equal(coords[:, 0].cpu(), torch.tensor([19.946, 13.521, 1.248]))


def test_get_element_symbols_mol2():
    file = "tests/data/6rnt_ligand.mol2"
    molparser = MolecularParser()
    names = molparser.parse_file(
        mol_file=file, ext=os.path.splitext(file)[1]
    ).element_symbols

    assert names[0] == "P"
