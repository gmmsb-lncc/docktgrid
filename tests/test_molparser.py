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
