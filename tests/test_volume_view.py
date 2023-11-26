import torch

from docktgrid.molecule import MolecularComplex
from docktgrid.molparser import MolecularParser
from docktgrid.view import VolumeView


def test_volume_view_num_channels():
    view = VolumeView()
    assert view.get_num_channels() == 1 * 3


def test_volume_view():
    view = VolumeView()
    molparser = MolecularParser()
    c = MolecularComplex(
        "6rnt_protein.pdb", "6rnt_ligand.pdb", molparser=molparser, path="tests/data/"
    )

    assert torch.all(view.get_molecular_complex_channels(c)[0])

    assert not torch.all(view.get_protein_channels(c)[0][:-34])

    assert torch.all(view.get_protein_channels(c)[0][:-35])

    assert torch.all(view.get_ligand_channels(c)[0][-35:])
