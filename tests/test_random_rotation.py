import random

import numpy as np
import torch

from docktgrid import MolecularComplex, MolecularParser, RandomRotation


def test_random_rotation_is_actually_rotating():
    random.seed(42)
    np.random.seed(42)

    molparser = MolecularParser()
    complex = MolecularComplex(
        "tests/data/6rnt_protein.pdb", "tests/data/6rnt_ligand.pdb", molparser
    )
    complex_unrotated = MolecularComplex(
        "tests/data/6rnt_protein.pdb", "tests/data/6rnt_ligand.pdb", molparser
    )

    # sanity check
    assert torch.allclose(complex.coords, complex_unrotated.coords)

    transform = RandomRotation()
    transform(coords=complex.coords, ligand_center=complex.ligand_center)

    assert not torch.allclose(complex.coords, complex_unrotated.coords)
    assert not torch.allclose(complex.ligand_center, complex_unrotated.ligand_center)
