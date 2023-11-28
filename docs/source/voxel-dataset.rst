Datasets and model training
---------------------------

We can use the `VoxelDataset` class to iterate over a dataset of protein-ligand files.
The `VoxelDataset` class is a subclass of `torch.utils.data.Dataset` and can be used
with `torch.utils.data.DataLoader` to iterate over the dataset in batches.

First, we define the voxel grid parameters:

.. code-block:: python
    
    from docktgrid.voxel import VoxelGrid
    from docktgrid.view import VolumeView, BasicView

    voxel = VoxelGrid(
        views=[VolumeView(), BasicView()],
        vox_size=1.0,
        box_dims=[24.0, 24.0, 24.0],
    )

In the simplest case, we could just provide a list of protein-ligand files to the `VoxelDataset` class:

.. code-block:: python
    
    from docktgrid.voxel_dataset import VoxelDataset

    pdbs = ["1xap", "2weg", "4bb9", "4qsu", "6std"]
    root_dir="../tests/data/dataset"

    data = VoxelDataset(
        protein_files=[f"{pdb}_protein.pdb" for pdb in pdbs],
        ligand_files=[f"{pdb}_ligand.pdb" for pdb in pdbs],
        labels=range(len(pdbs)),
        voxel=voxel,
        transform=None,
        root_dir=root_dir,
    )

    len(data)
    >>> 5

However, we can also load the files into memory and provide `MolecularData` objects instead,
which allows for much faster training:

.. code-block:: python

    from docktgrid.molparser import MolecularParser
    from docktgrid.transforms import RandomRotation
    import os

    parser = MolecularParser()
    protein_file = os.path.join(root_dir, "{}_protein.pdb")
    ligand_file = os.path.join(root_dir, "{}_ligand.pdb")

    proteins = [parser.parse_file(protein_file.format(pdb), ".pdb") for pdb in pdbs]
    ligands = [parser.parse_file(ligand_file.format(pdb), ".pdb") for pdb in pdbs]

    data = VoxelDataset(
        protein_files=proteins,
        ligand_files=ligands,
        labels=range(len(pdbs)),
        voxel=voxel,
        transform=[RandomRotation()],   # <<< use a random rotation transform here
        root_dir=root_dir,
    )

We can also use a random rotation transform to augment the dataset, as shown above.

Finally, the `VoxelDataset` we created can be used with a `DataLoader` to iterate over the dataset in batches:

.. code-block:: python

    from torch.utils.data import DataLoader

    dataloader = DataLoader(data, batch_size=2, shuffle=True)

    for x, y in dataloader:
        # training code here...
        break
    
    x.shape, y.shape
    
    >>> (torch.Size([2, 21, 24, 24, 24]), torch.Size([2]))


