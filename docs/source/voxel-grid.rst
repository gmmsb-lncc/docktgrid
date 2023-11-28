Create a voxel grid
-------------------

To create a voxel grid of a protein-ligand complex, we need to instantiate a `VoxelGrid` object. 

This object is initialized with the following parameters:

- `views`: a list of `View` objects
- `vox_size`: the size of the voxels in Angstroms
- `box_dims`: the dimensions of the box in which the voxel grid is created.


.. code-block:: python

    from docktgrid.voxel import VoxelGrid
    from docktgrid.view import BasicView, VolumeView

    voxel = VoxelGrid(
        views=[VolumeView(), BasicView()],  # you can add multiple views; they are executed in order
        vox_size=1.0,                       # size of the voxel (in Angstrom)
        box_dims=[24.0, 24.0, 24.0],        # dimensions of the box (in Angstrom)
    )

    voxel.shape
    >>> (21, 24, 24, 24)


Next, we need a `MolecularComplex` object to create the voxel grid from.

.. code-block:: python

    from docktgrid.molecule import MolecularComplex
    
    protein_file = "path/to/protein.pdb"
    ligand_file = "path/to/ligand.pdb"
    mol = MolecularComplex(protein_file, ligand_file)

Finally, we can create the voxel grid.

.. code-block:: python

    grid = voxel.voxelize(mol)

    grid.shape
    >>> torch.Size([21, 24, 24, 24])

The grid uses the "channels first" format, i.e. the shape of the grid is `(n_channels, n_x, n_y, n_z)`.

The `VolumeView` adds three channels to the grid: one for the protein-ligand complex, one for the protein, and one for the ligand. These channels include all atoms of the respective entity.

The `BasicView` has channels for carbon, hydrogen, oxygen, nitrogen, sulfur and other (other elements). Each entity (complex, protein, and ligand) gets its own set of channels, totaling 18 channels.

Channels are listed in the tensor in the following order:

.. code-block:: python

    VolumeView().get_channels_names() + BasicView().get_channels_names()
    >>> ['complex_volume',
        'protein_volume',
        'ligand_volume',
        'carbon_complex',
        'hydrogen_complex',
        'oxygen_complex',
        'nitrogen_complex',
        'sulfur_complex',
        'other_complex',
        'carbon_protein',
        'hydrogen_protein',
        'oxygen_protein',
        'nitrogen_protein',
        'sulfur_protein',
        'other_protein',
        'carbon_ligand',
        'hydrogen_ligand',
        'oxygen_ligand',
        'nitrogen_ligand',
        'sulfur_ligand',
        'other_ligand']


