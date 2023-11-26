"""Script to export voxelized protein structures to PyMOL sessions using DockTDeep.

This script requires the additional packages 'pymol' and 'gridDataFormats' to be installed.

"""
try:
    import pymol
except (ImportError, ModuleNotFoundError):
    raise ImportError(
        "The functionality you are trying to use requires the 'pymol' package. "
        "Please refer to https://pymol.org for instructions on installing "
        "PyMOL in your environment."
    )

try:
    import gridData
except (ImportError, ModuleNotFoundError):
    raise ImportError(
        "The functionality you are trying to use requires the 'gridDataFormats' package. "
        "Please install it using 'pip install gridDataFormats'."
    )


import argparse
import logging
import os
import time

import gridData as gdd
import numpy as np
import torch
from pymol import cmd

import docktgrid
from docktgrid import MolecularComplex, VoxelGrid
from docktgrid.config import is_using_gpu
from docktgrid.view import *


def export_voxels(args: argparse.Namespace) -> str:
    c = MolecularComplex(
        args.protein_file,
        args.ligand_file,
        molparser=docktgrid.MolecularParser(),
        path=args.dir,
    )

    views = [eval(view)() for view in args.views]
    voxel = VoxelGrid(views=views, vox_size=args.vox_size, box_dims=args.box_dims)

    if not args.voxel_grid:
        grid = voxel.voxelize(c)
    else:
        grid = np.load(args.voxel_grid)

    # create a temporary directory with a random name to save the voxelized structures
    tmp_dir = "tmp_dockt_" + str(time.time()).replace(".", "")
    logging.info("Saving voxelized structures in: {}".format(tmp_dir))
    os.mkdir(tmp_dir)

    export_voxel_dx(c, grid, voxel, tmp_dir + "/c")
    return tmp_dir


def get_channel_names(args):
    return [v for view in args.views for v in eval(view)().get_channels_names()]


def main(args: argparse.Namespace):
    logging.info("Using GPU: {}".format(is_using_gpu()))
    tmp_dir = export_voxels(args)

    # load all the files in the temporary directory in ascending order
    files = sorted(os.listdir(tmp_dir))
    for file, ch_name in zip(files, get_channel_names(args)):
        cmd.load(os.path.join(tmp_dir, file), object=ch_name)

    cmd.load(os.path.join(args.dir, args.protein_file))
    cmd.load(os.path.join(args.dir, args.ligand_file))

    logging.info("Saving PSE file in: {}".format(args.output))
    cmd.save(args.output)

    logging.info("Removing temporary directory: {}".format(tmp_dir))
    os.system("rm -rf {}".format(tmp_dir))


def export_voxel_dx(
    molecule: docktgrid.MolecularComplex,
    voxel: torch.Tensor,
    voxel_grid: docktgrid.VoxelGrid,
    name_prefix: str = "",
):
    """Export each voxel channel to a file in OpenDX format.

    OpenDX format is generally accepted in popular molecular visualization
    softwares (such as PyMOL and VMD).

    Args:
        mol_complex: molecular complex
        voxels: voxelized complex, optional.
        name_prefix: Export file name prefix.

    Returns:
        Files in DX format for each channel of voxel grid.

    """
    if voxel is not None and type(voxel) == torch.Tensor:
        voxel = voxel.detach().cpu().numpy()

    # get shapes right
    vox_grid = voxel.reshape(voxel_grid.shape[0], voxel_grid.grid.points[0].shape[0])
    xs, ys, zs = voxel_grid.grid.axes_dims

    for c in range(voxel_grid.shape[0]):
        voxs = vox_grid[c, :]
        voxs.shape = (xs, ys, zs)
        # align origin and export
        lig_center = molecule.ligand_center.detach().cpu().numpy()
        box_dims = voxel_grid.grid._box_dims.detach().cpu().numpy()
        center = lig_center - (box_dims / 2.0)
        g = gdd.Grid(voxs, origin=center, delta=voxel_grid.grid._vox_size)
        g.export((name_prefix + "_{:02d}".format(c + 1)), "DX")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--protein-file", type=str, help="Protein PDB file name.", required=True
    )
    parser.add_argument(
        "--ligand-file", type=str, help="Ligand PDB file name.", required=True
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="",
        help="Directory where the PDB files are located. Optional.",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        type=str,
        default=["BasicView"],
        help="Views to use. Default: BasicView.",
    )
    parser.add_argument(
        "--vox-size",
        type=float,
        default=0.5,
        help="Voxel size.",
    )
    parser.add_argument(
        "--box-dims",
        type=float,
        nargs="+",
        default=[24.0, 24.0, 24.0],
        help="Box dimensions.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="voxelized.pse",
        help="Output file name.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    parser.add_argument(
        "--voxel-grid",
        type=str,
        default="",
        help="Voxel grid file name (.npy). If provided, the voxel grid will not be computed.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    main(parser.parse_args())
