"""Generate the voxel dataset.

This script loads protein-ligand files and voxelizes them using the specified parameters.
This script assumes protein and ligand for each complex are in separate files and both 
begin with the same unique identifier, e.g. 1abc_protein.pdb and 1abc_ligand_rnum.pdb.

Usage examples:
    python -m scripts.generate_voxel_dataset --help
    
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from docktgrid import MolecularParser, VoxelDataset, VoxelGrid
from docktgrid.view import *


def main(args):
    protein_files = sorted(
        get_files(args.protein_pattern, args.data_dir, args.recursive)
    )
    ligand_files = sorted(get_files(args.ligand_pattern, args.data_dir, args.recursive))

    if not protein_files or not ligand_files:
        raise FileNotFoundError(
            f"No files found with pattern: {args.protein_pattern} or {args.ligand_pattern} in directory: {args.data_dir}"
        )

    if len(protein_files) != len(ligand_files):
        raise ValueError(
            f"Number of protein files ({len(protein_files)}) is not equal to number of ligand files ({len(ligand_files)}). Check the patterns."
        )

    protein_mols, ligand_mols = get_molecular_data(protein_files, ligand_files)
    dataset = get_voxel_dataset(args, protein_mols, ligand_mols)
    generate_and_save_voxels(args, protein_files, ligand_files, dataset)


def generate_and_save_voxels(args, protein_files, ligand_files, dataset):
    for i, (voxs, _) in tqdm(
        enumerate(dataset), desc="Voxelizing and saving tensors", total=len(dataset)
    ):
        output_dir = (
            os.path.dirname(protein_files[i])
            if args.output_dir == ""
            else args.output_dir
        )

        os.makedirs(os.path.join(output_dir, "../voxels"), exist_ok=True)

        ending_pattern = args.protein_pattern.split("*")[-1]
        basename = os.path.basename(protein_files[i]).replace(ending_pattern, "")
        output_file = os.path.join(
            output_dir, "../voxels", f"{basename}.{args.output_file_format}"
        )

        if args.output_file_format == "npy":
            np.save(output_file, voxs.detach().cpu().numpy())
        elif args.output_file_format == "pt":
            torch.save(voxs, output_file)
        else:
            raise ValueError(
                f"Output file format {args.output_file_format} not supported."
            )

        # save a .conf file with the parameters names and values, save in a json style
        params = vars(args)
        with open(os.path.join(output_dir, "../voxels", "voxel.conf"), "w") as f:
            json.dump(params, f, indent=4)


def get_molecular_data(protein_files, ligand_files):
    mol_parser = MolecularParser()

    protein_mols = []
    for file in tqdm(sorted(protein_files), desc="Processing protein files"):
        # join ptn and cofacs if they exist
        cofactors_dir = os.path.join(os.path.dirname(file), "cofactors")
        if "protein" in file and os.path.exists(cofactors_dir):
            cofactors = get_files("*.pdb", cofactors_dir)
            output_file = file + ".cofacs.pdb"  # tmp file
            join_files([file, *cofactors], output_file)

            mol = mol_parser.parse_file(output_file, os.path.splitext(output_file)[1])
            os.remove(output_file)
        else:
            mol = mol_parser.parse_file(file, os.path.splitext(file)[1])
        protein_mols.append(mol)

    ligand_mols = []
    for file in tqdm(sorted(ligand_files), desc="Processing ligand files"):
        mol = mol_parser.parse_file(file, os.path.splitext(file)[1])
        ligand_mols.append(mol)
    return protein_mols, ligand_mols


def get_voxel_dataset(args, protein_mols, ligand_mols):
    voxel = VoxelGrid(
        views=[eval(v)() for v in args.views],
        vox_size=args.voxel_size,
        box_dims=args.box_dims,
    )

    return VoxelDataset(
        protein_mols,
        ligand_mols,
        range(len(protein_mols)),
        voxel,
    )


def join_files(files: list[str], output_file: str) -> None:
    """Join the files in the list into a single file."""
    with open(output_file, "w") as outfile:
        for fname in files:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)


def get_files(pattern: str, root_dir: str, recursive: bool = False) -> list[str]:
    return glob.glob(os.path.join(root_dir, pattern), recursive=recursive)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data-dir", default="data/pdbbind2020-refined-prepared", help="root directory for data files")
    parser.add_argument("--voxel-size", type=float, default=1.0, help="voxel size in Angstroms")
    parser.add_argument("--box-dims", type=float, nargs=3, default=[24.0, 24.0, 24.0], help="box dimensions in Angstroms")
    parser.add_argument("--views", nargs="+", type=str, default=["VolumeView", "BasicView"], help="views to use")
    parser.add_argument("--protein-pattern", default="**/*_protein.pdb", help="glob pattern for finding protein files")
    parser.add_argument("--ligand-pattern", default="**/*_ligand_rnum.pdb", help="glob pattern for finding ligand files")
    parser.add_argument("-r", "--recursive", action="store_true", help="search for files recursively")
    parser.add_argument("--output-dir", default="", help="output directory (default: if empty, use the same as the protein file directory)")
    parser.add_argument("--output-file-format", default="npy", choices=["npy", "pt"], help="output file format")
    # fmt: on
    args = parser.parse_args()
    main(args)
