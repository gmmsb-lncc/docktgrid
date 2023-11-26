"""Preprocess the dataset for training machine learning models.

This script loads the data files and stores them in serializable file formats, which can 
be quickly loaded and used for training machine learning models.

Usage examples:
    * python -m docktgrid.scripts.preprocess_dataset --pattern '*.pdb' --dir tests/data/dataset
    * python -m docktgrid.scripts.preprocess_dataset --pattern '**/*_protein.pdb' --dir data/pdbbind2020-refined-prepared --recursive

Use --help to see all options.
"""

import argparse
import glob
import os
import pickle

from tqdm import tqdm

from docktgrid.molparser import MolecularParser


def main(args):
    files = get_files(args.pattern, args.dir, args.recursive)
    if not files:
        raise FileNotFoundError(
            "No files found with pattern: {} in directory: {}".format(
                args.pattern, args.dir
            )
        )

    # create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    parser = MolecularParser()
    for file in tqdm(files):
        # join ptn and cofacs if they exist
        cofactors_dir = os.path.join(os.path.dirname(file), "cofactors")
        if "protein" in file and os.path.exists(cofactors_dir):
            cofactors = get_files("*.pdb", cofactors_dir)
            output_file = os.path.basename(file) + ".cofactors.pdb"  # tmp file
            output_file = os.path.join(args.output, output_file)
            join_files([file, *cofactors], output_file)
            mol = parser.parse_file(output_file, os.path.splitext(output_file)[1])
            os.remove(output_file)

        else:
            mol = parser.parse_file(file, os.path.splitext(file)[1])

        with open(
            os.path.join(args.output, os.path.basename(file) + ".pkl"), "wb"
        ) as f:
            pickle.dump(mol, f)


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
    parser.add_argument("-p", "--pattern", default="*.pdb", help="glob pattern for finding files")
    parser.add_argument("-d", "--dir", default="data", help="root directory for data files")
    parser.add_argument("-o", "--output", default="data/processed", help="output directory")
    parser.add_argument("-r", "--recursive", action="store_true", help="recursively search for files")
    # fmt: on
    args = parser.parse_args()
    main(args)
