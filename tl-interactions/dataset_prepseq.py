"""Prepare the sequence part of a dataset: archive the sequences and list the proteins with PDB hits for training.

A protein goes into ``train.txt`` if BLAST found a hit in a PDB chain deposited on or before the cutoff date with at
least the given sequence identity. The output directory receives:

- ``sequences.zip``: the input sequence directory (its contents at the archive root)
- ``train.txt``: IDs of the proteins with such hits, one per line
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import pandas as pd

from parse_blast_tsv import read_blast_tsv


def log(message):
    """Print a progress message to stderr, keeping stdout for the results."""
    print(message, file=sys.stderr, flush=True)


def read_protein_ids(sequence_dir):
    """Read protein IDs from the FASTA files in a directory.

    Parameters
    ----------
    sequence_dir : Path
        Directory of ``*.fasta`` files. The ID is the first ``|``-separated field of each record title, as in
        :func:`parse_blast_tsv.read_blast_tsv`.

    Returns
    -------
    set of str
    """
    return {
        line[1:].split("|", 1)[0].strip()
        for path in sequence_dir.glob("*.fasta")
        for line in path.read_text().splitlines()
        if line.startswith(">")
    }


def read_deposit_dates(path):
    """Read the deposition dates from a PDB ``entries.idx`` file.

    Parameters
    ----------
    path : Path
        PDB entry index, e.g. from https://files.wwpdb.org/pub/pdb/derived_data/index/entries.idx.

    Returns
    -------
    pandas.Series
        Deposition dates indexed by PDB ID.
    """
    entries = pd.read_csv(
        path,
        sep="\t",
        skiprows=2,
        header=None,
        usecols=[0, 2],
        names=["pdb_id", "deposit_date"],
        quoting=csv.QUOTE_NONE,  # titles contain unbalanced quotes, which would otherwise swallow lines
    )
    return pd.to_datetime(entries.set_index("pdb_id")["deposit_date"], format="%m/%d/%y")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sequence_dir", type=Path, help="Directory of per-protein FASTA files")
    parser.add_argument("blast_tsv", type=Path, help="Tabular BLAST report readable by parse_blast_tsv.py")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument("--pdb-index", type=Path, required=True, help="PDB entries.idx")
    parser.add_argument(
        "--cutoff-date", type=pd.Timestamp, default="2021-09-30",
        help="Latest PDB deposition date of a hit (default: %(default)s)",
    )
    parser.add_argument(
        "--min-identity", type=float, default=30, help="Minimum percent identity of a hit (default: %(default)s)"
    )
    args = parser.parse_args()

    log(f"Reading sequences from {args.sequence_dir}")
    proteins = read_protein_ids(args.sequence_dir)
    log(f"Reading BLAST report {args.blast_tsv}")
    hits, hsps = read_blast_tsv(args.blast_tsv)
    print(f"Proteins in {args.sequence_dir}: {len(proteins):,}")
    print(f"Unique subjects in {args.blast_tsv}: {hsps['subject'].nunique():,}")

    # Filter chains (not subjects): a subject may group identical chains deposited before and after the cutoff.
    # Chains of obsolete entries are missing from the index, have no deposition date and fail the comparison.
    log(f"Reading PDB index {args.pdb_index}")
    keys = ["query", "subject"]
    deposit_dates = hits["accession"].str[:4].map(read_deposit_dates(args.pdb_index))
    old_subjects = hits.loc[deposit_dates <= args.cutoff_date, keys].drop_duplicates()
    similar_subjects = hsps.loc[hsps["pident"] >= args.min_identity, keys].drop_duplicates()
    train = proteins & set(similar_subjects.merge(old_subjects, on=keys)["query"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Writing {args.output_dir / 'sequences.zip'}")
    shutil.make_archive(args.output_dir / "sequences", "zip", root_dir=args.sequence_dir)
    log(f"Writing {args.output_dir / 'train.txt'}")
    (args.output_dir / "train.txt").write_text("".join(f"{protein}\n" for protein in sorted(train)))

    print(f"Proteins with hits (deposited <= {args.cutoff_date:%Y-%m-%d}, pident >= {args.min_identity}): {len(train):,}")
    print(f"Proteins filtered out (no such hits): {len(proteins - train):,}")


if __name__ == "__main__":
    main()
