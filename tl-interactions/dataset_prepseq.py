"""Prepare the sequence part of a dataset: archive the sequences and split the proteins by their PDB hits.

Only BLAST hits with at least the given sequence identity count. A protein goes into ``seqs-train.txt`` if it has a hit
in a PDB chain deposited on or before the cutoff date. The others go into ``seqs-test.txt``, except those with a hit in a
chain of unknown deposition date (obsolete entries missing from the PDB index), which go into neither. The output
directory receives:

- ``sequences.zip``: the input sequence directory (its contents at the archive root)
- ``seqs-train.txt``, ``seqs-test.txt``: protein IDs, one per line
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

    # Date chains (not subjects): a subject may group identical chains deposited before and after the cutoff.
    # Chains of obsolete entries are missing from the index and have no deposition date.
    log(f"Reading PDB index {args.pdb_index}")
    keys = ["query", "subject"]
    similar_chains = hits.merge(hsps.loc[hsps["pident"] >= args.min_identity, keys].drop_duplicates(), on=keys)
    deposit_dates = similar_chains["accession"].str[:4].map(read_deposit_dates(args.pdb_index))
    known = set(similar_chains.loc[deposit_dates <= args.cutoff_date, "query"])
    train = proteins & known
    undated = proteins & set(similar_chains.loc[deposit_dates.isna(), "query"]) - known
    test = proteins - known - undated

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Writing {args.output_dir / 'sequences.zip'}")
    shutil.make_archive(args.output_dir / "sequences", "zip", root_dir=args.sequence_dir)
    for name, split in [("seqs-train.txt", train), ("seqs-test.txt", test)]:
        log(f"Writing {args.output_dir / name}")
        (args.output_dir / name).write_text("".join(f"{protein}\n" for protein in sorted(split)))

    criteria = f"pident >= {args.min_identity}, deposited <= {args.cutoff_date:%Y-%m-%d}"
    print(f"Train proteins (hits with {criteria}): {len(train):,}")
    print(f"Test proteins (no such hits): {len(test):,}")
    print(f"Excluded from test (hits with pident >= {args.min_identity} in chains of unknown date): {len(undated):,}")


if __name__ == "__main__":
    main()
