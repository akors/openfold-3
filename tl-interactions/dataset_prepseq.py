#!/usr/bin/env python3

"""Prepare the sequence part of a dataset: fetch the HuRI protein sequences and split the proteins by MMseqs2 cluster.

The proteins are those of the HuRI search space (``in_space_3``). They are clustered with MMseqs2 and searched against
the PDB. Test proteins are drawn from clusters without a PDB hit; validation and training proteins at random from the
pooled remaining clusters, so no cluster spans two splits. The dataset directory receives:

- ``sequences.zip``: one ``<protein_id>.fasta`` file per protein
- ``seqs-train.txt``, ``seqs-val.txt``, ``seqs-test.txt``: protein IDs, one per line
- ``proteins.tsv``: protein ID and gene ID of every protein
- ``interactions.tsv``: a copy of the HuRI interactions (gene pairs)
- ``meta.ini``: split and MMseqs2 parameters

Slow steps (download, PDB database, PDB search, clustering) are cached and skipped on reruns.
"""

import argparse
import configparser
import logging
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

from download_gencode_fasta import download_sequences

logger = logging.getLogger(__name__)

SPLITS = ["train", "val", "test"]

# MMseqs2 parameters, written to meta.ini. Names are the long mmseqs options (underscores for dashes), except the
# short-only ones in MMSEQS_FLAGS.
SEARCH_PARAMS = {"min_seq_id": 0.25, "coverage": 0.5, "cov_mode": 1, "sensitivity": 7.5}
CLUSTER_PARAMS = {
    "min_seq_id": 0.3,
    "coverage": 0.8,
    "cov_mode": 0,
    "cluster_mode": 1,
    "single_step_clustering": True,
    "sensitivity": 7.5,
}
MMSEQS_FLAGS = {"coverage": "-c", "sensitivity": "-s"}


class DatasetError(Exception):
    """Inconsistent inputs or cached outputs."""


def mmseqs_args(params):
    """Turn a parameter dict into mmseqs command-line arguments; ``True`` values become bare flags."""
    args = []
    for name, value in params.items():
        flag = MMSEQS_FLAGS.get(name, "--" + name.replace("_", "-"))
        args += [flag] if value is True else [flag, str(value)]
    return args


def run(cmd, log_path):
    """Run a command, echoing its combined output to stderr and to a log file.

    Raises
    ------
    subprocess.CalledProcessError
        If the command exits with a nonzero status.
    """
    logger.info("Running %s", " ".join(map(str, cmd)))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as log_file:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            sys.stderr.write(line)
            log_file.write(line)

    if process.wait():
        raise subprocess.CalledProcessError(process.returncode, cmd)


def rename_prefix(old, new):
    """Rename all files named ``<old>*`` to ``<new>*`` (MMseqs2 databases and outputs are file families)."""
    for path in old.parent.glob(old.name + "*"):
        path.rename(new.parent / (new.name + path.name[len(old.name):]))


def read_proteins(genes_tsv, proteins_tsv, interactions_tsv):
    """Read the proteins of the HuRI search space.

    Parameters
    ----------
    genes_tsv : Path
        HuRI Supplementary Table 1 (genes and their ``in_space_3`` flag).
    proteins_tsv : Path
        HuRI Supplementary Table 2 (ORFs with versioned transcript, protein and gene IDs).
    interactions_tsv : Path
        HuRI interactions, two unversioned gene IDs per line.

    Returns
    -------
    pandas.Series
        Unversioned gene ID, indexed by versioned protein ID.

    Raises
    ------
    DatasetError
        If a gene in the interactions has no protein.
    """
    genes = pd.read_csv(genes_tsv, sep="\t")
    space = set(genes.loc[genes["in_space_3"] == 1, "Ensembl_gene_id"])

    # Table 2 has versioned gene IDs, and some proteins appear once per ORF
    proteins = pd.read_csv(proteins_tsv, sep="\t")
    proteins["gene"] = proteins["ensembl_gene_id"].str.split(".").str[0]
    proteins = proteins.loc[proteins["gene"].isin(space), ["ensembl_protein_id", "gene"]].drop_duplicates()
    protein_gene = proteins.set_index("ensembl_protein_id")["gene"]

    interactions = pd.read_csv(interactions_tsv, sep="\t", header=None)
    missing = set(interactions.stack()) - set(protein_gene)
    if missing:
        raise DatasetError(f"{len(missing)} genes in {interactions_tsv} have no protein, e.g. {sorted(missing)[:5]}")
    return protein_gene


def prepare_sequences(proteins, zip_path, download_dir):
    """Make sure ``zip_path`` holds ``<id>.fasta`` for every protein, downloading them if the archive is missing.

    Raises
    ------
    DatasetError
        If the existing archive lacks sequences, or GENCODE lacks proteins.
    """
    names = {f"{protein}.fasta" for protein in proteins}

    # An existing archive is only checked, never updated
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            missing = names - set(archive.namelist())
        if missing:
            raise DatasetError(f"{len(missing)} sequences missing from {zip_path}: {' '.join(sorted(missing))}")
        logger.info("Using sequences in %s", zip_path)
        return

    logger.info("Downloading sequences into %s", download_dir)
    missing = set(proteins) - download_sequences(proteins, download_dir)
    if missing:
        raise DatasetError(f"{len(missing)} proteins not found in GENCODE: {' '.join(sorted(missing))}")

    # Archive exactly the required files, so stray files in the download directory stay out
    logger.info("Writing %s", zip_path)
    partial = zip_path.with_name(zip_path.name + ".partial")
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            archive.write(download_dir / name, name)
    partial.rename(zip_path)


def write_fasta(proteins, zip_path, fasta_path):
    """Write the sequences of the proteins from ``zip_path`` into one FASTA file, titled by bare protein ID."""
    with zipfile.ZipFile(zip_path) as archive, open(fasta_path, "w") as fasta:
        for protein in sorted(proteins):
            sequence = archive.read(f"{protein}.fasta").decode().splitlines()[1:]
            fasta.write(f">{protein}\n" + "".join(line.strip() for line in sequence) + "\n")


# The mmseqs steps below write to a .partial name and rename on success, so an interrupted run is not mistaken for a
# cached result.


def mmseqs_pdb_db(pdb_db, tmp_dir):
    """Download the MMseqs2 PDB database, unless ``<pdb_db>.dbtype`` exists."""
    if Path(f"{pdb_db}.dbtype").exists():
        return

    partial = pdb_db.with_name(pdb_db.name + ".partial")
    pdb_db.parent.mkdir(parents=True, exist_ok=True)
    run(["mmseqs", "databases", "PDB", partial, tmp_dir], pdb_db.with_suffix(".log"))
    rename_prefix(partial, pdb_db)


def mmseqs_search(fasta_path, pdb_db, m8_path, tmp_dir):
    """Search the proteins against the PDB, unless ``m8_path`` exists."""
    if m8_path.exists():
        logger.info("Using PDB search %s", m8_path)
        return

    partial = m8_path.with_name(m8_path.name + ".partial")
    run(
        ["mmseqs", "easy-search", fasta_path, pdb_db, partial, tmp_dir, *mmseqs_args(SEARCH_PARAMS)],
        m8_path.with_suffix(".log"),
    )
    partial.rename(m8_path)


def mmseqs_cluster(fasta_path, prefix, tmp_dir):
    """Cluster the proteins, unless ``<prefix>_cluster.tsv`` exists."""
    if Path(f"{prefix}_cluster.tsv").exists():
        logger.info("Using clustering %s_cluster.tsv", prefix)
        return

    # easy-cluster writes <prefix>_cluster.tsv, <prefix>_rep_seq.fasta and <prefix>_all_seqs.fasta
    partial = prefix.with_name(prefix.name + ".partial")
    run(
        ["mmseqs", "easy-cluster", fasta_path, partial, tmp_dir, *mmseqs_args(CLUSTER_PARAMS)],
        prefix.with_suffix(".log"),
    )
    rename_prefix(partial, prefix)


def read_hits(m8_path, proteins):
    """Read the proteins with a PDB hit from an MMseqs2 search.

    Raises
    ------
    DatasetError
        If the search has queries outside ``proteins`` (a stale cache).
    """
    hits = set(pd.read_csv(m8_path, sep="\t", header=None, usecols=[0])[0])
    if not hits <= set(proteins):
        raise DatasetError(f"PDB search has queries outside the protein set; delete {m8_path} to redo the search")
    return hits


def read_clusters(cluster_tsv, protein_gene):
    """Read the clusters, merging clusters that contain proteins of the same gene.

    Isoforms of different length can fail the coverage threshold and land in different clusters.

    Returns
    -------
    list of list of str
        Clusters of protein IDs, each sorted, in sorted order.

    Raises
    ------
    DatasetError
        If the clustered proteins are not exactly the protein set (a stale cache).
    """
    members = pd.read_csv(cluster_tsv, sep="\t", header=None, names=["rep", "member"])
    if set(members["member"]) != set(protein_gene.index):
        raise DatasetError(f"Clustered proteins differ from the protein set; delete {cluster_tsv} to redo the clustering")

    # Union-find over representatives and genes: a gene links the clusters of its proteins
    parent = {}

    def find(node):
        while parent.setdefault(node, node) != node:
            node = parent[node]
        return node

    for rep, member in zip(members["rep"], members["member"]):
        parent[find(("gene", protein_gene[member]))] = find(("rep", rep))

    merged = members.groupby(members["rep"].map(lambda rep: find(("rep", rep))))["member"]
    return sorted(sorted(group) for _, group in merged)


def split_clusters(clusters, hits, seed, test_fraction, val_fraction):
    """Split the clusters: test from hit-free clusters, then validation and training from the pooled rest.

    Parameters
    ----------
    clusters : list of list of str
        Clusters of protein IDs, in a fixed order so the split depends only on the seed.
    hits : set of str
        Proteins with a PDB hit.
    seed : int
        Seed of the shuffle.
    test_fraction, val_fraction : float
        Fractions of all proteins in the test and validation sets.

    Returns
    -------
    dict of str to list of str
        Protein IDs per split.

    Raises
    ------
    DatasetError
        If the hit-free clusters hold too few proteins for the test set.
    """
    total = sum(map(len, clusters))
    pool = list(clusters)
    random.Random(seed).shuffle(pool)
    split = {name: [] for name in SPLITS}

    # Test: whole hit-free clusters until the test fraction is reached
    hit_free = [cluster for cluster in pool if not hits.intersection(cluster)]
    for cluster in hit_free:
        if len(split["test"]) >= test_fraction * total:
            break
        split["test"] += cluster
    if len(split["test"]) < test_fraction * total:
        raise DatasetError(
            f"Hit-free clusters hold {len(split['test']):,} proteins, fewer than {test_fraction} of {total:,}"
        )

    # Validation, then training: all remaining clusters, with and without hits, in shuffled order
    test = set(split["test"])
    for cluster in pool:
        if test.issuperset(cluster):
            continue
        name = "val" if len(split["val"]) < val_fraction * total else "train"
        split[name] += cluster
    return split


def write_meta(path, args):
    """Write the split and MMseqs2 parameters to ``meta.ini``."""
    config = configparser.ConfigParser()
    config["dataset"] = {
        "splits": ", ".join(SPLITS),
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "val_fraction": args.val_fraction,
    }

    # <pdb_db>.version is written by `mmseqs databases` and recorded raw
    version = subprocess.run(["mmseqs", "version"], capture_output=True, text=True, check=True).stdout.strip()
    config["mmseqs"] = {"version": version, "pdb_db_version": Path(f"{args.pdb_db}.version").read_text().strip()}
    for section, params in [("mmseqs.search", SEARCH_PARAMS), ("mmseqs.cluster", CLUSTER_PARAMS)]:
        config[section] = {name: str(value).lower() for name, value in params.items()}

    with open(path, "w") as file:
        config.write(file)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("genes_tsv", type=Path, help="HuRI Supplementary Table 1 (genes)")
    parser.add_argument("proteins_tsv", type=Path, help="HuRI Supplementary Table 2 (ORFs and proteins)")
    parser.add_argument("interactions_tsv", type=Path, help="HuRI interactions (HuRI.tsv)")
    parser.add_argument("dataset_dir", type=Path, help="Output directory")
    parser.add_argument("--seed", type=int, default=0, help="Random seed of the split (default: %(default)s)")
    parser.add_argument(
        "--test-fraction", type=float, default=0.1, help="Fraction of all proteins in test (default: %(default)s)"
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.1, help="Fraction of all proteins in validation (default: %(default)s)"
    )
    parser.add_argument("--cache-dir", type=Path, help="Cache directory (default: DATASET_DIR/cache)")
    parser.add_argument("--pdb-db", type=Path, help="MMseqs2 PDB database (default: CACHE_DIR/mmseqs-db/pdb)")
    args = parser.parse_args()
    args.cache_dir = args.cache_dir or args.dataset_dir / "cache"
    args.pdb_db = args.pdb_db or args.cache_dir / "mmseqs-db" / "pdb"

    # Progress goes to stderr, results to stdout
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    try:
        build_dataset(args)
    except DatasetError as error:
        sys.exit(f"Error: {error}")


def build_dataset(args):
    """Run all steps, from the HuRI tables to the split in ``args.dataset_dir``.

    Raises
    ------
    DatasetError
        If an output file already exists, or on inconsistent inputs or cached outputs.
    """
    # Never overwrite a split: the user deletes the old one first
    split_paths = {name: args.dataset_dir / f"seqs-{name}.txt" for name in SPLITS}
    meta_path = args.dataset_dir / "meta.ini"
    proteins_path = args.dataset_dir / "proteins.tsv"
    interactions_path = args.dataset_dir / "interactions.tsv"
    outputs = [*split_paths.values(), meta_path, proteins_path, interactions_path]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise DatasetError(f"Output files exist, delete them first: {' '.join(existing)}")

    out_dir = args.cache_dir / "mmseqs-out"
    tmp_dir = args.cache_dir / "tmp"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading proteins")
    protein_gene = read_proteins(args.genes_tsv, args.proteins_tsv, args.interactions_tsv)
    proteins = sorted(protein_gene.index)
    print(f"Proteins: {len(proteins):,} of {protein_gene.nunique():,} genes")

    # Sequences: the archive is the dataset's copy, the FASTA file is the mmseqs input
    zip_path = args.dataset_dir / "sequences.zip"
    download_dir = args.cache_dir / "sequences"
    prepare_sequences(proteins, zip_path, download_dir)
    fasta_path = args.cache_dir / "sequences.fasta"
    write_fasta(proteins, zip_path, fasta_path)

    # PDB search: only the query column is used, to tell which proteins have a hit
    mmseqs_pdb_db(args.pdb_db, tmp_dir)
    m8_path = out_dir / "pdb-search.m8"
    mmseqs_search(fasta_path, args.pdb_db, m8_path, tmp_dir)
    hits = read_hits(m8_path, proteins)

    prefix = out_dir / "protein-clustering"
    mmseqs_cluster(fasta_path, prefix, tmp_dir)
    clusters = read_clusters(f"{prefix}_cluster.tsv", protein_gene)
    split = split_clusters(clusters, hits, args.seed, args.test_fraction, args.val_fraction)

    for name, path in split_paths.items():
        logger.info("Writing %s", path)
        path.write_text("".join(f"{protein}\n" for protein in sorted(split[name])))
    write_meta(meta_path, args)

    # The pairs step reads the interactions and the protein-to-gene mapping from the dataset
    logger.info("Writing %s and %s", proteins_path, interactions_path)
    protein_gene.sort_index().to_csv(proteins_path, sep="\t", header=["gene_id"], index_label="protein_id")
    shutil.copyfile(args.interactions_tsv, interactions_path)

    # The per-protein downloads stay as a cache for later steps; the merged FASTA file is only the mmseqs input
    logger.info("Removing %s", fasta_path)
    fasta_path.unlink()

    print(f"Proteins with PDB hits: {len(hits):,}")
    print(f"Clusters: {len(clusters):,}, hit-free: {sum(not hits.intersection(c) for c in clusters):,}")
    for name in SPLITS:
        print(f"{name}: {len(split[name]):,} proteins")


if __name__ == "__main__":
    main()
