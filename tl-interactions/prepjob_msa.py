#!/usr/bin/env python3

"""Prepare an MSA job for OpenFold3's Snakemake workflow (``scripts/snakemake_msa/MSA_Snakefile``) from a dataset.

Reads the multimers written by ``dataset_buildmultimers.py`` and the sequences written by ``dataset_prepseq.py``, and
writes two files:

- ``JOB.fasta``: the sequence of every protein in the selected multimers, titled by protein ID
- ``JOB.json``: the Snakemake config, to run with ``snakemake -s MSA_Snakefile --configfile JOB.json``

Paths are written as given, so relative paths are relative to the directory Snakemake runs in.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from dataset_prepseq import DatasetError, write_fasta

logger = logging.getLogger(__name__)

# The example in docs/source/precomputed_msa_generation_how_to.md
DEFAULT_OPENFOLD_ENV = "~/miniforge3/envs/of3"

# Defaults from scripts/snakemake_msa/example_msa_config_protein.json; the databases are those that
# download_of3_databases.py fetches by default
DEFAULT_CONFIG = {
    "databases": ["uniref90", "uniprot", "mgnify"],
    "jackhmmer_output_format": "sto",
    "jackhmmer_threads": 4,
    "hhblits_threads": 16,
    "tmpdir": "/tmp",
    "run_template_search": False,
}


def select_multimer_files(dataset_dir, specs):
    """Find the multimer files selected by ``specs``.

    Parameters
    ----------
    dataset_dir : Path
        Dataset directory.
    specs : list of str or None
        ``NAME`` (all splits of the multimer set) or ``NAME/SPLIT``; ``None`` selects all multimer files.

    Returns
    -------
    list of Path
        Sorted multimer files.

    Raises
    ------
    DatasetError
        If a spec matches no file.
    """
    multimers_dir = dataset_dir / "multimers"
    if specs is None:
        paths = set(multimers_dir.glob("*/multimers-*.tsv"))
        if not paths:
            raise DatasetError(f"No multimer files in {multimers_dir}")
        return sorted(paths)

    paths = set()
    for spec in specs:
        name, _, split = spec.partition("/")
        matches = list(multimers_dir.glob(f"{name}/multimers-{split or '*'}.tsv"))
        if not matches:
            raise DatasetError(f"No multimer files match {spec!r} in {multimers_dir}")
        paths.update(matches)
    return sorted(paths)


def read_proteins(paths):
    """Read the proteins of all multimers in ``paths``, as a sorted list of protein IDs."""
    proteins = set()
    for path in paths:
        multimers = pd.read_csv(path, sep="\t")
        proteins.update(multimers["protein_a"], multimers["protein_b"])
    return sorted(proteins)


def parse_set(item):
    """Parse ``KEY=VALUE`` into a key and a value; the value is parsed as JSON if possible, else kept as a string."""
    key, sep, value = item.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {item!r}")
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path, help="Dataset directory")
    parser.add_argument("job", type=Path, help="Path and name of the job; JOB.json and JOB.fasta are written")
    parser.add_argument("output_dir", type=Path, help="Directory the MSAs are written to, one per protein")
    parser.add_argument("--database-path", required=True, help="Base directory of the alignment databases")
    parser.add_argument(
        "--multimers",
        nargs="+",
        metavar="SPEC",
        help="Multimer sets (NAME) or single splits (NAME/SPLIT) to take the proteins from (default: all)",
    )
    parser.add_argument(
        "--openfold-env", default=DEFAULT_OPENFOLD_ENV, help="OpenFold3 conda environment (default: %(default)s)"
    )
    parser.add_argument(
        "--set",
        type=parse_set,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Add a key to the config, overriding defaults; VALUE is parsed as JSON if possible, else a string",
    )
    args = parser.parse_args()

    # Progress goes to stderr, results to stdout
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    try:
        prepare_job(args)
    except DatasetError as error:
        sys.exit(f"Error: {error}")


def prepare_job(args):
    """Write ``JOB.fasta`` and ``JOB.json``.

    Raises
    ------
    DatasetError
        If an output file already exists, or the multimer selection matches nothing.
    """
    # Never overwrite a job: the user deletes the old one first
    fasta_path = args.job.with_name(args.job.name + ".fasta")
    config_path = args.job.with_name(args.job.name + ".json")
    existing = [str(path) for path in (fasta_path, config_path) if path.exists()]
    if existing:
        raise DatasetError(f"Output files exist, delete them first: {' '.join(existing)}")

    paths = select_multimer_files(args.dataset_dir, args.multimers)
    for path in paths:
        logger.info("Reading %s", path)
    proteins = read_proteins(paths)

    args.job.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing %s", fasta_path)
    write_fasta(proteins, args.dataset_dir / "sequences.zip", fasta_path)

    # The Snakefile also runs tools without a shell, which would not expand ~
    config = {
        "input_fasta": str(fasta_path),
        "openfold_env": str(Path(args.openfold_env).expanduser()),
        "base_database_path": args.database_path,
        "output_directory": str(args.output_dir),
        **DEFAULT_CONFIG,
        **dict(args.set),
    }
    logger.info("Writing %s", config_path)
    config_path.write_text(json.dumps(config, indent=4) + "\n")

    print(f"Multimer files: {len(paths)}, proteins: {len(proteins):,}")


if __name__ == "__main__":
    main()
