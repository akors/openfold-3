#!/usr/bin/env python3

"""Build the multimers (protein pairs) of a dataset, per split: HuRI interactions as positives, and uniformly sampled
negatives.

Reads the output of ``dataset_prepseq.py`` from the dataset directory. Genes with more than one protein and
homodimers are left out. The mode decides which proteins a split's multimers are made of:

- ``pure``: both proteins from the split (Park & Marcotte's C3);
- ``mixed``: one protein from train and the other from the split (C2); train multimers are the same in both modes.

Negatives are drawn uniformly from the split's allowed pairs that are not HuRI interactions, ``negatives_ratio`` per
positive. ``DATASET_DIR/multimers/NAME`` receives:

- ``multimers-train.tsv``, ``multimers-val.tsv``, ``multimers-test.tsv``: ``protein_a``, ``protein_b``, ``label``
  (1 or 0); in mixed mode ``protein_a`` is the train protein
- ``meta.ini``: parameters and multimer counts
"""

import argparse
import configparser
import logging
import random
import sys
import zipfile
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from dataset_prepseq import SPLITS, DatasetError

logger = logging.getLogger(__name__)

MODES = ["pure", "mixed"]

# Draws per negative needed before sampling gives up
MAX_DRAWS_PER_NEGATIVE = 100

# Two protein IDs, in the order they are written
Pair = tuple[str, str]

# (pool_a, pool_b, same_pool): the proteins a split's pairs take their first and second protein from
Pools = tuple[list[str], list[str], bool]


def read_lengths(proteins: Iterable[str], sequences_dir: Path, zip_path: Path) -> dict[str, int]:
    """Read the sequence lengths of the proteins from ``<sequences_dir>/<id>.fasta``.

    ``sequences_dir`` is the per-protein cache that ``dataset_prepseq.py`` downloads into. If it does not exist,
    ``zip_path`` is unpacked into it first. An existing cache is used as is.

    Parameters
    ----------
    proteins : iterable of str
        Protein IDs to read.
    sequences_dir : Path
        Directory with one ``<protein_id>.fasta`` file per protein.
    zip_path : Path
        The dataset's ``sequences.zip``, unpacked if ``sequences_dir`` does not exist.

    Returns
    -------
    dict of str to int
        Sequence length, keyed by protein ID.

    Raises
    ------
    DatasetError
        If the cache lacks a protein.
    """
    if not sequences_dir.exists():
        logger.info("Unpacking %s into %s", zip_path, sequences_dir)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(sequences_dir)

    lengths = {}
    for protein in proteins:
        try:
            sequence = (sequences_dir / f"{protein}.fasta").read_text().splitlines()[1:]
        except FileNotFoundError:
            raise DatasetError(f"{protein} missing from {sequences_dir}; delete it to unpack {zip_path} again")
        lengths[protein] = sum(len(line.strip()) for line in sequence)
    return lengths


def read_dataset(
    dataset_dir: Path, cache_dir: Path
) -> tuple[dict[str, set[str]], dict[str, str], set[frozenset[str]], dict[str, int]]:
    """Read the split, the protein-to-gene mapping, the interactions and the sequence lengths.

    Parameters
    ----------
    dataset_dir : Path
        Dataset directory written by ``dataset_prepseq.py``.
    cache_dir : Path
        Cache directory; the sequences are read from its ``sequences`` subdirectory.

    Returns
    -------
    split : dict of str to set of str
        Protein IDs per split.
    protein_gene : dict of str to str
        Gene ID of every protein.
    interactions : set of frozenset of str
        HuRI interactions as unordered gene pairs, homodimers included.
    lengths : dict of str to int
        Sequence length of every protein.

    Raises
    ------
    DatasetError
        If the split and the mapping do not cover the same proteins, or sequences are missing.
    """
    # Protein IDs of each split, one per line
    split = {}
    for name in SPLITS:
        split[name] = set((dataset_dir / f"seqs-{name}.txt").read_text().split())

    proteins = pd.read_csv(dataset_dir / "proteins.tsv", sep="\t")
    protein_gene = dict(zip(proteins["protein_id"], proteins["gene_id"]))

    # HuRI lists each interaction as two gene IDs; a frozenset makes (A, B) and (B, A) the same pair
    pairs = pd.read_csv(dataset_dir / "interactions.tsv", sep="\t", header=None)
    interactions = set()
    for gene_a, gene_b in pairs.itertuples(index=False):
        interactions.add(frozenset((gene_a, gene_b)))

    all_split_proteins = set.union(*split.values())
    if all_split_proteins != set(protein_gene):
        raise DatasetError(f"Split and proteins.tsv in {dataset_dir} do not cover the same proteins")

    lengths = read_lengths(sorted(protein_gene), cache_dir / "sequences", dataset_dir / "sequences.zip")
    return split, protein_gene, interactions, lengths


def one_protein_per_gene(protein_gene: dict[str, str]) -> dict[str, str]:
    """Map each gene to its protein, leaving out genes with more than one protein.

    HuRI screened one ORF per gene, but a few genes have two proteins in Supplementary Table 2, and the table does not
    say which one was screened. Those genes are left out of the multimers altogether.

    Parameters
    ----------
    protein_gene : dict of str to str
        Gene ID of every protein.

    Returns
    -------
    dict of str to str
        Protein ID, keyed by gene ID, for the genes with exactly one protein.
    """
    gene_proteins = {}
    for protein, gene in protein_gene.items():
        gene_proteins.setdefault(gene, []).append(protein)

    gene_protein = {}
    dropped = {}
    for gene, proteins in sorted(gene_proteins.items()):
        if len(proteins) == 1:
            gene_protein[gene] = proteins[0]
        else:
            dropped[gene] = sorted(proteins)

    n_proteins = sum(len(proteins) for proteins in dropped.values())
    logger.info("Leaving out %d genes (%d proteins) with more than one protein", len(dropped), n_proteins)
    for gene, proteins in dropped.items():
        logger.debug("%s", "\t".join([gene, *proteins]))
    return gene_protein


def pools(mode: str, split: dict[str, set[str]], gene_protein: dict[str, str]) -> dict[str, Pools]:
    """The proteins that the pairs of each split are drawn from.

    A pair takes its first protein from ``pool_a`` and its second from ``pool_b``. The pools are the same for train
    and for every split in pure mode. In mixed mode, validation and test pairs take their first protein from train.

    Parameters
    ----------
    mode : {"pure", "mixed"}
        Which proteins the validation and test pairs use.
    split : dict of str to set of str
        Protein IDs per split.
    gene_protein : dict of str to str
        Protein ID of every gene; only these proteins enter the pools.

    Returns
    -------
    dict of str to tuple of (list of str, list of str, bool)
        ``(pool_a, pool_b, same_pool)`` per split. The pools are sorted lists of proteins, so that
        sampling from them depends only on the seed.
    """
    reps = set(gene_protein.values())
    reps_in = {}
    for name in SPLITS:
        reps_in[name] = sorted(reps & split[name])

    split_pools = {}
    for name in SPLITS:
        if mode == "mixed" and name != "train":
            split_pools[name] = (reps_in["train"], reps_in[name], False)
        else:
            split_pools[name] = (reps_in[name], reps_in[name], True)
    return split_pools


def orient(a: str, b: str, pool_a: set[str], pool_b: set[str], same_pool: bool) -> Pair | None:
    """Put a pair in the order it is written, or return ``None`` if it does not belong to the pools.

    If both proteins come from the same pool, the pair is sorted by protein ID. Otherwise the protein from ``pool_a``
    comes first.

    Parameters
    ----------
    a, b : str
        Protein IDs, in any order.
    pool_a, pool_b : set of str
        The pools of a split, as sets for fast lookups.
    same_pool : bool
        Whether both proteins come from the same pool.

    Returns
    -------
    tuple of (str, str) or None
        The ordered pair, or ``None`` if it does not belong to the pools.
    """
    if same_pool:
        if a in pool_a and b in pool_a:
            return tuple(sorted((a, b)))
        return None

    # Swap so that the candidate for pool_a comes first
    if a in pool_b:
        a, b = b, a
    if a in pool_a and b in pool_b:
        return a, b
    return None


def build_positives(
    interactions: set[frozenset[str]],
    gene_protein: dict[str, str],
    split_pools: dict[str, Pools],
    lengths: dict[str, int],
    length_cutoff: int,
) -> dict[str, set[Pair]]:
    """Map the HuRI interactions to protein pairs and assign them to splits.

    Homodimers, interactions of genes missing from ``gene_protein``, pairs that belong to no split and pairs longer
    than ``length_cutoff`` are left out.

    Parameters
    ----------
    interactions : set of frozenset of str
        HuRI interactions as unordered gene pairs.
    gene_protein : dict of str to str
        Protein ID of every gene that takes part.
    split_pools : dict of str to tuple of (list of str, list of str, bool)
        ``(pool_a, pool_b, same_pool)`` per split, from :func:`pools`.
    lengths : dict of str to int
        Sequence length of every protein.
    length_cutoff : int
        Maximum summed sequence length of a pair.

    Returns
    -------
    dict of str to set of tuple of (str, str)
        Positive pairs per split, oriented as by :func:`orient`.
    """
    # The same pools as sets, for fast membership tests
    pool_sets = {}
    for name, (pool_a, pool_b, same_pool) in split_pools.items():
        pool_sets[name] = (set(pool_a), set(pool_b), same_pool)

    positives = {name: set() for name in SPLITS}
    for genes in interactions:
        # A homodimer is a frozenset with one gene
        if len(genes) != 2:
            continue

        gene_a, gene_b = genes
        if gene_a not in gene_protein or gene_b not in gene_protein:
            continue
        a, b = gene_protein[gene_a], gene_protein[gene_b]
        if lengths[a] + lengths[b] > length_cutoff:
            continue

        # The splits' pairs do not overlap, so a pair belongs to at most one split
        for name, (pool_a, pool_b, same_pool) in pool_sets.items():
            pair = orient(a, b, pool_a, pool_b, same_pool)
            if pair is not None:
                positives[name].add(pair)
                break
    return positives


def sample_negatives(
    count: int,
    pool_a: list[str],
    pool_b: list[str],
    same_pool: bool,
    interactions: set[frozenset[str]],
    protein_gene: dict[str, str],
    lengths: dict[str, int],
    length_cutoff: int,
    rng: random.Random,
) -> set[Pair]:
    """Draw ``count`` distinct pairs uniformly from ``pool_a`` × ``pool_b``.

    Draws are rejected if they are homodimers, HuRI interactions (at the gene level), longer than ``length_cutoff``, or
    already drawn. Rejecting draws keeps the accepted pairs uniform over all allowed pairs.

    Parameters
    ----------
    count : int
        Number of negatives to draw.
    pool_a, pool_b : list of str
        Sorted proteins to draw the first and the second protein from.
    same_pool : bool
        Whether both pools are the same; the pairs are then sorted by protein ID.
    interactions : set of frozenset of str
        HuRI interactions as unordered gene pairs, which are never drawn.
    protein_gene : dict of str to str
        Gene ID of every protein.
    lengths : dict of str to int
        Sequence length of every protein.
    length_cutoff : int
        Maximum summed sequence length of a pair.
    rng : random.Random
        Random number generator to draw with.

    Returns
    -------
    set of tuple of (str, str)
        The negative pairs, ordered like the positives.

    Raises
    ------
    DatasetError
        If too few draws are accepted.
    """
    negatives = set()
    for _ in range(MAX_DRAWS_PER_NEGATIVE * count):
        if len(negatives) == count:
            break

        a = rng.choice(pool_a)
        b = rng.choice(pool_b)
        if a == b:
            continue
        if lengths[a] + lengths[b] > length_cutoff:
            continue
        if frozenset((protein_gene[a], protein_gene[b])) in interactions:
            continue

        # Same order as the positives, so that a pair drawn twice is recognized
        if same_pool:
            a, b = sorted((a, b))
        negatives.add((a, b))

    if len(negatives) < count:
        raise DatasetError(f"Only {len(negatives):,} of {count:,} negatives found; lower the ratio or raise the cutoff")
    return negatives


def write_multimers(path: Path, positives: Iterable[Pair], negatives: Iterable[Pair]) -> None:
    """Write the labelled multimers, sorted by protein IDs.

    Parameters
    ----------
    path : Path
        Output TSV file, with columns ``protein_a``, ``protein_b`` and ``label``.
    positives, negatives : iterable of tuple of (str, str)
        Pairs to write with label 1 and 0.
    """
    rows = []
    for a, b in positives:
        rows.append((a, b, 1))
    for a, b in negatives:
        rows.append((a, b, 0))

    multimers = pd.DataFrame(rows, columns=["protein_a", "protein_b", "label"])
    multimers = multimers.sort_values(["protein_a", "protein_b"])
    multimers.to_csv(path, sep="\t", index=False)


def write_meta(path: Path, params: dict[str, object], counts: dict[tuple[str, str], int]) -> None:
    """Write the multimer parameters and counts to ``meta.ini``.

    Parameters
    ----------
    path : Path
        Output file.
    params : dict of str to object
        Parameters of the build, written to the ``[multimers]`` section.
    counts : dict of (str, str) to int
        Multimer counts, keyed by split and ``"positives"`` or ``"negatives"``, written to the ``[counts]`` section.
    """
    config = configparser.ConfigParser()
    config["multimers"] = params
    # Keys like train_positives, val_negatives
    config["counts"] = {}
    for (split_name, kind), count in counts.items():
        config["counts"][f"{split_name}_{kind}"] = str(count)
    with open(path, "w") as file:
        config.write(file)


def positives_to_keep(
    positives: dict[str, set[Pair]], splits: list[str], negatives_ratio: int, total: int | None
) -> dict[str, int]:
    """Number of positives to keep per split, so that the selected splits hold at most ``total`` multimers.

    All splits are scaled by the same fraction, so the proportions between them stay roughly those of an unlimited
    build. Every split keeps at least one positive (if it has any). Whatever rounding and that rule add beyond
    ``total`` is taken off the largest splits.

    Parameters
    ----------
    positives : dict of str to set of tuple of (str, str)
        All positive pairs per split.
    splits : list of str
        Selected splits.
    negatives_ratio : int
        Negatives per positive.
    total : int or None
        At most this many multimers (positives and negatives) over the selected splits; ``None`` keeps all.

    Returns
    -------
    dict of str to int
        Number of positives to keep, keyed by split.

    Raises
    ------
    DatasetError
        If ``total`` is too small for one positive and its negatives in every split.
    """
    full_counts = {}
    for split_name in splits:
        full_counts[split_name] = len(positives[split_name])

    # Every positive comes with negatives_ratio negatives
    full_total = (1 + negatives_ratio) * sum(full_counts.values())
    if total is None or total >= full_total:
        logger.info("Keeping all %s multimers", f"{full_total:,}")
        return full_counts

    fraction = total / full_total
    keep = {}
    for split_name, count in full_counts.items():
        keep[split_name] = min(count, max(1, round(fraction * count)))

    # Rounding up and the at-least-one rule can overshoot; take the surplus from the largest splits
    surplus = max(0, sum(keep.values()) - total // (1 + negatives_ratio))
    for split_name in sorted(keep, key=keep.get, reverse=True):
        removed = min(surplus, max(0, keep[split_name] - 1))
        keep[split_name] -= removed
        surplus -= removed
    if surplus > 0:
        raise DatasetError(f"--total {total} is too small to keep one positive and its negatives in every split")
    return keep


def build_multimers(
    dataset_dir: Path,
    mode: str,
    splits: Iterable[str] = SPLITS,
    total: int | None = None,
    seed: int = 0,
    length_cutoff: int = 2048,
    negatives_ratio: int = 3,
    name: str | None = None,
    cache_dir: Path | None = None,
) -> dict[tuple[str, str], int]:
    """Build the multimers of a dataset into ``<dataset_dir>/multimers/<name>``.

    Parameters
    ----------
    dataset_dir : Path
        Dataset directory written by ``dataset_prepseq.py``.
    mode : {"pure", "mixed"}
        Which proteins the validation and test multimers use.
    splits : iterable of str
        Splits to build; all by default.
    total : int, optional
        At most this many multimers (positives and negatives) over the selected splits; all by default.
    seed : int
        Random seed of the sampling.
    length_cutoff : int
        Maximum summed sequence length of a multimer.
    negatives_ratio : int
        Negatives per positive multimer.
    name : str, optional
        Name of the multimer set; the mode by default.
    cache_dir : Path, optional
        Cache directory; ``<dataset_dir>/cache`` by default.

    Returns
    -------
    dict of (str, str) to int
        Multimer counts of the selected splits, keyed by split and ``"positives"`` or ``"negatives"``.

    Raises
    ------
    DatasetError
        If an output file already exists, on inconsistent inputs, or if negatives cannot be sampled.
    """
    name = name or mode
    splits = [split_name for split_name in SPLITS if split_name in splits]
    cache_dir = cache_dir or dataset_dir / "cache"
    out_dir = dataset_dir / "multimers" / name
    multimer_paths = {}
    for split_name in splits:
        multimer_paths[split_name] = out_dir / f"multimers-{split_name}.tsv"
    meta_path = out_dir / "meta.ini"

    # Never overwrite a multimer set: the user deletes the old one first
    existing = []
    for path in [*multimer_paths.values(), meta_path]:
        if path.exists():
            existing.append(str(path))
    if existing:
        raise DatasetError(f"Output files exist, delete them first: {' '.join(existing)}")

    logger.info("Reading %s", dataset_dir)
    split, protein_gene, interactions, lengths = read_dataset(dataset_dir, cache_dir)
    gene_protein = one_protein_per_gene(protein_gene)
    logger.info("Genes: %s, HuRI interactions: %s", f"{len(gene_protein):,}", f"{len(interactions):,}")

    # Positives first, then as many negatives per split as the ratio asks for
    split_pools = pools(mode, split, gene_protein)
    positives = build_positives(interactions, gene_protein, split_pools, lengths, length_cutoff)

    keep = positives_to_keep(positives, splits, negatives_ratio, total)

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split_name in splits:
        # One random stream per split, so that a split does not depend on which other splits are built
        rng = random.Random(f"{seed}-{split_name}")

        # The shuffle draws the same random numbers however many positives are kept, and negatives are accepted in a
        # fixed order, so a smaller total gives a subset of a larger one
        split_positives = sorted(positives[split_name])
        rng.shuffle(split_positives)
        split_positives = split_positives[:keep[split_name]]

        pool_a, pool_b, same_pool = split_pools[split_name]
        count = negatives_ratio * len(split_positives)

        logger.info("Sampling %s negatives for %s", f"{count:,}", split_name)
        negatives = sample_negatives(
            count, pool_a, pool_b, same_pool, interactions, protein_gene, lengths, length_cutoff, rng
        )

        logger.info("Writing %s", multimer_paths[split_name])
        write_multimers(multimer_paths[split_name], split_positives, negatives)
        counts[split_name, "positives"] = len(split_positives)
        counts[split_name, "negatives"] = len(negatives)

    params = {
        "mode": mode,
        "splits": ", ".join(splits),
        "total": "none" if total is None else total,
        "seed": seed,
        "length_cutoff": length_cutoff,
        "negatives_ratio": negatives_ratio,
    }
    write_meta(meta_path, params, counts)
    return counts


def main() -> None:
    """Parse the command line, build the multimers and print their counts per split."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path, help="Dataset directory written by dataset_prepseq.py")
    parser.add_argument("--mode", choices=MODES, required=True, help="Which proteins the val and test multimers use")
    parser.add_argument(
        "--splits", nargs="+", choices=SPLITS, default=SPLITS, help="Splits to build (default: all)"
    )
    parser.add_argument(
        "--total",
        type=int,
        help="At most this many multimers over the selected splits, in the proportions of a full build (default: all)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed of the sampling (default: %(default)s)")
    parser.add_argument(
        "--length-cutoff",
        type=int,
        default=2048,
        help="Maximum summed sequence length of a multimer (default: %(default)s)",
    )
    parser.add_argument(
        "--negatives-ratio", type=int, default=3, help="Negatives per positive multimer (default: %(default)s)"
    )
    parser.add_argument("--name", help="Name of the multimer set under DATASET_DIR/multimers (default: the mode)")
    parser.add_argument("--cache-dir", type=Path, help="Cache directory (default: DATASET_DIR/cache)")
    args = parser.parse_args()

    # Progress goes to stderr, results to stdout
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    try:
        counts = build_multimers(
            args.dataset_dir,
            args.mode,
            splits=args.splits,
            total=args.total,
            seed=args.seed,
            length_cutoff=args.length_cutoff,
            negatives_ratio=args.negatives_ratio,
            name=args.name,
            cache_dir=args.cache_dir,
        )
    except DatasetError as error:
        sys.exit(f"Error: {error}")

    for (split_name, kind), count in counts.items():
        if kind == "positives":
            print(f"{split_name}: {count:,} positives, {counts[split_name, 'negatives']:,} negatives")


if __name__ == "__main__":
    main()
