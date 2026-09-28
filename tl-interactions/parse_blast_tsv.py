"""Parse a tabular BLAST report into hit and HSP tables and pickle them.

The report must have the columns in :data:`OUTFMT`, e.g. formatted from a zstd-compressed BLAST archive with::

    zstd -dc hits.asn.zst | blast_formatter -archive - -outfmt "<OUTFMT>" | zstd -o hits.tsv.zst

The pickle holds a dict ``{"hits": DataFrame, "hsps": DataFrame}``; see :func:`read_blast_tsv`.
"""

import argparse
from compression import zstd
from contextlib import nullcontext
from pathlib import Path

import pandas as pd

OUTFMT = (
    "6 qseqid sallacc evalue bitscore score pident nident positive length mismatch gapopen gaps "
    "qstart qend sstart send qlen slen staxids stitle"
)


def read_blast_tsv(path):
    """Read a tabular BLAST report with the columns in :data:`OUTFMT` into hit and HSP tables.

    Queries are identified by the first ``|``-separated field of their FASTA title (e.g. the ENSP ID of a GENCODE
    header). A BLAST subject is one database sequence, which may stand for several identical PDB chains; it is named
    by its first chain accession. ``hits`` has one row per chain and ``hsps`` one row per local alignment. Join them on
    ``["query", "subject"]``.

    Parameters
    ----------
    path : str or Path
        Tabular BLAST report, optionally compressed (``.zst``, ``.gz``, ``.bz2``, ``.xz``).

    Returns
    -------
    hits : pandas.DataFrame
        Columns ``query, subject, accession``.
    hsps : pandas.DataFrame
        Columns ``query, subject`` followed by the remaining :data:`OUTFMT` fields except ``sallacc``. ``staxids`` is
        the ``;``-separated set of taxids over all chains of the subject.
    """
    names = OUTFMT.split()[1:]
    path = Path(path)
    # pandas only reads zstd through the optional ``zstandard`` package, so decompress with the standard library
    with zstd.open(path, "rt") if path.suffix == ".zst" else nullcontext(path) as source:
        hsps = pd.read_csv(source, sep="\t", names=names, dtype={"staxids": str})

    hsps.insert(0, "query", hsps.pop("qseqid").str.split("|", n=1).str[0])
    accessions = hsps.pop("sallacc").str.split(";")
    hsps.insert(1, "subject", accessions.str[0])

    hits = hsps[["query", "subject"]].assign(accession=accessions).drop_duplicates(["query", "subject"])
    hits = hits.explode("accession", ignore_index=True)
    return hits, hsps


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, epilog=f'OUTFMT = "{OUTFMT}"', formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("blast_tsv", type=Path, help="Tabular BLAST report, optionally compressed")
    parser.add_argument("output", type=Path, help="Output pickle file")
    args = parser.parse_args()

    hits, hsps = read_blast_tsv(args.blast_tsv)
    pd.to_pickle({"hits": hits, "hsps": hsps}, args.output)
    print(f"Wrote {len(hits)} hit rows and {len(hsps)} HSP rows for {hits['query'].nunique()} queries to {args.output}")


if __name__ == "__main__":
    main()
