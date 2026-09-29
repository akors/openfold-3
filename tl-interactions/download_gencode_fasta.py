"""Download GENCODE human protein sequences for a list of versioned Ensembl protein IDs.

Writes one ``<protein_id>.fasta`` file per ID into the output directory.
"""

import argparse
import gzip
import sys
import urllib.request
from pathlib import Path

GENCODE_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{release}/"
    "gencode.v{release}.pc_translations.fa.gz"
)


def iter_fasta(lines):
    """Yield ``(header, sequence_lines)`` tuples from an iterable of FASTA lines.

    Parameters
    ----------
    lines : Iterable[str]
        Lines of a FASTA file.

    Yields
    ------
    tuple[str, list[str]]
        Header line (including ``>``) and the sequence lines that follow it.
    """
    header, seq = None, []
    for line in lines:
        line = line.rstrip("\n")
        if line.startswith(">"):
            if header is not None:
                yield header, seq
            header, seq = line, []
        elif line:
            seq.append(line)
    if header is not None:
        yield header, seq


def download_sequences(protein_ids, output_dir, release="27"):
    """Download GENCODE sequences into one ``<protein_id>.fasta`` file per ID.

    Parameters
    ----------
    protein_ids : Iterable[str]
        Versioned Ensembl protein IDs.
    output_dir : Path
        Directory to write the FASTA files to, created if missing.
    release : str
        GENCODE human release.

    Returns
    -------
    set of str
        The IDs found in GENCODE.
    """
    wanted = set(protein_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    found = set()
    url = GENCODE_URL.format(release=release)
    with urllib.request.urlopen(url) as response, gzip.open(response, "rt") as fasta:
        for header, seq in iter_fasta(fasta):
            # GENCODE headers look like >ENSP...|ENST...|ENSG...|...
            protein_id = header[1:].split("|", 1)[0]
            if protein_id in wanted:
                (output_dir / f"{protein_id}.fasta").write_text("\n".join([header, *seq]) + "\n")
                found.add(protein_id)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "protein_ids", type=argparse.FileType("r"), help="File with one versioned ENSP ID per line, or - for stdin"
    )
    parser.add_argument("output_dir", type=Path, help="Directory to write <protein_id>.fasta files to")
    parser.add_argument("--release", default="27", help="GENCODE human release (default: %(default)s)")
    args = parser.parse_args()

    with args.protein_ids:
        wanted = {line.strip() for line in args.protein_ids if line.strip()}
    found = download_sequences(wanted, args.output_dir, args.release)

    missing = wanted - found
    print(f"Wrote {len(found)} of {len(wanted)} sequences to {args.output_dir}")
    if missing:
        print(
            f"{len(missing)} IDs not found in GENCODE v{args.release}, e.g. {sorted(missing)[:5]}", file=sys.stderr
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
