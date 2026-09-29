#!/usr/bin/env bash
# Cluster the HuRI sequences with MMseqs2 for a homology-aware train/val/test split, and search them against the PDB.
#
# Clustering: connected-component clustering (--cluster-mode 1) at 30% identity and 80% coverage of both sequences.
# Single-step clustering compares all sequences, not just the representatives of earlier cascade rounds, so no
# above-threshold hit links two clusters.
#
# PDB search: hits with at least 25% identity covering at least 50% of the PDB chain (--cov-mode 1), in BLAST tabular
# format. Paths are relative to the repository root.
set -euo pipefail

cd "$(dirname "$0")/.."

in_fasta=data/sequences/huri.fasta
pdb_db=data/mmseqs-db/PDB
out_dir=data/mmseqs-out/huri-v1

mkdir -p "$out_dir"
mmseqs easy-cluster "$in_fasta" "$out_dir/clu30" "$out_dir/tmp" \
    --min-seq-id 0.3 -c 0.8 --cov-mode 0 --cluster-mode 1 --single-step-clustering -s 7.5 \
    2>&1 | tee "$out_dir/clu30.log"

mmseqs easy-search "$in_fasta" "$pdb_db" "$out_dir/pdb25.m8" "$out_dir/tmp" \
    --min-seq-id 0.25 -c 0.5 --cov-mode 0 -s 7.5 \
    2>&1 | tee "$out_dir/pdb25.log"
