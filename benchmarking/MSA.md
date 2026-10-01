# MSA generation benchmarks

OpenFold3 protein MSAs, made with [MSA_Snakefile_benchmark](MSA_Snakefile_benchmark): OpenFold3's
`scripts/snakemake_msa/MSA_Snakefile`, plus a Snakemake benchmark record for each alignment. Each protein gets three
jackhmmer searches (HMMER 3.4), against uniref90, uniprot and mgnify, with the default config from
`tl-interactions/prepjob_msa.py`.

Input: dataset `huri-v1`, multimer set `mixed50` (50 pairs), which has **96 unique proteins** with 43,540 residues in
total. Lengths are 75–1566, mean 454 and median 399.

## Arrhenius

Job 3237961, 2026-10-01. Raw records are in `data/benchmarks/msa-huri-v1-mixed50.arrhenius/` (not in git):
- `msa_huri-v1_mixed50.sbatch`: the job script
- `benchmarks/`: one record per alignment
- `run-3237961/`: metadata, phase times, and cgroup usage sampled every 30 s
- `slurm-3237961.out`: the job's Slurm log

### Hardware and setup

| | |
|---|---|
| Node | `n261`, `cpu` partition, x86_64 |
| CPU | 2× AMD EPYC 9755 (Zen 5), 128 cores each, 256 cores, no SMT, 8 NUMA nodes |
| RAM | 755 GB per node |
| Local disk | 1.7 TB NVMe per node; 854 GB quota for this job |
| Allocation | 128 cores (half a node), 345.6 GB RAM (2.7 GB/core default) |
| Software | Apptainer image `openfold3-msa_v0.5.0.sif` (amd64), Snakemake 9.21.0, HMMER 3.4; openfold-3 commit `127f7a67` |
| Databases | uniref90 (90.6 GB), uniprot (120.1 GB), mgnify (138.3 GB), copied to local NVMe before the alignments; MSAs were written to project storage |
| Parallelism | 8 jackhmmer threads per alignment, 16 alignments at a time (128 cores); 16 slots were busy from start to minute ~104, then the last alignments drained by minute ~116 |

### Runtimes

| Phase | Wall time |
|---|---|
| Copy databases to local NVMe (349 GB, ~2.3 GB/s) | 150 s: uniref90 32 s, uniprot 52 s, mgnify 66 s |
| 288 alignments (96 proteins × 3 databases) | 6,954 s (1 h 56 min) |
| **Whole job** | **7,113 s (1 h 58 min 33 s)** |

Per alignment, measured with 8 threads while 15 other alignments ran on the same node:

| Database | Mean | Median | Min | Max | CPU time / wall time |
|---|---|---|---|---|---|
| uniref90 | 269 s | 228 s | 180 s | 884 s | 5.9 of 8 |
| uniprot | 362 s | 312 s | 229 s | 1,065 s | 5.7 of 8 |
| mgnify | 438 s | 418 s | 308 s | 1,051 s | 5.5 of 8 |

Wall time grows with sequence length: Pearson r between length and time is 0.56 for uniref90, 0.55 for uniprot and
0.42 for mgnify. Summed over the three databases, the slowest protein (1566 residues) took 2,935 s.

Per protein, summing the three searches:
- Wall time: mean 1,069 s, median 959 s, range 757–2,935 s; about 2.4 s per residue.
- CPU time: mean 6,051 s (1.7 core-h).

Memory:
- Most alignments are small: median peak memory per alignment is 0.3–0.5 GB.
- The largest were 39 GB (uniprot) and 24 GB (uniref90), both for long proteins.
- Peak memory over the whole job was 44.6 GB (`seff`).
- The job's memory counter sat at ~335 GB of the 345.6 GB limit for the whole run. That is page cache from the copied
  databases, which don't fit in the limit, so files were evicted and read again from NVMe later in the run.

**Average total time per sequence:**
- **1,069 s (17.8 min)** of wall time for one protein's three searches (8 threads per search, 16 searches at a time on
  the node).
- In throughput terms, the whole job took **74 s of half-node time per sequence (7,113 s / 96)**, which is
  **2.63 core-h per sequence**.

### Slurm allocation usage

| | |
|---|---|
| Account | `naiss2026-3-524-cpu` |
| Charged | 128 cores × 1 h 58 min 33 s = **252.9 core-h** (billing = 128 per hour). This is 2.5 % of the monthly 10,000 core-h. |
| Actually used | 167.4 core-h of CPU time, so CPU efficiency was 66 % (`seff`) |
| Per sequence | 2.63 core-h charged, 1.74 core-h used |

### Notes

- **Threads are underused.** jackhmmer averaged only ~5.7 of its 8 threads. Running more alignments with fewer threads
  each (for example 32 × 4) should give better throughput on the same allocation. This is untested.
- **Copying the databases is cheap.** It took 2 % of the job's time. Whether it is faster than reading the databases
  straight from project storage was not measured.
- **Extrapolation to the full dataset.** huri-v1 has 17,421 proteins. At 2.63 core-h per protein that is
  **~46k core-h, about 4.6 months of the CPU allocation**. This assumes similar lengths and these settings.
- **Output size.** The MSAs took 14 GB, about 146 MB per protein, which would be ~2.5 TB for all of huri-v1.
