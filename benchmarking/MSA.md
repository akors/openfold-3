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

### Thread layout test: 32 × 4

Job 3248592, 2026-10-01, node `n368` (same hardware). Raw records are in
`data/benchmarks/msa-huri-v1-mixed50train-32x4.arrhenius/`, together with the job script.

The run used **4 threads per alignment, 32 alignments at a time**, on the same 128 cores, with databases copied to local
disk as before. The input was 56 proteins, the `mixed50/train` split, with mean length 459. All 56 were in the run
above, so the 168 alignments compare one to one.

| | 16 × 8 (job 3237961) | 32 × 4 (job 3248592) |
|---|---|---|
| Proteins | 96 | 56 |
| Copy databases | 150 s | 142 s |
| Alignments | 6,954 s | 3,896 s |
| Whole job | 7,113 s | 4,047 s (1 h 07 min) |
| Charged | 252.9 core-h | 143.9 core-h |
| CPU efficiency (`seff`) | 66 % | 75 % |
| Charged per sequence | 2.63 core-h | 2.57 core-h |
| Half-node time per sequence | 74 s | 72 s |
| Mean wall time per sequence (3 searches) | 1,069 s | 1,779 s |
| Peak memory (`seff`) | 44.6 GB | 44.6 GB |

Per alignment, for the same 56 proteins:
- **Threads are almost fully used.** jackhmmer kept 3.7–3.8 of its 4 threads busy, against 5.5–6.0 of 8.
- **Each search is slower but cheaper.** With half the threads, a search takes 1.67× as long. It occupies 17 % fewer
  core-seconds, though: 7,116 per protein against 8,529. Total CPU time is 8–10 % higher, probably because 32 searches
  compete for memory bandwidth and cache.

**Whole job.** The node was busy while all 32 slots were full: 121 of 128 cores on average for the first ~51 minutes of
alignments. In the last ~14 minutes only 23 cores were busy on average, because a few long searches ran on alone at 4
threads each. The longest search took 2,247 s (uniprot), and the slowest protein took 5,703 s over its three searches.
With only 56 proteins, that tail and the database copy take up a large share of the job. As a result the measured gain
per sequence is only 2 %.

**In production,** with thousands of proteins per job, the tail and the database copy shrink to almost nothing. The
cost would then approach the core-seconds per protein, giving **an estimated ~2.0 core-h per sequence with 32 × 4,
against ~2.4 with 16 × 8 (about −17 %)**. This is an estimate and hasn't been measured.

**Average total time per sequence (32 × 4):**
- **1,779 s (29.7 min)** of wall time for one protein's three searches.
- In throughput terms, **72 s of half-node time per sequence (4,047 s / 56)**, which is **2.57 core-h**.

### Notes

- **Use fewer threads per alignment.** At 8 threads, jackhmmer kept only ~5.7 busy. At 4 threads it keeps ~3.8 busy and
  needs ~17 % fewer core-hours per protein; see the thread layout test. Two threads per alignment (64 × 2) is untested.
- **Copying the databases is cheap.** It took 2 % of the job's time. Whether it is faster than reading the databases
  straight from project storage was not measured.
- **Extrapolation to the full dataset.** huri-v1 has 17,421 proteins. At 2.63 core-h per protein that is
  **~46k core-h, about 4.6 months of the CPU allocation**. With 32 × 4 in large jobs, the estimate drops to ~35k core-h
  (~2.0 core-h per protein). Both assume similar protein lengths.
- **Output size.** The MSAs took 14 GB, about 146 MB per protein, which would be ~2.5 TB for all of huri-v1.

## Berzelius

Job 17684806, 2026-10-01. Raw records are in `data/benchmarks/msa-huri-v1-mixed50.berzelius/` (not in git):
- `msa-huri-v1-mixed50.sbatch`: the job script
- `benchmarks/`: one record per alignment
- `run-17684806/`: metadata, phase times (database copies only, see below), and cgroup usage sampled every 30 s
- `slurm-17684806.out`: the job's Slurm log

**258 of the 288 alignments finished.** The repository was updated (`git pull`) at 20:04 while the job ran, which
replaced `MSA_Snakefile_benchmark`. Snakemake reads the Snakefile again for every alignment it starts, so the 30
alignments started in the next 28 s failed at once with "Snakefile not found" (18 mgnify, 7 uniref90, 5 uniprot).
Alignments started later ran normally. The job ended as FAILED after the rest were done, so `phases.tsv` has no row for
the alignments. The per-alignment numbers below come from the 258 records; the whole-job time for all 288 is estimated
from them.

### Hardware and setup

| | |
|---|---|
| Node | `node226`, `berzelius-cpu` partition, x86_64 |
| CPU | 2× AMD EPYC 9534 (Zen 4), 64 cores each, 128 cores, SMT 2 (256 threads), 4 NUMA nodes |
| RAM | 1.1 TB per node |
| Local disk | 6.4 TB NVMe per node (`/scratch/local`), no per-job quota |
| Allocation | 128 cores (the whole node), 994 GB RAM (7.76 GB/core default) |
| Software | Apptainer image `openfold3-msa_v0.5.0-amd64.sif`, Snakemake 9.21.0, HMMER 3.4; openfold-3 commit `127f7a67` |
| Databases | uniref90 (90.6 GB), uniprot (120.1 GB), mgnify (138.3 GB), copied to local NVMe before the alignments; MSAs were written to project storage |
| Parallelism | 8 jackhmmer threads per alignment, 16 alignments at a time (128 cores); 16 slots were busy from 18:11 to ~20:04 (job minutes 5–118), then the remaining alignments drained by 20:31 (minute ~145) |

### Runtimes

| Phase | Wall time |
|---|---|
| Copy databases to local NVMe (349 GB, ~1.2 GB/s) | 284 s: uniref90 71 s, uniprot 97 s, mgnify 116 s |
| 258 of 288 alignments | 8,390 s (2 h 20 min), from the Snakemake log |
| Whole job, as run | 8,710 s (2 h 25 min 10 s) |
| **Whole job, estimated for all 288** | **~9,500 s (~2 h 38 min)** |

The estimate adds the 30 missing alignments at the per-database mean times below, spread over the 16 slots (~1,000 s).

Per alignment, measured with 8 threads while 15 other alignments ran on the same node:

| Database | n | Mean | Median | Min | Max | CPU time / wall time |
|---|---|---|---|---|---|---|
| uniref90 | 89 | 366 s | 321 s | 266 s | 1,114 s | 4.5 of 8 |
| uniprot | 91 | 480 s | 427 s | 361 s | 1,382 s | 4.2 of 8 |
| mgnify | 78 | 601 s | 592 s | 501 s | 996 s | 3.9 of 8 |

Wall time grows with sequence length: Pearson r between length and time is 0.54 for uniref90, 0.54 for uniprot and
0.55 for mgnify. Summed over the three databases, the slowest complete protein (1141 residues) took 3,111 s.

Per protein, summing the three searches (the 71 proteins with all three records):
- Wall time: mean 1,466 s, median 1,337 s, range 1,152–3,111 s; about 3.4 s per residue.
- CPU time: mean 5,999 s (1.67 core-h).

Memory:
- Most alignments are small: median peak memory per alignment is 0.3–0.5 GB.
- The largest were 38 GB (uniprot) and 24 GB (uniref90), both for long proteins.
- Peak memory over the whole job was 47.8 GB (`seff`).
- The job's memory counter peaked at ~720 GB of the 994 GB limit. That is page cache from the copied databases
  (349 GB), which fit in the limit.

CPU use: while all 16 slots were busy (end of the database copy to 20:03), the job used on average 66 of its 128 cores.
Over the whole job, CPU efficiency was 47 % (`seff`).

**Average total time per sequence:**
- **1,447 s (24.1 min)** of wall time for one protein's three searches: the sum of the three per-database means (8
  threads per search, 16 searches at a time on the node). The mean over the 71 complete proteins is 1,466 s.
- In throughput terms, the estimated whole job takes **~99 s of full-node time per sequence (~9,500 s / 96)**, which
  is **~3.5 core-h per sequence**.

### Slurm allocation usage

| | |
|---|---|
| Account | `berzelius-2026-136` |
| Charged | 128 cores × 2 h 25 min 10 s = **309.7 core-h = 19.4 GPU-h** (16 cores = 1 GPU-h; billing = 128 per hour). This is 0.11 % of the group's monthly 18,000 GPU-h. |
| Estimated for all 288 | ~338 core-h = ~21 GPU-h |
| Actually used | 146 core-h of CPU time (`seff`: 6-01:54:19), so CPU efficiency was 47 % |
| Per sequence | ~3.5 core-h (~0.22 GPU-h) charged for all 288, 1.67 core-h used |

### Notes

- **Threads are underused.** jackhmmer averaged only ~4.2 of its 8 threads.
- **Copying the databases** took ~3 % of the job's time. Reading the databases straight from project storage was not
  measured.
- **Extrapolation to the full dataset.** huri-v1 has 17,421 proteins. At ~3.5 core-h per protein that is
  **~61k core-h, or ~3,800 GPU-h, about 21 % of the group's monthly allocation**. This assumes similar lengths and these
  settings.
- **Output size.** The 258 MSAs took 12 GB, about 140 MB per protein for all three, which would be ~2.4 TB for all of
  huri-v1.
