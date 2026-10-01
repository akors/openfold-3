# TODO

See [README.md](README.md) for the plan behind these tasks.

## High level

- [ ] **Dataset:** HuRI protein pairs with a train/validation/test split and a negative set
  - [x] Map HuRI genes to proteins and fetch sequences (GENCODE v27)
  - [x] Cluster HuRI proteins and search them against the PDB with MMseqs2
  - [x] Decide the protein-level train/validation/test split (see README)
  - [x] Split the proteins: test from hit-free clusters, then train/validation from the pooled rest
  - [x] Build dimers (positives and negatives): `dataset_buildmultimers.py`, pure and mixed modes
  - [x] Prepare MSA jobs for OpenFold3's Snakemake workflow: `prepjob_msa.py`
  - [ ] Compute MSAs (into `../data/datasets/huri-v1/msas/`)
  - [ ] Prepare OpenFold3 input files: preparsed MSAs and query JSONs
- [ ] **OpenFold3 setup:** inference with the default weights works and the outputs we need can be extracted
- [ ] **Benchmarking:** OpenFold3 speed and memory on Intel + Hopper and Grace Hopper nodes
- [ ] **Predictions:** run OpenFold3 on the full set of dimers and fill the feature cache
- [ ] **Baselines:** chain-pair ipTM, ipSAE, and logistic regression over OpenFold3's scalar scores
- [ ] **Tier 0:** learned pooling over PAE/PDE/distogram logits
- [ ] **Tier 1:** head on the confidence module's pair representation
- [ ] **Tier 2** (only if Tier 1 plateaus): fine-tune a copy of the confidence Pairformer
- [ ] **Write-up**

## Next tasks

1. [x] **Decide how to build dimers.** Done in `dataset_buildmultimers.py`:
   - pure (C3) and mixed (C2) modes;
   - uniform negatives, 3 per positive;
   - a 2048-residue cap per multimer for now;
   - genes with more than one protein, and homodimers, left out.

   Follow-ups:
   - [ ] Check the sizes of the pure val/test sets: pure test has 1,762 positives and pure val only 403, although
     both splits hold 1,743 proteins.
   - [ ] Compare the length distributions of positives and negatives (length as a shortcut).
   - [ ] Add a degree-only baseline; consider degree-balanced negatives (see README, open questions).
   - [ ] Decide whether to also exclude `HI-union` interactions from the negatives.
   - [ ] Set the final length cap once benchmarking shows what fits.
   - [x] Build `huri-v1` (`huri-v0` is for development), so far with the 50-multimer sets `pure50` and `mixed50`.
2. [ ] **Compute MSAs** for `huri-v1`'s `mixed50` with `prepjob_msa.py` and the Snakemake workflow, into
   `../data/datasets/huri-v1/msas/`.
3. [ ] **Prepare the OpenFold3 input files** so that inference does no avoidable work:
   - Preparse each protein's MSAs into one NPZ file (`scripts/data_preprocessing/preparse_alignments_of3.py`), with
     `max_seq_counts` equal to the inference defaults (`uniprot_hits` 50000, which pairing reads); the raw `.sto`
     parser is pure Python and would otherwise run again for every pair a protein is in.
   - Check that every protein has an NPZ file, that no MSA is empty, and that the first MSA row matches the sequence.
   - One query per pair, named canonically, with chains A and B and paths relative to the repository root; labels stay
     in the multimer TSVs. Templates off; paired MSAs built online from `uniprot_hits`.
   - Write the queries in shards, bucketed by total length, so a failed job costs one shard and each length bucket
     can get its own runner settings.
4. [ ] **Capture the latent features and decide how to store them**, using a tiny set of predictions (fewer than 10
   dimers from `mixed50`):
   - Make the code changes: a hook on `aux_heads.pairformer_embedding` versus a small opt-in change to OpenFold3.
   - Run the tiny set to confirm that the default `openbind-2025-06-30-174k` weights load and run, that
     `write_latent_outputs` and `write_features` produce the keys and shapes we expect, that the captured tensors
     are there, and that paired MSAs are generated for heterodimers.
   - From the latents produced, decide on compression and pruning: inter-chain blocks only, one diffusion sample,
     bf16, contact-probability pruning or PCA; measure the real size per pair.
5. [ ] **Benchmark inference on `mixed50`** to measure the time and cost of inference on our hardware and find good
   settings.
   - Wall-clock time and peak GPU memory against token count, MSA time versus inference time
   - Both node types: Intel + Hopper and Grace Hopper
   - CPU time for featurization and online MSA pairing, and whether the data loader keeps the GPU busy
   - Settings to tune: number of diffusion samples, number of seeds, chunk size and memory options
