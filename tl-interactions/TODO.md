# TODO

See [README.md](README.md) for the plan behind these tasks.

## High level

- [ ] **Dataset:** HuRI protein pairs with a train/validation/test split and a negative set
  - [x] Map HuRI genes to proteins and fetch sequences (GENCODE v27)
  - [x] Cluster HuRI proteins and search them against the PDB with MMseqs2
  - [x] Decide the protein-level train/validation/test split (see README)
  - [x] Split the proteins: test from hit-free clusters, then train/validation from the pooled rest
  - [ ] Build dimers (positives and negatives)
- [ ] **OpenFold3 setup:** inference with the default weights works and the outputs we need can be extracted
- [ ] **Benchmarking:** OpenFold3 speed and memory on Intel + Hopper and Grace Hopper nodes
- [ ] **Predictions:** run OpenFold3 on the full set of dimers and fill the feature cache
- [ ] **Baselines:** chain-pair ipTM, ipSAE, and logistic regression over OpenFold3's scalar scores
- [ ] **Tier 0:** learned pooling over PAE/PDE/distogram logits
- [ ] **Tier 1:** head on the confidence module's pair representation
- [ ] **Tier 2** (only if Tier 1 plateaus): fine-tune a copy of the confidence Pairformer
- [ ] **Write-up**

## Next tasks

1. [ ] **Decide how to build dimers.**
   - Pair-level split (the protein split is decided, see README)
   - Negatives: sampling strategy and positive-to-negative ratio
   - Length cap per pair
2. [ ] **Run a tiny set of predictions (fewer than 10 dimers)** to confirm that:
   - the default `openbind-2025-06-30-174k` weights load and run;
   - `write_latent_outputs` and `write_features` produce the keys and shapes we expect;
   - paired MSAs are generated for heterodimers.
3. [ ] **Work out how to extract the intermediate data without running out of storage.** Planning only for now.
   - Hook on `aux_heads.pairformer_embedding` versus a small opt-in change to OpenFold3
   - Keeping only inter-chain blocks, one diffusion sample, bf16; measure the real size per pair
4. [ ] **Run a small set of predictions (fewer than 100 dimers)** to benchmark OpenFold3 and find good settings.
   - Wall-clock time and peak GPU memory against token count, MSA time versus inference time
   - Both node types
   - Settings to tune: number of diffusion samples, number of seeds, chunk size and memory options
