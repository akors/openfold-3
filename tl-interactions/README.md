# tl-interactions: transfer learning from AlphaFold3 to protein–protein interactions

This is a living document. It describes the plan and gets updated as the project moves along.

## Goal

The main goal is to **demonstrate the transfer-learning capabilities of AlphaFold3**. We train a small, specialized model on
top of a frozen AlphaFold3 (OpenFold3) to predict whether two human proteins bind each other. The training data is a
large-scale experimental interaction screen, and the question is whether AF3's internal representations carry more
information about binding than its built-in confidence scores (ipTM) do.

Additional goals:

- **Benchmarking** OpenFold3 inference speed on Intel + Hopper nodes and on Grace Hopper nodes.
- **Utilizing intermediate activations** of AlphaFold (pair and single representations) as features.
- **Using AF3 as a "foundation model"** from which specialized models can be derived, without retraining the large model.
- **Learning to use Claude Code and other AI models effectively** for research software work.

## Approach

```
HuRI pairs ─► sequences + MSAs ─► OpenFold3 inference (frozen, no ground truth) ─► feature cache ─► small head ─► P(bind)
```

1. Take protein pairs (dimers) from the human reference interactome, together with a set of negative pairs.
2. Predict each dimer with OpenFold3 using the default `openbind-2025-06-30-174k` weights. Only inference is needed,
   so no experimental structures are required.
3. Cache the features the new model reads, trimmed to the inter-chain part of each pair (see [Feature cache](#feature-cache)).
4. Train a separate small network, built from parts of the AF3 confidence module, with binary cross-entropy against the
   interaction label. The rest of OpenFold3 stays frozen.

### Why this can work without ground-truth structures

In OpenFold3 the ground-truth structures only serve to build targets for the existing losses (diffusion, distogram,
pLDDT/PAE/PDE) and for chain-permutation alignment. With everything except the new head frozen, none of those losses
run, and the new head's only target is the binary interaction label. Training is therefore a standalone PyTorch loop
over cached tensors rather than OpenFold3's training pipeline.

Caching in inference mode is also the right choice for consistency: in training mode the confidence module sees short
20-step "mini-rollouts" and randomly drops the trunk pair embedding, while at inference it sees full rollouts. Training
on inference-mode features matches what the head sees when it is deployed.

## Dataset: HuRI

The [Human Reference Interactome (HuRI)](http://www.interactome-atlas.org/) (Luck et al., *Nature* 2020) is a map of
binary protein–protein interactions from systematic yeast two-hybrid (Y2H) screening of about 17,500 human
protein-coding genes. It contains roughly 53,000 interactions between about 8,300 proteins.

Properties that shape the project:

- **Binary, direct interactions.** Y2H detects direct pairwise binding, which suits a model that looks at a dimer
  interface. It also picks up weak and transient interactions, which ipTM tends to under-score.
- **No true negatives.** Y2H misses many real interactions, so "not detected" does not mean "does not bind." Negatives are
  sampled from the screened search space; the sampling strategy is still open.
- **Hub proteins.** Some proteins have many partners. A split over pairs would let a model learn "protein X binds a lot"
  instead of learning about interfaces, so the splitting strategy has to account for this.

### Data preparation

The dataset is built in steps, each by a script in this directory. Data lives in `../data/`, and each dataset in its
own directory under `../data/datasets/`. [huri-exploration.ipynb](huri-exploration.ipynb) explores the HuRI tables
the first step starts from.

**1. Proteins and their split.** [dataset_prepseq.py](dataset_prepseq.py) takes the proteins of the HuRI search space
from the supplementary tables, fetches their sequences from GENCODE (release 27, matching HuRI) and splits them as
described [below](#train-validation-and-test-split), using MMseqs2 for the PDB search and the clustering. It writes
`sequences.zip`, `seqs-train.txt`, `seqs-val.txt`, `seqs-test.txt`, `meta.ini`, and for the next step
`proteins.tsv` (protein to gene) and `interactions.tsv` (a copy of `HuRI.tsv`). The slow steps are cached, and
existing outputs are never overwritten. It needs `mmseqs` on the `PATH`. From the repository root:

```bash
H=data/interactomes/HuRI
pixi run -e tl-interactions-analysis tl-interactions/dataset_prepseq.py \
    "$H/Supplementary Table 1.txt" "$H/Supplementary Table 2.txt" $H/HuRI.tsv data/datasets/huri-v2 \
    --pdb-db data/mmseqs-db/PDB
```

**2. Multimers.** [dataset_buildmultimers.py](dataset_buildmultimers.py) builds labelled multimers (protein pairs)
for each split. It writes them to `multimers/<name>/multimers-{train,val,test}.tsv` (columns `protein_a`,
`protein_b`, `label`) with a `meta.ini`. Genes with more than one protein (13, whose screened ORF is ambiguous) and
homodimers are left out. Positives are the HuRI interactions. Negatives are drawn uniformly from the allowed pairs that
are not HuRI interactions, `--negatives-ratio` (default 3) per positive. Multimers whose summed length exceeds
`--length-cutoff` (default 2048, OpenFold3's validation-set token cap) are dropped. `--mode` decides which proteins
make up the validation and test multimers:

- **pure:** both proteins from the split (C3 in Park & Marcotte, 2012);
- **mixed:** one protein from train and the other from the split (C2). Train multimers are the same in both modes.

For smaller sets, `--splits` builds only some splits and `--total N` keeps at most `N` multimers over them. All splits
are scaled by the same fraction, with at least one positive each.

```bash
pixi run -e tl-interactions-analysis tl-interactions/dataset_buildmultimers.py data/datasets/huri-v2 --mode pure
```

**3. OpenFold3 input files.** TODO: one query per pair.

**4. MSAs.** TODO: precomputed alignments for the proteins.

Still to decide:

- the negative set, sampled uniformly for now (see [Open questions](#open-questions-and-risks));
- the final length cap per pair, once benchmarking shows what fits.

### Train, validation and test split

Proteins (not yet pairs) are split by sequence cluster, so no cluster spans two sets. The *validation* set is for
model selection and gets looked at repeatedly; the *test* set is looked at once, for the final numbers.

1. Search all HuRI proteins against the PDB with MMseqs2 (≥25% identity, ≥50% coverage of the PDB chain).
2. Cluster all HuRI proteins with MMseqs2: single-step connected-component clustering at ≥30% identity and ≥80%
   coverage of both sequences, so that no hit above these thresholds links two clusters.
3. **Test:** a random subset of the clusters in which no member has a PDB hit.
4. **Train and validation:** the remaining clusters, with and without PDB hits, pooled and assigned at random. Keeping
   hit-free clusters in training means the model also learns from the kind of protein it is tested on.

Assumptions:

- "No PDB hit" means no hit at the thresholds of step 1. More remote structural homologs may still exist.
- OpenBind-0 has not seen structures of the test proteins. It was trained on the PDB through June 2025; we assume its
  distillation sets are OpenFold3's (MGnify monomers, which are metagenomic rather than human, and disordered regions
  of PDB entries), which the test proteins are not part of.

## Model

**Base model:** OpenFold3 with the default `openbind-2025-06-30-174k` weights (`of3-ob-2025-06-30-174k.pt`), frozen.
We run the full model in inference mode: trunk (MSA module and Pairformer), diffusion sampling, and the confidence
module.

**New head:** a separate small network built from parts of the AF3 confidence module (AF3 SI §4.3, Algorithm 31). It
scores inter-chain token pairs and pools them into one binding probability per chain pair.

Pooling is framed as **multiple-instance learning**. The label belongs to the chain pair (the "bag"), while the evidence
lives in a few interface token pairs (the "instances") that we cannot identify, because most HuRI pairs have no
experimental structure. ipTM averages over *all* tokens of the partner chain, so a confident interface on a large or
partly disordered protein gets diluted. The new head instead uses a pooling that concentrates on the most informative
pairs:

- **LogSumExp** (a soft maximum), or
- **attention pooling** (Ilse et al., 2018).

Both pooling directions (A→B and B→A) are used.

## Tiers

The project is staged so that every tier produces a result on its own and tells us whether the next one is worth it.
The target is **Tier 1 at least**.

### Tier 0: learned pooling over existing confidence outputs

- **Features:** the binned PAE (both directions), PDE, and distogram logits for each inter-chain token pair. OpenFold3
  already writes these with `write_latent_outputs: true`, so **no changes to OpenFold3 are needed**.
- **Model:** a small MLP per token pair, followed by multiple-instance pooling. In effect this is a "learned ipTM."
- **Question answered:** does learned pooling over the existing confidence distributions beat ipTM?

### Tier 1: head on the confidence module's pair representation

- **Features:** the pair representation `z_ij` *after* the confidence module's 4-block Pairformer. It has seen the
  predicted structure and is tuned to judge interface quality.
- **Capture:** OpenFold3 does not save this tensor (it is local to `AuxiliaryHeadsAllAtom.forward` in
  `openfold3/core/model/heads/head_modules.py`, and is freed as soon as the PAE and PDE heads have read it). The plan
  is a PyTorch forward hook on `aux_heads.pairformer_embedding`, registered from a small wrapper script, so
  OpenFold3's source stays unchanged. The module returns `(si, zij)`, so the same hook also yields the confidence
  module's single representation at no extra cost. The fallback is a small, opt-in change that adds the tensor to the
  model outputs.
- **Model:** a linear projection per token pair (a linear probe), followed by multiple-instance pooling. It has few
  parameters, which suits noisy binary labels.
- **Question answered:** do the confidence module's internal representations carry binding information beyond its
  PAE/PDE outputs?

### Tier 2: fine-tuning a copy of the confidence Pairformer

- **Features:** the *inputs* to the confidence Pairformer: `si_input`, `si_trunk`, `zij_trunk`, and the predicted
  representative-atom coordinates. Most of these are already written out: `si_trunk`, `zij_trunk` and
  `atom_positions_predicted` go into the output dictionary that `write_latent_outputs` dumps (`OpenFold3._rollout` in
  `openfold3/projects/of3_all_atom/model.py`). Only `si_input` needs capturing, and since it is an *argument* to
  `pairformer_embedding` rather than a return value, that takes a forward **pre**-hook (or a forward hook registered
  with `with_kwargs=True`) rather than the plain forward hook Tier 1 uses.
- **Model:** a copy of the confidence module's Pairformer, initialized from the checkpoint and fine-tuned (low learning
  rate or LoRA), followed by the Tier 1 head. A randomly initialized Pairformer would very likely overfit.
- **Cost:** the triangle updates need the full N×N pair matrix, including the within-chain blocks, so the cache cannot be
  trimmed to the inter-chain block.
- **Only if** Tier 1 plateaus.

## Feature cache

The frozen model runs once per pair, and head training runs on cached tensors. Storage is the main constraint. Rough
sizes for a typical pair of two 400-residue proteins (800 tokens):

| Cache | Size per pair |
|---|---|
| Raw `*_latent_output.pt`, 5 diffusion samples, fp32 | ~2 GB |
| Tier 0: one sample, inter-chain blocks only, bf16 | ~80 MB |
| Tier 1: confidence `z`, inter-chain blocks in both directions, bf16 | ~80 MB |
| Tier 2: full-matrix inputs, bf16 | ~200 MB |

At 10,000 pairs, even the trimmed caches reach about 1 TB. The rules we follow:

- trim each output right after inference and delete the raw dump;
- keep one diffusion sample (the top-ranked one) to start with; the trunk representation and the distogram do not
  depend on the sample;
- store only the inter-chain blocks for Tiers 0 and 1;
- if still too large, keep only token pairs with non-negligible contact probability, or reduce `z` from 128 to about 32
  dimensions with PCA.

## Evaluation

**Baselines** (in order of cost):

1. Chain-pair ipTM straight from OpenFold3 (no training).
2. ipSAE (Dunbrack, 2025), which fixes ipTM's dilution by flexible or disordered regions.
3. Logistic regression over scalars OpenFold3 already produces: chain-pair ipTM, the lowest inter-chain PAE, mean
   interface pLDDT, and total inter-chain contact probability.

**Metrics:**

- AUROC, together with **AUPRC at a realistic prior**. True interactions are rare among all human protein pairs, and a
  classifier that looks excellent on a balanced test set can have very low precision at realistic prevalence. We report
  AUPRC next to its chance baseline (the prevalence).

## Benchmarking

We measure OpenFold3 inference on two node types:

- Intel CPUs with NVIDIA Hopper GPUs
- NVIDIA Grace Hopper nodes

Planned measurements: wall-clock time and peak GPU memory as a function of token count, and the split between MSA
generation and model inference. Paired MSAs are likely to be the biggest compute cost of the project, so they get their
own numbers.

## Open questions and risks

- **No true negatives.** Y2H misses many real interactions, so "not detected" does not mean "does not bind". We
  assume it does anyway: that is the data we have, and there is no ground truth to replace it. The labels are
  therefore noisy by construction. How sensitive the screens are can be estimated from the controls in
  Supplementary Table 5, where each version of the assay was run as a pairwise test on a positive reference set
  (PRSv1, 84 well-documented binary interactions) and a random reference set (RRSv1, 90 random pairs), in both
  orientations. A pair counts as detected if either orientation scored positive, and tests that are invalid or that
  auto-activated are left out. The numbers are our own count:

  | Assay version | PRSv1 detected | RRSv1 detected |
  |---|---|---|
  | 1 | 18 of 79 (23%) | 1 of 84 (1.2%) |
  | 2 | 17 of 80 (21%) | 0 of 90 |
  | 3 | 23 of 84 (27%) | 0 of 90 |
  | Any of the three | 31 of 84 (37%) | 1 of 90 (1.1%) |

  A retested pairwise test is the best case. The screens only sample the search space, so their sensitivity is
  lower, and HuRI probably detects no more than about a third of the true binary interactions in its search space.
  Our uniform negatives are nevertheless mostly correct, because true interactions are rare among random pairs.
  Assuming that 37% sensitivity and ignoring false positives, HuRI's 52,548 interactions stand for at least
  ~140,000 true ones. That leaves ≥ ~90,000 undetected among the ~1.5 × 10⁸ gene pairs of the search space, so about
  0.06% of random negatives. This is a rough lower bound, and it is not evenly spread: missed interactions pile up
  on proteins that Y2H handles badly.
- **The protein-degree shortcut.** Some proteins have far more HuRI partners than others, and uniformly sampled
  negatives rarely contain them. Degree and abundance alone therefore separate positives from negatives, so a high
  AUROC could reflect "protein X has many partners" rather than anything about interfaces. The protein-level split
  limits this but does not remove it. Park & Marcotte (2012) is the reference for evaluating pair-input predictions
  under exactly this failure mode. Remedies still to weigh: a degree-only baseline next to the model, and
  degree-balanced negatives.
- **A PDB-derived control set.** A second dataset built from PDB complexes, with positives taken from chains that are
  in contact and artificial negatives from chain pairs that are not. Its positives have experimental structures and
  its negatives are cleaner than anything HuRI can offer, which makes it a positive control: if the head cannot
  separate that set, the HuRI numbers say nothing. Open: how to sample the negatives, and whether it serves as a
  control only or as a training set in its own right.
- **Missing information.** Interactions that depend on disorder, short linear motifs, post-translational modifications,
  or a third partner may leave little signal in the structure model's representations.
- **Paired MSAs** for heterodimers matter for prediction quality and dominate compute.

## AI usage disclosure

This project uses AI assistants deliberately, and learning to use them well is one of its goals. This section records
how they are used, so readers can judge which parts to trust and how far.

**Tools.** Claude Code (Anthropic), running Claude Opus 5.5, with access to this repository. Other assistants or models
get added here as they come into use.

**What AI has contributed so far:**

- Explanations of the AF3 confidence module, its losses and its scalar scores (pTM, ipTM, gPDE, sample ranking), taken
  from the AF3 supplementary information and cross-checked against the OpenFold3 source.
- The model design discussed in this README: multiple-instance pooling, the tiered plan, the feature-cache strategy, and
  the evaluation pitfalls.
- A first draft of this README, which was then reviewed and edited by the project authors.

**What stays with the authors.** The research question, the dataset, the choice of weights, the data-preparation
decisions and every design decision are made by the human authors. The authors are responsible for the results and for
the correctness of everything in this repository.

**How AI output is handled:**

- Code written with AI assistance is reviewed before it is committed. Such commits carry a `Co-Authored-By` trailer
  naming the assistant and model (see `AGENTS.md`).
- Statements about AF3 or OpenFold3 are checked against the supplementary information or the source code.
- Literature references and dataset statistics suggested by an AI are checked against the original publication before
  they are relied on. At the time of writing, the HuRI counts and the ipSAE reference in this README still need that
  check.
- Numbers marked as estimates (cache sizes, for example) are back-of-the-envelope calculations, not measurements.

**Data.** AI tools see only this repository and publicly available data: HuRI, GENCODE, the PDB and published model
weights.

## References

- Abramson et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature*.
  Supplementary information: `docs/41586_2024_7487_MOESM1_ESM.pdf`.
- Luck et al. (2020). A reference map of the human binary protein interactome. *Nature*.
- Ilse, Tomczak & Welling (2018). Attention-based deep multiple instance learning. *ICML*.
- Dunbrack (2025). Rēs ipSAE loquunt: What's wrong with AlphaFold's ipTM score and how to fix it. *bioRxiv*.
- Park & Marcotte (2012). Flaws in evaluation schemes for pair-input computational predictions. *Nature Methods*.
