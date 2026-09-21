# OpenFold3 Agent Codebase Map

Last structurally verified: 2026-09-21.

## Purpose and Use

This document is the starting point for agents and contributors working in the
OpenFold3 repository. Read it before searching the tree. It records the main
runtime paths, ownership boundaries, and high-value modification points so a
new task can begin with a targeted read instead of a repository-wide scan.

This is an orientation map, not an API guarantee. Before changing code:

1. Start from the symbol named here that most directly owns the behavior.
2. Read that symbol, its nearest caller, and a neighboring test or configuration.
3. Verify the current implementation; line numbers and details can drift.
4. Follow `AGENTS.md`: explain the smallest proposed change and obtain explicit
   permission before editing code.
5. Update this map when a task discovers a changed ownership boundary or a
   durable new path.

Prefer symbol searches over broad directory scans. Paths in this file are
repository-relative. A `Class.method` or `module.py:symbol` notation identifies
the intended search anchor.

## Quick Task Routing

| Task | Start here | Then inspect |
| --- | --- | --- |
| Change CLI behavior | `openfold3/run_openfold.py` | `openfold3/entry_points/experiment_runner.py`, `validator.py` |
| Change inference inputs | `InferenceDataset.create_all_features` | query primitives and the matching featurizer |
| Change training inputs/crops | `BaseOF3Dataset.create_all_features` | concrete dataset `__getitem__`, sample processor, featurizer |
| Add/change model architecture | `OpenFold3.__init__` and `OpenFold3.forward` | module under `projects/of3_all_atom/` or `core/model/` |
| Change training behavior | `OpenFold3AllAtom.training_step` | `ModelRunner`, optimizer/config hooks |
| Change the objective | `OpenFold3Loss.loss` | component under `core/loss/`, loss configuration |
| Change prediction files | `OF3OutputWriter.write_all_outputs` | inference callbacks and output settings |
| Change checkpoint loading | inference runner `setup`/`lightning_module` | `entry_points/checkpoint_utils.py`, EMA handling |
| Change config or presets | `OF3ProjectEntry` | project config, runner validators, example YAMLs |
| Add or locate tests | nearest `tests/` mirror of the touched package | `docs/BUILD_TESTING.md`, `pixi.toml`, `pyproject.toml` |

## Repository Ownership

- `openfold3/run_openfold.py`: Click CLI commands for training and prediction.
- `openfold3/setup_openfold.py`: installation/cache/checkpoint/CCD setup entry.
- `openfold3/entry_points/`: experiment orchestration, validated runner
  configuration, checkpoint/parameter resolution, and command-level utilities.
- `openfold3/core/config/`: shared configuration utilities and schemas.
- `openfold3/core/data/`: query/structure primitives, preprocessing,
  featurization, datasets, datamodules, and cache handling.
- `openfold3/core/model/`: reusable neural-network building blocks.
- `openfold3/core/kernels/`: accelerated kernel selection and implementations.
- `openfold3/core/loss/`: differentiable objective components and aggregation.
- `openfold3/core/metrics/`: evaluation and confidence-related metrics.
- `openfold3/core/runners/`: shared Lightning runner behavior, callbacks, EMA,
  prediction serialization, and runtime utilities.
- `openfold3/projects/of3_all_atom/`: the concrete all-atom project: model,
  runner, model config, presets, and project-specific modules.
- `examples/`: executable query JSONs, runner YAMLs, training YAMLs, and a full
  reference configuration. Use these before inventing configuration keys.
- `scripts/`: offline preprocessing, dataset, MSA, analysis, and developer tools;
  these are generally not part of the online model forward path.
- `tests/` and package-local test directories: behavioral checks. Locate the
  nearest test to the ownership point being changed.
- `docs/source/`: user-facing installation, inference, training, input, data
  pipeline, parameter, kernel, and configuration references.

Generated caches, downloaded checkpoints, CCD data, MSAs, templates, and model
outputs are external runtime artifacts. Do not infer their formats from example
filenames alone; follow the loader/writer and its validated configuration.

## End-to-End Flow

```text
openfold3/run_openfold.py
  -> entry_points/experiment_runner.py
     -> core/data/framework/data_module.py
        -> core/data/framework/single_datasets/*
           -> core/data/pipelines/sample_processing/*
           -> core/data/pipelines/featurization/*
     -> projects/of3_all_atom/runner.py:OpenFold3AllAtom
        -> projects/of3_all_atom/model.py:OpenFold3
        -> core/loss/loss_module.py:OpenFold3Loss
     -> core/runners/writer.py:OF3OutputWriter (prediction only)
```

PyTorch Lightning provides the outer training and prediction loops. The
project-specific Lightning module is `OpenFold3AllAtom`; its `.model` attribute
is the complete neural network, an `OpenFold3` instance.

## Executable Entry Points and Configuration

Installed console scripts are declared in `pyproject.toml`:

- `run_openfold` -> `openfold3.run_openfold:cli`.
- `setup_openfold` -> `openfold3.setup_openfold:main`.
- `validate-openfold3-rocm` -> the ROCm environment validator.

`run_openfold train` reads a runner YAML, applies explicit CLI seed overrides,
validates it as `TrainingExperimentConfig`, constructs
`TrainingExperimentRunner`, and calls `setup()` followed by `run()`.

`run_openfold predict` reads query JSON plus optional runner/CLI settings,
resolves a checkpoint, validates inference configuration, constructs
`InferenceExperimentRunner`, and runs Lightning prediction. The public examples
under `examples/example_inference_inputs/` and `examples/example_runner_yamls/`
are the best starting points for accepted inputs.

Keep these configuration layers distinct:

1. Pydantic experiment schemas in `openfold3/entry_points/validator.py` validate
  orchestration, trainer, data, checkpoint, logging, and output settings.
2. `OF3ProjectEntry` in `projects/of3_all_atom/project_entry.py` starts from the
  base model config, applies named presets from
  `config/model_setting_presets.yml`, then applies `model_update.custom`.
3. The resulting `ml_collections.ConfigDict` controls model architecture and
  model-coupled runtime behavior.
4. Explicit CLI options may override selected runner values. Inspect the Click
  command and `update_config_with_cli_args()` before assuming precedence.

Useful configuration references are
`examples/reference_full_config/full_config.yml`,
`docs/source/configuration_reference.md`, and the typed project configs under
`openfold3/projects/of3_all_atom/config/`.

## Input Preprocessing

There is not one shared top-level preprocessing function for every mode. The
pipeline is assembled by the dataset classes, with separate training and
inference entry points that call common sample-processing and featurization
modules.

### Training

The central coordinator is
`BaseOF3Dataset.create_all_features()` in
`openfold3/core/data/framework/single_datasets/base_of3.py`. In order, it
creates:

1. Cropped target/ground-truth structure and reference-conformer features via
   `create_structure_features()`.
2. MSA features via `create_msa_features()` and `MsaSampleProcessorTrain`.
3. Template features via `create_template_features()`.
4. Per-example loss switches/weights via `create_loss_features()`.

Concrete dataset subclasses implement `__getitem__()` and call this shared
coordinator. `DataModule` in
`openfold3/core/data/framework/data_module.py` constructs datasets and
dataloaders for Lightning.

Validation uses `ValidationDataset.create_all_features()` in
`single_datasets/validation.py`. It delegates to the training coordinator and
then adds homology-similarity and antibody-antigen features used by validation
metrics.

The lower-level transformations live under:

- `openfold3/core/data/pipelines/sample_processing/`: structure cropping,
  MSA sampling, template processing, and conformer processing.
- `openfold3/core/data/pipelines/featurization/`: conversion of processed
  biological objects into model tensor dictionaries.
- `openfold3/core/data/primitives/`: query, structure, tokenization, and other
  core data representations.

### Inference

The central coordinator is `InferenceDataset.create_all_features()` in
`openfold3/core/data/framework/single_datasets/inference.py`. It:

1. Converts a `Query` into a tokenized `AtomArray` and reference molecules.
2. Creates structure and conformer features without ground truth.
3. Creates MSA features using `MsaSampleProcessorInference`.
4. Creates template features.
5. Adds inference-only pocket-sampling features.

`InferenceDataset.__getitem__()` adds query/seed metadata and a `valid_sample`
flag. `InferenceDataModule` in `core/data/framework/data_module.py` supplies
these batches to prediction.

### Best Modification Points

- Change training sample composition/cropping: `sample_processing/structure.py`
  and the concrete training dataset classes.
- Change model tensor definitions: the matching module in
  `pipelines/featurization/`.
- Change MSA selection/processing: `sample_processing/msa.py`.
- Add a feature in both modes: add it to both `BaseOF3Dataset.create_all_features()`
  and `InferenceDataset.create_all_features()`, unless it is deliberately
  training-only or inference-only.

## Complete Model Object

Yes. There are two useful object boundaries:

- `ExperimentRunner.lightning_module`, a cached property in
  `openfold3/entry_points/experiment_runner.py`, instantiates
  `OF3ProjectEntry.runner(...)`. For the all-atom project this is one
  `OpenFold3AllAtom` Lightning module containing the model, loss, EMA state,
  optimizer hooks, and training/prediction steps.
- `OpenFold3AllAtom.model` is the complete neural network. It is created by
  `ModelRunner.__init__()` in `openfold3/core/runners/model_runner.py` as
  `model_class(config)`, where `model_class` is `OpenFold3`.

`OpenFold3.__init__()` in `openfold3/projects/of3_all_atom/model.py` assembles
the input embedder, template embedder, MSA stack, Pairformer stack, diffusion
module/sampler, and auxiliary heads. Its `forward()` is the architecture-level
end-to-end model call.

The class/factory chain is:

```text
OF3ProjectEntry.runner
  = OpenFold3AllAtom
    -> ModelRunner(model_class=OpenFold3, config=...)
       -> self.model = OpenFold3(config)
```

Use the Lightning module when changing training/checkpoint/EMA behavior. Use
its `.model` when changing the network architecture or calling the neural
network directly.

## Model Call Contract

`OpenFold3.forward(batch)` returns `(batch, output)`, not just `output`.
Callers in the all-atom runner unpack both values. The returned batch can differ
from the input because the model adds a diffusion sample dimension and, during
training/validation, applies safe multi-chain permutation alignment to ground
truth in place.

The authoritative feature and output shape inventory is the docstring on
`OpenFold3.forward()` in `projects/of3_all_atom/model.py`. Important output
families include trunk single/pair representations, predicted atom positions,
confidence logits, distogram logits, and training-only diffusion values.

Training and inference do not consume exactly the same dictionary:

- Training/validation batches include a nested `ground_truth` dictionary and
  loss-control features.
- Inference batches carry non-tensor metadata such as `atom_array`, `query_id`,
  seed, and validity/repetition flags for prediction and serialization.
- `OpenFold3AllAtom.predict_step()` computes confidence scores after the model
  call and stores them in `outputs["confidence_scores"]` before returning the
  `(batch, outputs)` pair to the prediction writer.

When adding a feature, trace all four contracts: creation, collation/device
movement, `OpenFold3.forward()` consumption, and training loss or output writer
consumption. Do not assume every batch value is a tensor.

## Prediction Outputs

Prediction files are owned by `OF3OutputWriter` in
`openfold3/core/runners/writer.py`. The inference runner registers it in
`InferenceExperimentRunner.callbacks`, then Lightning calls
`on_predict_batch_end()` after each prediction batch.

`OF3OutputWriter.write_all_outputs()` creates:

- Predicted structures through `write_structure_prediction()`.
- Aggregated and full confidence files through `write_confidence_scores()`.
- Optional input feature dictionaries (`*_batch.pt`).
- Optional latent/raw model outputs (`*_latent_output.pt`).

The output layout is approximately:

```text
<output_dir>/<query_id>/seed_<seed>/
  <query_id>_seed_<seed>_sample_<n>_model.<pdb|cif>
  <query_id>_seed_<seed>_sample_<n>_confidences_aggregated.json
  <query_id>_seed_<seed>_sample_<n>_confidences.<json|npz>
```

The configured root comes from
`experiment_settings.output_dir` via `ExperimentRunner.output_dir`.

## Checkpoints, Parameters, and EMA

Parameter discovery/download and default cache paths are owned by
`openfold3/entry_points/parameters.py`; setup orchestration is in
`openfold3/setup_openfold.py`. User-facing behavior is documented in
`docs/source/Installation.md` and `docs/source/parameters_reference.md`.

Training checkpoints are Lightning checkpoints and include EMA state through
runner checkpoint hooks. Training resume/selective-loading behavior is
configured through the experiment validator and implemented by the training
experiment runner plus `openfold3/core/utils/checkpoint_loading_utils.py`.

Inference loading is deliberately stricter:

1. `InferenceExperimentRunner.setup()` loads the selected checkpoint.
2. `get_state_dict_from_checkpoint(..., init_from_ema_weights=True)` selects
  EMA-derived model weights.
3. `_load_state_dict_with_version_validation()` checks missing/unexpected keys
  and the model version tensor before loading the Lightning module.

Do not bypass version validation or silently use non-EMA weights when adapting
inference. Checkpoint compatibility is coupled to the model version and
parameter-set release; filenames alone are not sufficient evidence.

## Final Training Loss

The final differentiable scalar is assembled by `OpenFold3Loss.loss()` in
`openfold3/core/loss/loss_module.py`:

```text
total = confidence loss
      + diffusion loss (unless confidence-only training)
      + all-atom distogram loss (unless confidence-only training)
```

Each component applies its own configured weighting before it is added to
`cum_loss`. `OpenFold3Loss.forward()` returns the scalar and, when requested, a
detached component breakdown. Validation may use `loss_chunked()` to average
per-sample losses with lower memory use.

The live all-atom hook is `OpenFold3AllAtom.training_step()` in
`openfold3/projects/of3_all_atom/runner.py`. It selects the standard or manual
per-sample-gradient-clipping path; both call the model, then
`self.loss(batch, outputs, _return_breakdown=True)`, log the breakdown, and
return the scalar to Lightning for backpropagation. The base
`ModelRunner.training_step()` expresses the same shared pattern but is
overridden for this project.

To change the objective:

- Change aggregation or add a top-level term in `OpenFold3Loss.loss()`.
- Change the mathematics of an existing term in its module under
  `openfold3/core/loss/`.
- Change weights/settings in the model loss configuration, rooted at
  `model_config.architecture.loss_module`.
- Change per-example enablement/weighting in
  `pipelines/featurization/loss_weights.py` and the dataset loss settings.

## AF3 Supplement Algorithms

This section maps Algorithms 1-31 from the AlphaFold 3 supplementary
information (`docs/41586_2024_7487_MOESM1_ESM.pdf`) to OpenFold3. Evidence is:

- **Direct**: the source explicitly cites the AF3 algorithm number.
- **Correlated**: the symbol implements the named computation but does not cite
  that exact number.
- **Shared**: the pseudocode operation is composed from reusable helpers rather
  than represented by one dedicated class or function.

Do not confuse these with AlphaFold 2 or AlphaFold-Multimer algorithms carrying
the same number; source comments normally qualify those as `AF2` or
`AF2-Multimer`.

| Algorithm | Supplement name | OpenFold3 location | Evidence |
| --- | --- | --- | --- |
| 1 | Main Inference Loop | `projects/of3_all_atom/model.py:OpenFold3.forward` and `OpenFold3.run_trunk` | Direct |
| 2 | Input Feature Embedder | `core/model/feature_embedders/input_embedders.py:InputEmbedderAllAtom` | Direct |
| 3 | Relative Position Encoding | `core/utils/relpos.py:relpos_complex`, called by `InputEmbedderAllAtom` | Direct at caller |
| 4 | One-hot encoding with nearest bin | `core/utils/tensor_utils.py:binned_one_hot` | Correlated shared helper |
| 5 | Atom Attention Encoder | `core/model/layers/sequence_local_atom_attention.py:AtomAttentionEncoder`, `RefAtomFeatureEmbedder`, and `NoisyPositionEmbedder` | Direct |
| 6 | Atom Attention Decoder | `core/model/layers/sequence_local_atom_attention.py:AtomAttentionDecoder` | Direct |
| 7 | Atom Transformer | Block-form `DiffusionTransformer` and `CrossAttentionPairBias`, instantiated as `AtomAttentionEncoder.atom_transformer` and `AtomAttentionDecoder.atom_transformer`; block conversion is in `core/utils/atom_attention_block_utils.py` | Shared correlation |
| 8 | MSA Module | `core/model/feature_embedders/input_embedders.py:MSAModuleEmbedder` (lines 1-4) and `core/model/latent/msa_module.py:MSAModuleStack`/`MSAModuleBlock` (lines 5-15) | Direct |
| 9 | Outer Product Mean | `core/model/layers/outer_product_mean.py:OuterProductMean` | Direct |
| 10 | MSA Pair Weighted Averaging | `core/model/layers/msa.py:MSAPairWeightedAveraging` | Direct |
| 11 | Transition | `core/model/layers/transition.py:SwiGLUTransition` | Direct |
| 12 | Triangle Multiplication Outgoing | `core/model/layers/triangular_multiplicative_update.py:TriangleMultiplicationOutgoing` (plus fused variant) | Direct |
| 13 | Triangle Multiplication Incoming | `core/model/layers/triangular_multiplicative_update.py:TriangleMultiplicationIncoming` (plus fused variant) | Direct |
| 14 | Triangle Attention Starting Node | `core/model/layers/triangular_attention.py:TriangleAttentionStartingNode` | Direct |
| 15 | Triangle Attention Ending Node | `core/model/layers/triangular_attention.py:TriangleAttentionEndingNode` | Direct |
| 16 | Template Embedder | `core/model/feature_embedders/template_embedders.py:TemplatePairEmbedderAllAtom` and `core/model/latent/template_module.py:TemplateEmbedderAllAtom` | Direct |
| 17 | Pairformer Stack | `core/model/latent/pairformer.py:PairFormerStack` and `PairFormerBlock` | Direct |
| 18 | Sample Diffusion | `core/model/structure/diffusion_module.py:SampleDiffusion` | Direct |
| 19 | Centre Random Augmentation | `core/model/structure/augmentation.py:centre_random_augmentation` | Direct |
| 20 | Diffusion Module | `core/model/structure/diffusion_module.py:DiffusionModule` | Direct |
| 21 | Diffusion Conditioning | `core/model/layers/diffusion_conditioning.py:DiffusionConditioning` | Direct |
| 22 | Fourier Embedding | `core/model/feature_embedders/input_embedders.py:FourierEmbedding` | Direct |
| 23 | Diffusion Transformer | `core/model/layers/diffusion_transformer.py:DiffusionTransformer` and `DiffusionTransformerBlock` | Direct |
| 24 | Attention with Pair Bias | `core/model/layers/attention_pair_bias.py:AttentionPairBias`, `DiffusionAttentionPairBias`, and `CrossAttentionPairBias` | Direct |
| 25 | Conditioned Transition Block | `core/model/layers/transition.py:ConditionedTransitionBlock` | Direct |
| 26 | Adaptive LayerNorm | `core/model/primitives/normalization.py:AdaLN` | Direct |
| 27 | Smooth LDDT Loss | `core/loss/diffusion.py:smooth_lddt_loss` | Direct |
| 28 | Weighted Rigid Align | `core/loss/diffusion.py:weighted_rigid_align` | Direct |
| 29 | Express Coordinates in Frame | `core/loss/confidence.py:express_coords_in_frames` | Direct |
| 30 | Compute Alignment Error | `core/loss/confidence.py:compute_alignment_error` | Direct |
| 31 | Confidence Head | `core/model/heads/head_modules.py:AuxiliaryHeadsAllAtom`; projections are in `core/model/heads/prediction_heads.py:PairformerEmbedding`, `PredictedAlignedErrorHead`, `PredictedDistanceErrorHead`, `PerResidueLDDTAllAtom`, and `ExperimentallyResolvedHeadAllAtom` | Direct |

Algorithm 7 has no class literally named `AtomTransformer`: OpenFold3 expresses
its rectangular sequence-local attention by configuring the shared diffusion
transformer with query/key block sizes. Algorithm 4 is similarly a general
tensor helper. These are implementation organization differences, not evidence
that the operations are absent.

## AF3 Supplement Equations

This section maps the supplement's numbered Equations 1-20. A single code
symbol may implement several equations because target construction, binning,
and cross-entropy are combined in one loss function.

| Equation | Supplement quantity | OpenFold3 location | Evidence |
| --- | --- | --- | --- |
| 1 | Weighted PDB chain/interface sampling | `core/data/framework/single_datasets/pdb.py:DatapointCollection.create_datapoint_cache` (nested `calculate_datapoint_probability`) | Correlated; its docstring calls this “Algorithm 1” from SI 2.5.1 |
| 2 | Align ground truth to denoised coordinates | `core/loss/diffusion.py:mse_loss` calling `weighted_rigid_align` | Correlated; aligner is directly tagged Algorithm 28 |
| 3 | Weighted aligned MSE | `core/loss/diffusion.py:mse_loss` | Direct |
| 4 | DNA/RNA/ligand per-atom weights | Weight construction at the start of `core/loss/diffusion.py:mse_loss` | Correlated |
| 5 | Polymer-ligand bond loss | `core/loss/diffusion.py:bond_loss` and `bond_loss_sparse` | Direct |
| 6 | Combined diffusion loss | `core/loss/diffusion.py:diffusion_loss` | Direct |
| 7 | Inference noise schedule | `core/model/structure/diffusion_module.py:create_noise_schedule` | Correlated; source cites the SI page rather than Equation 7 |
| 8 | Per-atom LDDT confidence target | Target construction inside `core/loss/confidence.py:all_atom_plddt_loss` | Correlated |
| 9 | pLDDT cross-entropy loss | `core/loss/confidence.py:all_atom_plddt_loss` | Correlated |
| 10 | PAE cross-entropy loss | `core/loss/confidence.py:pae_loss` | Correlated; uses Algorithms 29-30 |
| 11 | Expected PAE | `core/metrics/confidence.py:probs_to_expected_error`, called for PAE by `core/metrics/aggregate_confidence_ranking.py:_get_confidence_scores` | Correlated |
| 12 | PDE cross-entropy loss | `core/loss/confidence.py:pde_loss` | Direct |
| 13 | Expected PDE | `core/metrics/confidence.py:probs_to_expected_error`, called for PDE by `core/metrics/aggregate_confidence_ranking.py:_get_confidence_scores` | Correlated |
| 14 | Experimentally resolved loss | `core/loss/confidence.py:all_atom_experimentally_resolved_loss` | Direct |
| 15 | Final confidence + diffusion + distogram loss | `core/loss/loss_module.py:OpenFold3Loss.loss`; component weighting occurs in `confidence_loss`, `diffusion_loss`, and `all_atom_distogram_loss` | Correlated |
| 16 | Global PDE (gPDE) | `core/metrics/confidence.py:compute_global_predicted_distance_error` | Direct |
| 17 | pTM | `core/metrics/confidence.py:compute_ptm` with `interface=False` | Direct |
| 18 | ipTM | `core/metrics/confidence.py:compute_ptm` with `interface=True` | Direct |
| 19 | Full-complex sample ranking | `core/metrics/sample_ranking.py:full_complex_sample_ranking_metric` | Correlated; source cites SI 5.9.3 item 1 and implements the formula explicitly |
| 20 | Interface/bespoke sample ranking | `core/metrics/sample_ranking.py:compute_chain_pair_iptm` | Correlated; source cites SI 5.9.3 item 3 and implements the formula explicitly |

Equation 4 is not the dataset-level `loss_weights` feature. It is the molecular
type upweighting constructed inside `mse_loss`; dataset loss weights separately
enable and scale MSE, bond, smooth-LDDT, confidence, and distogram terms.

## Verification and Investigation Workflow

Use the narrowest executable check that covers the touched ownership boundary.
Tests and environment requirements vary by accelerator and optional kernel, so
consult `AGENTS.md`, `docs/BUILD_TESTING.md`, `pixi.toml`, and `pyproject.toml`
before running commands. Installation and accelerator prerequisites are in
`docs/source/Installation.md` and `docs/source/kernels.md`.

A productive investigation sequence is:

1. Resolve the symbol from **Quick Task Routing**.
2. Read its nearest caller and its config schema/default.
3. Search for direct tests and all call sites of the symbol.
4. State one expected behavior and one check that could disprove it.
5. After permission, make the smallest change and run that focused check.

For cross-cutting changes, verify the relevant subset of this chain:

```text
validated config
  -> dataset feature dictionary
  -> dataloader/collation
  -> OpenFold3.forward
  -> loss or confidence computation
  -> Lightning hook
  -> checkpoint or output writer
```

Common traps:

- Confusing offline preprocessing scripts with online dataset featurization.
- Editing only training or inference feature creation when both modes need the
  new feature.
- Treating `OpenFold3AllAtom` and its `.model` as interchangeable objects.
- Assuming `forward()` returns only an output dictionary.
- Changing a loss formula without checking per-example loss weights/settings.
- Loading a checkpoint without preserving EMA and version compatibility.
- Running broad GPU integration tests when a nearby unit/config test can first
  falsify the change.

## High-Value Symbol Index

- Main project registry: `openfold3/projects/of3_all_atom/project_entry.py`.
- Main architecture: `openfold3/projects/of3_all_atom/model.py`.
- Training/prediction wrapper: `openfold3/projects/of3_all_atom/runner.py` plus
  shared hooks in `openfold3/core/runners/model_runner.py`.
- Experiment orchestration and config-to-object construction:
  `openfold3/entry_points/experiment_runner.py`.
- Training feature coordinator:
  `core/data/framework/single_datasets/base_of3.py:create_all_features`.
- Inference feature coordinator:
  `core/data/framework/single_datasets/inference.py:create_all_features`.
- Prediction serialization: `core/runners/writer.py:OF3OutputWriter`.
- Scalar objective: `core/loss/loss_module.py:OpenFold3Loss`.
- Model input/output contract: `projects/of3_all_atom/model.py:OpenFold3.forward`.
- Experiment schemas: `entry_points/validator.py`.
- Checkpoint and default parameter resolution: `entry_points/parameters.py` and
  `core/utils/checkpoint_loading_utils.py`.
- Public console scripts and dependencies: `pyproject.toml`.

When updating this file, record stable ownership and control flow rather than a
task diary. Refresh the verification date when the named paths and symbols have
been checked against the current tree.