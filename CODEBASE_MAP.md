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
4. Follow [AGENTS.md](AGENTS.md): explain the smallest proposed change and obtain explicit
   permission before editing code.
5. Update this map when a task discovers a changed ownership boundary or a
   durable new path.

Prefer symbol searches over broad directory scans. Paths in this file are
relative to the top-level repository root, with the `openfold-3` submodule as
their common prefix. A `Class.method` or `module.py:symbol` notation identifies
the intended search anchor.

## Quick Task Routing

| Task | Start here | Then inspect |
| --- | --- | --- |
| Change CLI behavior | [openfold3/run_openfold.py](openfold3/run_openfold.py) | [openfold3/entry_points/experiment_runner.py](openfold3/entry_points/experiment_runner.py), [openfold3/entry_points/validator.py](openfold3/entry_points/validator.py) |
| Change inference inputs | [InferenceDataset.create_all_features](openfold3/core/data/framework/single_datasets/inference.py#L301) | query primitives and the matching featurizer |
| Change training inputs/crops | [BaseOF3Dataset.create_all_features](openfold3/core/data/framework/single_datasets/base_of3.py#L324) | concrete dataset `__getitem__`, sample processor, featurizer |
| Add/change model architecture | [OpenFold3.__init__](openfold3/projects/of3_all_atom/model.py#L66) and [OpenFold3.forward](openfold3/projects/of3_all_atom/model.py#L515) | modules under [openfold3/projects/of3_all_atom](openfold3/projects/of3_all_atom) or [openfold3/core/model](openfold3/core/model) |
| Change training behavior | [OpenFold3AllAtom.training_step](openfold3/projects/of3_all_atom/runner.py#L561) | [ModelRunner](openfold3/core/runners/model_runner.py#L27), optimizer/config hooks |
| Change the objective | [OpenFold3Loss.loss](openfold3/core/loss/loss_module.py#L40) | components under [openfold3/core/loss](openfold3/core/loss), loss configuration |
| Change prediction files | [OF3OutputWriter.write_all_outputs](openfold3/core/runners/writer.py#L233) | inference callbacks and output settings |
| Change checkpoint loading | inference runner [setup](openfold3/entry_points/experiment_runner.py#L797)/[lightning_module](openfold3/entry_points/experiment_runner.py#L681) | [openfold3/core/utils/checkpoint_loading_utils.py](openfold3/core/utils/checkpoint_loading_utils.py), EMA handling |
| Change config or presets | [OF3ProjectEntry](openfold3/projects/of3_all_atom/project_entry.py#L49) | project config, runner validators, example YAMLs |
| Add or locate tests | nearest [tests](openfold-3/tests) mirror of the touched package | [docs/BUILD_TESTING.md](openfold-3/docs/BUILD_TESTING.md), [pixi.toml](openfold-3/pixi.toml), [pyproject.toml](openfold-3/pyproject.toml) |

## Repository Ownership

- [openfold3/run_openfold.py](openfold3/run_openfold.py): Click CLI commands for training and prediction.
- [openfold3/setup_openfold.py](openfold3/setup_openfold.py): installation/cache/checkpoint/CCD setup entry.
- [openfold3/entry_points](openfold3/entry_points): experiment orchestration, validated runner
  configuration, checkpoint/parameter resolution, and command-level utilities.
- [openfold3/core/config](openfold3/core/config): shared configuration utilities and schemas.
- [openfold3/core/data](openfold3/core/data): query/structure primitives, preprocessing,
  featurization, datasets, datamodules, and cache handling.
- [openfold3/core/model](openfold3/core/model): reusable neural-network building blocks.
- [openfold3/core/kernels](openfold3/core/kernels): accelerated kernel selection and implementations.
- [openfold3/core/loss](openfold3/core/loss): differentiable objective components and aggregation.
- [openfold3/core/metrics](openfold3/core/metrics): evaluation and confidence-related metrics.
- [openfold3/core/runners](openfold3/core/runners): shared Lightning runner behavior, callbacks, EMA,
  prediction serialization, and runtime utilities.
- [openfold3/projects/of3_all_atom](openfold3/projects/of3_all_atom): the concrete all-atom project: model,
  runner, model config, presets, and project-specific modules.
- [examples](openfold-3/examples): executable query JSONs, runner YAMLs, training YAMLs, and a full
  reference configuration. Use these before inventing configuration keys.
- [scripts](openfold-3/scripts): offline preprocessing, dataset, MSA, analysis, and developer tools;
  these are generally not part of the online model forward path.
- [tests](openfold-3/tests) and package-local test directories: behavioral checks. Locate the
  nearest test to the ownership point being changed.
- [docs/source](openfold-3/docs/source): user-facing installation, inference, training, input, data
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

Installed console scripts are declared in [pyproject.toml](openfold-3/pyproject.toml):

- `run_openfold` -> [openfold3.run_openfold:cli](openfold3/run_openfold.py#L39).
- `setup_openfold` -> [openfold3.setup_openfold:main](openfold3/setup_openfold.py#L314).
- `validate-openfold3-rocm` -> the ROCm environment validator.

`run_openfold train` reads a runner YAML, applies explicit CLI seed overrides,
validates it as [TrainingExperimentConfig](openfold3/entry_points/validator.py#L347), constructs
[TrainingExperimentRunner](openfold3/entry_points/experiment_runner.py#L378), and calls `setup()` followed by `run()`.

`run_openfold predict` reads query JSON plus optional runner/CLI settings,
resolves a checkpoint, validates inference configuration, constructs
[InferenceExperimentRunner](openfold3/entry_points/experiment_runner.py#L604), and runs Lightning prediction. The public examples
under [examples/example_inference_inputs](openfold-3/examples/example_inference_inputs) and [examples/example_runner_yamls](openfold-3/examples/example_runner_yamls)
are the best starting points for accepted inputs.

Keep these configuration layers distinct:

1. Pydantic experiment schemas in [openfold3/entry_points/validator.py](openfold3/entry_points/validator.py) validate
  orchestration, trainer, data, checkpoint, logging, and output settings.
2. [OF3ProjectEntry](openfold3/projects/of3_all_atom/project_entry.py#L49) in [openfold3/projects/of3_all_atom/project_entry.py](openfold3/projects/of3_all_atom/project_entry.py) starts from the
  base model config, applies named presets from
  [model_setting_presets.yml](openfold3/projects/of3_all_atom/config/model_setting_presets.yml), then applies `model_update.custom`.
3. The resulting `ml_collections.ConfigDict` controls model architecture and
  model-coupled runtime behavior.
4. Explicit CLI options may override selected runner values. Inspect the Click
  command and [update_config_with_cli_args](openfold3/entry_points/experiment_runner.py#L687) before assuming precedence.

Useful configuration references are
[examples/reference_full_config/full_config.yml](openfold-3/examples/reference_full_config/full_config.yml),
[docs/source/configuration_reference.md](openfold-3/docs/source/configuration_reference.md), and the typed project configs under
[openfold3/projects/of3_all_atom/config](openfold3/projects/of3_all_atom/config).

## Input Preprocessing

There is not one shared top-level preprocessing function for every mode. The
pipeline is assembled by the dataset classes, with separate training and
inference entry points that call common sample-processing and featurization
modules.

### Training

The central coordinator is
[BaseOF3Dataset.create_all_features](openfold3/core/data/framework/single_datasets/base_of3.py#L324). In order, it
creates:

1. Cropped target/ground-truth structure and reference-conformer features via
   `create_structure_features()`.
2. MSA features via `create_msa_features()` and `MsaSampleProcessorTrain`.
3. Template features via `create_template_features()`.
4. Per-example loss switches/weights via `create_loss_features()`.

Concrete dataset subclasses implement `__getitem__()` and call this shared
coordinator. [DataModule](openfold3/core/data/framework/data_module.py#L244) constructs datasets and
dataloaders for Lightning.

Validation uses [ValidationDataset.create_all_features](openfold3/core/data/framework/single_datasets/validation.py#L369). It delegates to the training coordinator and
then adds homology-similarity and antibody-antigen features used by validation
metrics.

The lower-level transformations live under:

- [openfold3/core/data/pipelines/sample_processing](openfold3/core/data/pipelines/sample_processing): structure cropping,
  MSA sampling, template processing, and conformer processing.
- [openfold3/core/data/pipelines/featurization](openfold3/core/data/pipelines/featurization): conversion of processed
  biological objects into model tensor dictionaries.
- [openfold3/core/data/primitives](openfold3/core/data/primitives): query, structure, tokenization, and other
  core data representations.

### Inference

The central coordinator is [InferenceDataset.create_all_features](openfold3/core/data/framework/single_datasets/inference.py#L301). It:

1. Converts a `Query` into a tokenized `AtomArray` and reference molecules.
2. Creates structure and conformer features without ground truth.
3. Creates MSA features using `MsaSampleProcessorInference`.
4. Creates template features.
5. Adds inference-only pocket-sampling features.

[InferenceDataset.__getitem__](openfold3/core/data/framework/single_datasets/inference.py#L352) adds query/seed metadata and a `valid_sample`
flag. [InferenceDataModule](openfold3/core/data/framework/data_module.py#L659) supplies
these batches to prediction.

### Best Modification Points

- Change training sample composition/cropping: [openfold3/core/data/pipelines/sample_processing/structure.py](openfold3/core/data/pipelines/sample_processing/structure.py)
  and the concrete training dataset classes.
- Change model tensor definitions: the matching module in
  [openfold3/core/data/pipelines/featurization](openfold3/core/data/pipelines/featurization).
- Change MSA selection/processing: [openfold3/core/data/pipelines/sample_processing/msa.py](openfold3/core/data/pipelines/sample_processing/msa.py).
- Add a feature in both modes: add it to both [BaseOF3Dataset.create_all_features](openfold3/core/data/framework/single_datasets/base_of3.py#L324)
  and [InferenceDataset.create_all_features](openfold3/core/data/framework/single_datasets/inference.py#L301), unless it is deliberately
  training-only or inference-only.

## Complete Model Object

Yes. There are two useful object boundaries:

- [ExperimentRunner.lightning_module](openfold3/entry_points/experiment_runner.py#L166), a cached property in
  [openfold3/entry_points/experiment_runner.py](openfold3/entry_points/experiment_runner.py), instantiates
  `OF3ProjectEntry.runner(...)`. For the all-atom project this is one
  `OpenFold3AllAtom` Lightning module containing the model, loss, EMA state,
  optimizer hooks, and training/prediction steps.
- `OpenFold3AllAtom.model` is the complete neural network. It is created by
  [ModelRunner.__init__](openfold3/core/runners/model_runner.py#L30) in [openfold3/core/runners/model_runner.py](openfold3/core/runners/model_runner.py) as
  `model_class(config)`, where `model_class` is `OpenFold3`.

[OpenFold3.__init__](openfold3/projects/of3_all_atom/model.py#L66) assembles
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

[OpenFold3.forward](openfold3/projects/of3_all_atom/model.py#L515) returns `(batch, output)`, not just `output`.
Callers in the all-atom runner unpack both values. The returned batch can differ
from the input because the model adds a diffusion sample dimension and, during
training/validation, applies safe multi-chain permutation alignment to ground
truth in place.

The authoritative feature and output shape inventory is the docstring on
[OpenFold3.forward](openfold3/projects/of3_all_atom/model.py#L515). Important output
families include trunk single/pair representations, predicted atom positions,
confidence logits, distogram logits, and training-only diffusion values.

Training and inference do not consume exactly the same dictionary:

- Training/validation batches include a nested `ground_truth` dictionary and
  loss-control features.
- Inference batches carry non-tensor metadata such as `atom_array`, `query_id`,
  seed, and validity/repetition flags for prediction and serialization.
- [OpenFold3AllAtom.predict_step](openfold3/projects/of3_all_atom/runner.py#L924) computes confidence scores after the model
  call and stores them in `outputs["confidence_scores"]` before returning the
  `(batch, outputs)` pair to the prediction writer.

When adding a feature, trace all four contracts: creation, collation/device
movement, [OpenFold3.forward](openfold3/projects/of3_all_atom/model.py#L515) consumption, and training loss or output writer
consumption. Do not assume every batch value is a tensor.

## Prediction Outputs

Prediction files are owned by [OF3OutputWriter](openfold3/core/runners/writer.py#L79). The inference runner registers it in
[InferenceExperimentRunner.callbacks](openfold3/entry_points/experiment_runner.py#L832), then Lightning calls
[on_predict_batch_end](openfold3/core/runners/writer.py#L304) after each prediction batch.

[OF3OutputWriter.write_all_outputs](openfold3/core/runners/writer.py#L233) creates:

- Predicted structures through [write_structure_prediction](openfold3/core/runners/writer.py#L115).
- Aggregated and full confidence files through [write_confidence_scores](openfold3/core/runners/writer.py#L179).
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
[openfold3/entry_points/parameters.py](openfold3/entry_points/parameters.py); setup orchestration is in
[openfold3/setup_openfold.py](openfold3/setup_openfold.py). User-facing behavior is documented in
[docs/source/Installation.md](openfold-3/docs/source/Installation.md) and [docs/source/parameters_reference.md](openfold-3/docs/source/parameters_reference.md).

Training checkpoints are Lightning checkpoints and include EMA state through
runner checkpoint hooks. Training resume/selective-loading behavior is
configured through the experiment validator and implemented by the training
experiment runner plus [openfold3/core/utils/checkpoint_loading_utils.py](openfold3/core/utils/checkpoint_loading_utils.py).

Inference loading is deliberately stricter:

1. [InferenceExperimentRunner.setup](openfold3/entry_points/experiment_runner.py#L797) loads the selected checkpoint.
2. [get_state_dict_from_checkpoint](openfold3/core/utils/checkpoint_loading_utils.py#L54) with `init_from_ema_weights=True` selects
  EMA-derived model weights.
3. [_load_state_dict_with_version_validation](openfold3/entry_points/experiment_runner.py#L765) checks missing/unexpected keys
  and the model version tensor before loading the Lightning module.

Do not bypass version validation or silently use non-EMA weights when adapting
inference. Checkpoint compatibility is coupled to the model version and
parameter-set release; filenames alone are not sufficient evidence.

## Final Training Loss

The final differentiable scalar is assembled by [OpenFold3Loss.loss](openfold3/core/loss/loss_module.py#L40):

```text
total = confidence loss
      + diffusion loss (unless confidence-only training)
      + all-atom distogram loss (unless confidence-only training)
```

Each component applies its own configured weighting before it is added to
`cum_loss`. [OpenFold3Loss.forward](openfold3/core/loss/loss_module.py#L120) returns the scalar and, when requested, a
detached component breakdown. Validation may use [loss_chunked](openfold3/core/loss/loss_module.py#L89) to average
per-sample losses with lower memory use.

The live all-atom hook is [OpenFold3AllAtom.training_step](openfold3/projects/of3_all_atom/runner.py#L561). It selects the standard or manual
per-sample-gradient-clipping path; both call the model, then
`self.loss(batch, outputs, _return_breakdown=True)`, log the breakdown, and
return the scalar to Lightning for backpropagation. The base
[ModelRunner.training_step](openfold3/core/runners/model_runner.py#L100) expresses the same shared pattern but is
overridden for this project.

To change the objective:

- Change aggregation or add a top-level term in [OpenFold3Loss.loss](openfold3/core/loss/loss_module.py#L40).
- Change the mathematics of an existing term in its module under
  [openfold3/core/loss](openfold3/core/loss).
- Change weights/settings in the model loss configuration, rooted at
  `model_config.architecture.loss_module`.
- Change per-example enablement/weighting in
  [openfold3/core/data/pipelines/featurization/loss_weights.py](openfold3/core/data/pipelines/featurization/loss_weights.py) and the dataset loss settings.

## AF3 Supplement Algorithms

This section maps Algorithms 1-31 from the AlphaFold 3 supplementary
information ([docs/41586_2024_7487_MOESM1_ESM.pdf](openfold-3/docs/41586_2024_7487_MOESM1_ESM.pdf)) to OpenFold3. Evidence is:

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
| 1 | Main Inference Loop | [OpenFold3.forward](openfold3/projects/of3_all_atom/model.py#L515) and [OpenFold3.run_trunk](openfold3/projects/of3_all_atom/model.py#L175) | Direct |
| 2 | Input Feature Embedder | [InputEmbedderAllAtom](openfold3/core/model/feature_embedders/input_embedders.py#L38) | Direct |
| 3 | Relative Position Encoding | [relpos_complex](openfold3/core/utils/relpos.py#L108), called by [InputEmbedderAllAtom](openfold3/core/model/feature_embedders/input_embedders.py#L38) | Direct at caller |
| 4 | One-hot encoding with nearest bin | [binned_one_hot](openfold3/core/utils/tensor_utils.py#L74) | Correlated shared helper |
| 5 | Atom Attention Encoder | [AtomAttentionEncoder](openfold3/core/model/layers/sequence_local_atom_attention.py#L267), [RefAtomFeatureEmbedder](openfold3/core/model/layers/sequence_local_atom_attention.py#L40), and [NoisyPositionEmbedder](openfold3/core/model/layers/sequence_local_atom_attention.py#L166) | Direct |
| 6 | Atom Attention Decoder | [AtomAttentionDecoder](openfold3/core/model/layers/sequence_local_atom_attention.py#L559) | Direct |
| 7 | Atom Transformer | Block-form [DiffusionTransformer](openfold3/core/model/layers/diffusion_transformer.py#L187) and [CrossAttentionPairBias](openfold3/core/model/layers/attention_pair_bias.py#L397), instantiated by the atom attention encoder/decoder; block conversion is in [openfold3/core/utils/atom_attention_block_utils.py](openfold3/core/utils/atom_attention_block_utils.py) | Shared correlation |
| 8 | MSA Module | [MSAModuleEmbedder](openfold3/core/model/feature_embedders/input_embedders.py#L177) (lines 1-4) and [MSAModuleStack](openfold3/core/model/latent/msa_module.py#L287)/[MSAModuleBlock](openfold3/core/model/latent/msa_module.py#L38) (lines 5-15) | Direct |
| 9 | Outer Product Mean | [OuterProductMean](openfold3/core/model/layers/outer_product_mean.py#L28) | Direct |
| 10 | MSA Pair Weighted Averaging | [MSAPairWeightedAveraging](openfold3/core/model/layers/msa.py#L511) | Direct |
| 11 | Transition | [SwiGLUTransition](openfold3/core/model/layers/transition.py#L233) | Direct |
| 12 | Triangle Multiplication Outgoing | [TriangleMultiplicationOutgoing](openfold3/core/model/layers/triangular_multiplicative_update.py#L1195) (plus fused variant) | Direct |
| 13 | Triangle Multiplication Incoming | [TriangleMultiplicationIncoming](openfold3/core/model/layers/triangular_multiplicative_update.py#L1203) (plus fused variant) | Direct |
| 14 | Triangle Attention Starting Node | [TriangleAttentionStartingNode](openfold3/core/model/layers/triangular_attention.py#L205) | Direct |
| 15 | Triangle Attention Ending Node | [TriangleAttentionEndingNode](openfold3/core/model/layers/triangular_attention.py#L208) | Direct |
| 16 | Template Embedder | [TemplatePairEmbedderAllAtom](openfold3/core/model/feature_embedders/template_embedders.py#L28) and [TemplateEmbedderAllAtom](openfold3/core/model/latent/template_module.py#L490) | Direct |
| 17 | Pairformer Stack | [PairFormerStack](openfold3/core/model/latent/pairformer.py#L218) and [PairFormerBlock](openfold3/core/model/latent/pairformer.py#L37) | Direct |
| 18 | Sample Diffusion | [SampleDiffusion](openfold3/core/model/structure/diffusion_module.py#L256) | Direct |
| 19 | Centre Random Augmentation | [centre_random_augmentation](openfold3/core/model/structure/augmentation.py#L43) | Direct |
| 20 | Diffusion Module | [DiffusionModule](openfold3/core/model/structure/diffusion_module.py#L92) | Direct |
| 21 | Diffusion Conditioning | [DiffusionConditioning](openfold3/core/model/layers/diffusion_conditioning.py#L28) | Direct |
| 22 | Fourier Embedding | [FourierEmbedding](openfold3/core/model/feature_embedders/input_embedders.py#L573) | Direct |
| 23 | Diffusion Transformer | [DiffusionTransformer](openfold3/core/model/layers/diffusion_transformer.py#L187) and [DiffusionTransformerBlock](openfold3/core/model/layers/diffusion_transformer.py#L34) | Direct |
| 24 | Attention with Pair Bias | [AttentionPairBias](openfold3/core/model/layers/attention_pair_bias.py#L34), [DiffusionAttentionPairBias](openfold3/core/model/layers/attention_pair_bias.py#L212), and [CrossAttentionPairBias](openfold3/core/model/layers/attention_pair_bias.py#L397) | Direct |
| 25 | Conditioned Transition Block | [ConditionedTransitionBlock](openfold3/core/model/layers/transition.py#L281) | Direct |
| 26 | Adaptive LayerNorm | [AdaLN](openfold3/core/model/primitives/normalization.py#L81) | Direct |
| 27 | Smooth LDDT Loss | [smooth_lddt_loss](openfold3/core/loss/diffusion.py#L370) | Direct |
| 28 | Weighted Rigid Align | [weighted_rigid_align](openfold3/core/loss/diffusion.py#L32) | Direct |
| 29 | Express Coordinates in Frame | [express_coords_in_frames](openfold3/core/loss/confidence.py#L35) | Direct |
| 30 | Compute Alignment Error | [compute_alignment_error](openfold3/core/loss/confidence.py#L96) | Direct |
| 31 | Confidence Head | [AuxiliaryHeadsAllAtom](openfold3/core/model/heads/head_modules.py#L40); projections are in [PairformerEmbedding](openfold3/core/model/heads/prediction_heads.py#L28), [PredictedAlignedErrorHead](openfold3/core/model/heads/prediction_heads.py#L371), [PredictedDistanceErrorHead](openfold3/core/model/heads/prediction_heads.py#L442), [PerResidueLDDTAllAtom](openfold3/core/model/heads/prediction_heads.py#L514), and [ExperimentallyResolvedHeadAllAtom](openfold3/core/model/heads/prediction_heads.py#L582) | Direct |

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
| 1 | Weighted PDB chain/interface sampling | [DatapointCollection.create_datapoint_cache](openfold3/core/data/framework/single_datasets/pdb.py#L143) (nested `calculate_datapoint_probability`) | Correlated; its docstring calls this “Algorithm 1” from SI 2.5.1 |
| 2 | Align ground truth to denoised coordinates | [mse_loss](openfold3/core/loss/diffusion.py#L106) calling [weighted_rigid_align](openfold3/core/loss/diffusion.py#L32) | Correlated; aligner is directly tagged Algorithm 28 |
| 3 | Weighted aligned MSE | [mse_loss](openfold3/core/loss/diffusion.py#L106) | Direct |
| 4 | DNA/RNA/ligand per-atom weights | Weight construction at the start of [mse_loss](openfold3/core/loss/diffusion.py#L106) | Correlated |
| 5 | Polymer-ligand bond loss | [bond_loss](openfold3/core/loss/diffusion.py#L182) and [bond_loss_sparse](openfold3/core/loss/diffusion.py#L240) | Direct |
| 6 | Combined diffusion loss | [diffusion_loss](openfold3/core/loss/diffusion.py#L474) | Direct |
| 7 | Inference noise schedule | [create_noise_schedule](openfold3/core/model/structure/diffusion_module.py#L52) | Correlated; source cites the SI page rather than Equation 7 |
| 8 | Per-atom LDDT confidence target | Target construction inside [all_atom_plddt_loss](openfold3/core/loss/confidence.py#L127) | Correlated |
| 9 | pLDDT cross-entropy loss | [all_atom_plddt_loss](openfold3/core/loss/confidence.py#L127) | Correlated |
| 10 | PAE cross-entropy loss | [pae_loss](openfold3/core/loss/confidence.py#L273) | Correlated; uses Algorithms 29-30 |
| 11 | Expected PAE | [probs_to_expected_error](openfold3/core/metrics/confidence.py#L28), called for PAE by [_get_confidence_scores](openfold3/core/metrics/aggregate_confidence_ranking.py#L32) | Correlated |
| 12 | PDE cross-entropy loss | [pde_loss](openfold3/core/loss/confidence.py#L367) | Direct |
| 13 | Expected PDE | [probs_to_expected_error](openfold3/core/metrics/confidence.py#L28), called for PDE by [_get_confidence_scores](openfold3/core/metrics/aggregate_confidence_ranking.py#L32) | Correlated |
| 14 | Experimentally resolved loss | [all_atom_experimentally_resolved_loss](openfold3/core/loss/confidence.py#L434) | Direct |
| 15 | Final confidence + diffusion + distogram loss | [OpenFold3Loss.loss](openfold3/core/loss/loss_module.py#L40); component weighting occurs in [confidence_loss](openfold3/core/loss/confidence.py#L512), [diffusion_loss](openfold3/core/loss/diffusion.py#L474), and [all_atom_distogram_loss](openfold3/core/loss/distogram.py#L25) | Correlated |
| 16 | Global PDE (gPDE) | [compute_global_predicted_distance_error](openfold3/core/metrics/confidence.py#L49) | Direct |
| 17 | pTM | [compute_ptm](openfold3/core/metrics/confidence.py#L79) with `interface=False` | Direct |
| 18 | ipTM | [compute_ptm](openfold3/core/metrics/confidence.py#L79) with `interface=True` | Direct |
| 19 | Full-complex sample ranking | [full_complex_sample_ranking_metric](openfold3/core/metrics/sample_ranking.py#L24) | Correlated; source cites SI 5.9.3 item 1 and implements the formula explicitly |
| 20 | Interface/bespoke sample ranking | [compute_chain_pair_iptm](openfold3/core/metrics/sample_ranking.py#L208) | Correlated; source cites SI 5.9.3 item 3 and implements the formula explicitly |

Equation 4 is not the dataset-level `loss_weights` feature. It is the molecular
type upweighting constructed inside `mse_loss`; dataset loss weights separately
enable and scale MSE, bond, smooth-LDDT, confidence, and distogram terms.

## Verification and Investigation Workflow

Use the narrowest executable check that covers the touched ownership boundary.
Tests and environment requirements vary by accelerator and optional kernel, so
consult [AGENTS.md](AGENTS.md), [docs/BUILD_TESTING.md](openfold-3/docs/BUILD_TESTING.md), [pixi.toml](openfold-3/pixi.toml), and [pyproject.toml](openfold-3/pyproject.toml)
before running commands. Installation and accelerator prerequisites are in
[docs/source/Installation.md](openfold-3/docs/source/Installation.md) and [docs/source/kernels.md](openfold-3/docs/source/kernels.md).

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

- Main project registry: [openfold3/projects/of3_all_atom/project_entry.py](openfold3/projects/of3_all_atom/project_entry.py).
- Main architecture: [openfold3/projects/of3_all_atom/model.py](openfold3/projects/of3_all_atom/model.py).
- Training/prediction wrapper: [openfold3/projects/of3_all_atom/runner.py](openfold3/projects/of3_all_atom/runner.py) plus
  shared hooks in [openfold3/core/runners/model_runner.py](openfold3/core/runners/model_runner.py).
- Experiment orchestration and config-to-object construction:
  [openfold3/entry_points/experiment_runner.py](openfold3/entry_points/experiment_runner.py).
- Training feature coordinator:
  [BaseOF3Dataset.create_all_features](openfold3/core/data/framework/single_datasets/base_of3.py#L324).
- Inference feature coordinator:
  [InferenceDataset.create_all_features](openfold3/core/data/framework/single_datasets/inference.py#L301).
- Prediction serialization: [OF3OutputWriter](openfold3/core/runners/writer.py#L79).
- Scalar objective: [OpenFold3Loss](openfold3/core/loss/loss_module.py#L31).
- Model input/output contract: [OpenFold3.forward](openfold3/projects/of3_all_atom/model.py#L515).
- Experiment schemas: [openfold3/entry_points/validator.py](openfold3/entry_points/validator.py).
- Checkpoint and default parameter resolution: [openfold3/entry_points/parameters.py](openfold3/entry_points/parameters.py) and
  [openfold3/core/utils/checkpoint_loading_utils.py](openfold3/core/utils/checkpoint_loading_utils.py).
- Public console scripts and dependencies: [pyproject.toml](openfold-3/pyproject.toml).

When updating this file, record stable ownership and control flow rather than a
task diary. Refresh the verification date when the named paths and symbols have
been checked against the current tree.