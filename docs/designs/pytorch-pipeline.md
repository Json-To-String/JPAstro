# Reproducible PyTorch galaxy classification pipeline

Status: draft for discussion; implementation choices below are proposals unless marked required.
Author: Json-To-String, with Codex
Date: 2026-11-05
Parent issue: [#8 Convert to Pytorch](https://github.com/Json-To-String/JPAstro/issues/8)
Feature PR: [#10 WIP: Pytorch support](https://github.com/Json-To-String/JPAstro/pull/10)
Design branch: `design/pytorch-pipeline`, based on `reproducibility` at `f68d97e4`.
Integration target: `reproducibility`; subsequent feature integration remains through PR #10.

## Purpose and completion criteria

A researcher must be able to install the project, acquire catalog and image data,
prepare auditable labels and independent partitions, augment training samples,
train a PyTorch model, and evaluate its checkpoint in a fresh process. Every
stage must run separately and compose into one documented workflow.

Required: no data leakage, reproducible science, and outputs understandable by
humans and readable by programs. A successful training notebook alone does not
satisfy these requirements. Accuracy is a measured outcome, not a merge threshold.

The existing notebook records a cached-data run with 505 rows, 83.17% test accuracy,
and 0.9053 ROC AUC. These historical outputs are not an independently reproduced
baseline or evidence that the fresh-download path works. Existing test results
have been inspected and must be treated as exploratory when designing a final study.

## Scope

Required stages are installation, acquisition, catalog preparation and splitting,
PyTorch datasets and augmentation, training/checkpointing, standalone evaluation,
and an integration smoke test. TensorBoard is requested by parent issue #8 and
must accompany ordinary saved metrics. Existing TensorFlow scripts and H5 models
can remain as clearly identified historical artifacts; the supported workflow
must run without TensorFlow. Supporting two active training backends is outside
this proposal. Dataset-wide cleanup overlaps issue #7 and should remain scoped
to files needed for this workflow.

## Scientific objective and scope of claims

Agreed primary objective: classify likely Perseus cluster membership versus
background galaxies at scale, producing an inspectable candidate catalog for
astronomers. Dwarf galaxy identification motivates downstream science but is not
a training target, required label, metric gate or automatic pipeline decision.
Astronomers select dwarf candidates from the sifted catalog after classification.
A high membership score is a prediction for follow-up, not confirmed membership.

Preserve label origin and confidence. PCC morphological candidates and
spectroscopically supported memberships are different evidence tiers. Ambiguous
morphological categories must not become confirmed background merely because
of the binary mapping. Stars are foreground contaminants, not background
galaxies; decide whether to exclude them upstream or explicitly define a broader
nonmember class. Record the policy and report contaminant performance separately.

Required scientific reporting concerns membership precision/recall, contamination,
ROC AUC and cross-entropy on independently labeled objects, with clear population
and selection denominators. Dwarf-specific definitions/evaluation are deferred.
Where measured, preserve magnitude/size metadata for astronomers and optional
stratified membership reporting; no dwarf labels are required to complete #8.
The classifier works on supplied cataloged cutouts, not source detection in full
survey images. Candidate-ranking and rejection reports must preserve record IDs
and coordinates so astronomers can inspect or crossmatch objects independently.

### Configurable brightness selection

Required: a brightness section in the versioned config with `enabled`, `band`,
`magnitude_column`, `max_magnitude` and an explicit missing-value policy. A
historical reproduction example may set r-band maximum magnitude to 19.4 with
inclusive <= comparison; this is a recorded experiment setting, not a universal
hardcoded limit. The configuration must support disabling the cut. Record the
photometric convention/extinction correction and distinguish brightness selection
from the red-pixel image-quality filter. Larger magnitude means fainter.

Apply the fixed configured cut before freezing the eligible dataset and split.
Record every exclusion and class count, and keep raw data unchanged. A changed
cut produces a new dataset version and requires a fresh split/leakage audit;
never silently refilter a frozen experiment. Missing magnitudes must follow the
explicit error/exclude/keep policy with a report. Choose/tune the cut using
training/development data, not final-test performance. Training selection and
bulk-inference selection must be recorded separately: scoring a fainter population
is allowed when explicitly requested, but its performance is not established by
a bright-only test set. Avoid representing excluded objects as predicted background.

Other-cluster testing is an optional external validation experiment, not a
prerequisite for the initial Perseus pipeline. It can measure transfer under a
specified domain change, not automatically improve the trained model. Distances,
angular scales, imaging depth/PSF, rendering and labels must be recorded; define
membership relative to each cluster rather than reusing Perseus's redshift window.
Adding another cluster to training is a separate experiment and removes its
status as an untouched external test. A genuinely held-out sky region or
independently labeled Perseus sample is a closer first test of the stated goal.
The exact spatial holdout and independence policy remain open.

The [Wittmann catalog](https://arxiv.org/abs/1911.00951) describes morphological
low-mass candidates. The [Euclid Perseus dwarf study](https://doi.org/10.1051/0004-6361/202450799)
provides a possible crossmatch/reference for future evaluation, not approved
training truth or a guarantee of independent labels. Audit shared objects,
label provenance and survey differences before using it.

## Device and interchangeable model contracts

Required config: `device = "auto" | "cpu" | "cuda"` (with an optional explicit
CUDA index). `auto` chooses CUDA when available, otherwise CPU; explicit CUDA
requests fail clearly when unavailable. Record requested and resolved device,
GPU model, PyTorch/CUDA versions and deterministic settings. Perform a small
startup tensor operation to expose incompatible runtime/device combinations.
Runtime errors and out-of-memory failures must not silently switch devices or
change the scientific configuration. Keep single-device training in scope;
multi-GPU/distributed training and mixed precision are future options.

All models, batches and relevant tensors use the resolved device consistently.
CPU remains fully supported for the smoke workflow and checkpoint evaluation.
Test auto selection and unavailable-device errors without requiring GPU hardware;
record a real CUDA short run when hardware is available. GPU availability cannot
be established by mock tests, and hardware-dependent numerical differences must
be documented. Pinning/transfers may optimize CUDA later without changing outputs.

Agreed delivery scope: one baseline model, with additional architectures and
plugin/registry machinery deferred. Select torchvision ResNet18 initialized with
explicit `ResNet18_Weights.IMAGENET1K_V1` for the initial research configuration;
replace the final layer with two raw logits for cross-entropy. Permit weights=None
for offline fixtures and explicitly recorded scratch-training runs. This baseline
already exists in the notebook and is smaller than historical ResNet50, reducing
migration changes. Torchvision lists about 11.7M parameters and 1.81 GFLOPs for
ResNet18 versus 25.6M and 4.09 GFLOPs for ResNet50; these are reference image-model
figures, not measured astronomy accuracy or end-to-end throughput.
See [torchvision's model definitions](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py).

Keep construction behind one `build_model(config)` function, record architecture,
explicit weight version/checksum, input size and normalization, and reject
unsupported names. Do not implement a registry or require a second model to close
#8. Maintain the two-logit output and checkpoint contracts so a later architecture
can be added without rewriting training/evaluation. Evaluation/inference recreates
the recorded architecture with no initialization-weight download.

Acceptance: a short CPU training/checkpoint/evaluation round trip for ResNet18,
and a real CUDA short run when hardware is available. Record training time,
inference throughput, batch size and device on representative data before making
speed claims. ResNet18's membership performance remains to be validated. A
MobileNet-family model is a future alternative if measured bulk-scoring cost is
the limiting factor, not an extra implementation requirement now.

## README acceptance contract

#17 owns a copyable walkthrough in the root README: clean CPU installation and
CUDA setup/verification, data-source prerequisites, download into a clean folder,
reuse of existing data, config selection/editing, red-image rejection, frozen
splits, training rotations, the supported model/initialization, training and optional
early stopping, Ctrl-C/resume, fresh-process evaluation, and output interpretation.
#11 owns installation details; #15 supplies device/model/config examples.
Explain the scientific target and dwarf/membership distinction in plain language.
Provide one small CPU example with exact commands and an example config; document
how to configure brightness, initialization and device settings and how to locate saved metrics and
rejected-object reports. Validate these commands in #17's integration exercise.
During design, clearly label unimplemented commands as planned; do not present a
future CLI as currently runnable. Retain historical TensorFlow instructions only
in a labeled legacy section once the supported workflow is delivered.

## Pipeline alignment with the user's workflow

The requested stages map to the existing subissues as follows. Installation and
integration remain supporting work; they do not add scientific stages.

| User stage | Subissue | Contract |
| --- | --- | --- |
| 1. SDSS download | #12 | One straightforward acquisition into a clean raw-image folder, with a manifest and basic failure checks |
| 2a. Red-pixel cleaning/filter | #14 implements; #13 records selections | Deterministic quality filtering before the split; preserve raw images and report red fraction, decision and exclusions |
| 3. Train/validation/test split | #13 | Split independent original-object groups once, persist membership and seed, audit leakage |
| 2b. Rotations | #14 | Apply configurable rotations to training samples after splitting; evaluation preprocessing stays fixed |
| 4. Choose model/hyperparameters | #15 | Explicit validated config file, including model/initialization, optimizer and early-stopping settings |
| 5. Train/fit | #15 | Configured optimizer, optional validation-only early stopping, best weights and resumable checkpoint on graceful quit |
| 6. Evaluation/metrics | #16 | Training/validation loss versus epoch, confusion matrix and probability-based ROC AUC; optional Brier score |
| Supporting install and integration | #11, #17 | Clean installation and an independently runnable, tested workflow |

Execution order is download -> fixed quality filter -> identity/group split ->
training rotations -> configured training -> evaluation. Rotations appear after
splitting because splitting generated images would allow views of the same object
into different partitions. Configuring a rotation policy beforehand is harmless;
applying it beforehand is unnecessary and increases the risk of leakage.

### Simple acquisition

Keep #12 small: download requested SDSS cutouts into a designated raw folder,
validate status and image decoding, write an image/source manifest, and clearly
report failed objects. A finite timeout and a small retry allowance are enough.
Reruns may skip verified expected files; a download manager, concurrency,
background services and sophisticated recovery are outside scope. Raw-image
checksums and source metadata support scientific reproducibility without adding
an acquisition framework. The supported pipeline must consume this folder on
subsequent runs without making acquisition mandatory.

### Red-pixel filtering and rotations

Red-pixel filtering is data-quality selection, distinct from creating rotated
training examples. #14 owns both image operations; #13 consumes the filter report
before freezing splits. This avoids a dependency cycle: the deterministic filter
can be delivered/tested before the rotation Dataset integration.

The historical implementation in `src/perseusResNet50.py` uses OpenCV HSV masks
`[0,80,20]..[10,255,255]` and `[160,100,20]..[179,255,255]`, and retains images
with red percentage <= 50. It also flags percentage >= 50 as red, so the equality
boundary is inconsistent. The denominator assumes square images. These are
reference behavior and known ambiguities, not approved new defaults.

Define red fraction using height * width; explicitly test HSV channel ordering,
threshold boundaries, invalid images and rectangular images. Report original
record ID, red fraction, filter policy/configuration, keep/reject and reason.
Agreed: reject whole contaminated images from the eligible dataset and preserve
the raw files. Do not repair or mask their pixels. The red-percentage threshold
and equality boundary still need agreement.
Any thresholds learned or tuned from data must use training data only. Fixed
rules may run before splitting, but defining them using held-out model outcomes
would contaminate that holdout. Report resulting class counts and make clear that
metrics describe the retained image-quality population.

Rotations must be configurable, training-only and reproducible from recorded
settings. The exact angle distribution, interpolation, padding and optional
flips remain discussion items; do not assume the notebook's +/-10 degrees or the
historical fixed angle list is the agreed policy.

### Configuration, stopping and checkpoint lifecycle

A versioned config file must specify device selection, supported model architecture, initialization/weights,
optimizer and its hyperparameters, batch size, epoch limit, seeds, transforms,
checkpoint selection metric/direction, and early-stopping enablement, monitored
validation metric, patience and minimum improvement. Keep the resolved config
with each run. Validate values before starting training. A config references the
frozen split instead of silently generating a different split for each model.

Save best weights whenever the configured validation criterion improves, and a
resumable last checkpoint at completed epoch boundaries. On Ctrl-C or a supported
graceful termination, preserve a consistent resumable checkpoint and record the
interruption. If interrupted mid-epoch, the initial implementation may resume
from the last completed epoch; document that boundary and never claim exact
mid-batch replay. Preserve the last checkpoint on normal completion and early
stopping too; do not overwrite best weights with worse interrupted weights.
Forced termination such as SIGKILL or power loss cannot guarantee a final write.
Checkpoint writes must not corrupt the previous usable file. Include optimizer,
scheduler if used, RNG, completed epoch and early-stopping/selection state.

### Metrics contract

Required: saved and plotted sample-weighted training/validation loss per epoch,
final test loss, confusion matrix (raw counts with class names), and binary ROC
curve/AUC calculated from positive-class probabilities. Do not evaluate the final
test set each epoch. Include accuracy and per-class precision/recall/F1 with
support counts for interpretation; precision-recall reporting remains useful.

Agreed: use cross-entropy for training and reported classification loss.
For the proposed two-logit model, use `torch.nn.CrossEntropyLoss` with raw logits
and integer class labels; convert logits to probabilities for ROC AUC reporting. Plain MSE on thresholded binary
class predictions is just the misclassification rate and adds little to accuracy.
If squared probability error is desired, report binary Brier score explicitly as
mean((p_positive - y)^2), optionally; it does not replace ROC AUC or classification
loss. See the [scikit-learn metric definitions](https://scikit-learn.org/stable/modules/model_evaluation.html).
Training-curve generation consumes persisted history, so it works in a fresh
process. Every metric must be reproducible from saved predictions/history and
state its partition, label mapping, probability column and threshold. Use only
validation data for threshold selection and early stopping.

## Scientific contract: required safeguards

1. Preserve input catalogs and image bytes with checksums and provenance. Record
   source release, query or source file, acquisition time, coordinate units,
   cutout size/scale, catalog row identifiers, and software/code versions.
   A seed cannot reproduce an upstream service that changes its data.
2. Parse labels through an explicit, versioned mapping. Unknown values and
   conflicting labels must stop preparation or enter an explicit excluded-record
   report; never silently disappear. Record counts and reasons for every cut,
   rejected image, unmatched object, and class-balancing operation.
3. Resolve object identity across catalogs before splitting. Group repeated
   observations, alternate IDs, duplicate images, and cutouts that violate the
   approved sky-overlap policy. A source-specific object ID or unequal image
   checksum alone is insufficient to establish independence. Identity and spatial
   grouping must be deterministic, with inspectable evidence and parameters.
4. Assign complete groups to one partition before augmentation or resampling.
   Persist membership and dataset fingerprints. Assert disjoint identity groups,
   duplicate content, and the approved sky-footprint exclusion rules across all
   partitions, including any external evaluation cohort. Ambiguous cases block
   release until resolved or explicitly excluded. Record connected/transitive
   matches rather than matching rows independently.
5. Fit preprocessing statistics, class weights, sampling, and learned transforms
   using training data only. Apply random augmentation only to training data.
   Validation and test preprocessing must be fixed and recorded. Never augment
   before splitting; derived samples inherit their original identity group.
6. Use validation for model selection, thresholds, calibration, early stopping,
   and hyperparameters. Reserve the final test cohort for the frozen experiment.
   Repeatedly inspecting a holdout makes it exploratory; record that history and
   do not claim it is untouched. External cohorts also need cross-catalog overlap
   checks. Distribution plots of test outcomes cannot guide model selection.
7. Evaluation must verify the checkpoint's training/validation provenance against
   the evaluation manifest and refuse forbidden overlaps or incompatible label
   mappings and preprocessing. A fresh process must not reconstruct a random split.
8. Record seeds, RNG states for resumable training, library versions, hardware,
   device and deterministic settings. Document reproducibility limits across
   devices; promise repeatable partitioning and auditable experiments rather than
   universal bitwise equality. Report actual denominators and per-class results.

These controls prevent specified leakage mechanisms; scientific independence
still requires an agreed observational unit and spatial policy. Do not claim
that a passing ID-disjointness check proves all forms of independence.

## Proposed stage boundaries and artifacts

Use an importable `src/jpastro/` package with a thin CLI and a notebook that calls
the same functions. This avoids notebook execution order as hidden state. Keep
the implementation small; CSV, JSON, TOML, and standard PyTorch are sufficient.
The following command names and module paths are proposed contracts, not existing APIs.

| Stage | Proposed module | Inputs | Durable outputs |
| --- | --- | --- | --- |
| Prepare | `prepare.py` | Original catalogs, label/cut/group policy | Manifest, exclusions, identity evidence, cohort counts |
| Download | `download.py` | Selected manifest, acquisition config | Validated images, acquisition records, image checksums |
| Quality filter | `data.py` (pure image-quality function) | Raw images, fixed red-pixel policy | Per-record quality/exclusion report; raw images retained |
| Split/validate | `splits.py` | Identity evidence, validated images, frozen split policy | Frozen membership, leakage audit, dataset fingerprint |
| Load/augment | `data.py` | Frozen manifest, transform config | Tensor/label batches; transform specification |
| Train | `train.py`, `model.py` | Training/validation membership, model/run config | Best and resumable checkpoints, history, TensorBoard logs |
| Evaluate | `evaluate.py` | Checkpoint, explicit evaluation membership | Predictions, metrics, figures, human report |
| Score | shared inference functions / thin CLI | Checkpoint, potentially unlabeled catalog/cutouts | Membership probabilities, ranked candidate catalog, rejection report |

Download can run before final splitting so image-content duplicates and failed
acquisitions are resolved before a dataset is frozen. Preparation must preserve
the full original selection trail. Redownloading or changing exclusions after
freezing creates a new dataset version and requires a fresh audit.

Proposed commands: `python -m jpastro prepare`, `download`, `split`, `train`, and
`evaluate`, and `score`. Each accepts explicit paths/configuration and has actionable failures.
Importing a module must not query a service, read a catalog, or start training.

### Manifest and run contracts

Proposed manifest fields: `schema_version`, `record_id`, `source_catalog`,
`source_row_id`, `source_object_id`, `object_group_id`, `ra_deg`, `dec_deg`,
`original_label`, `binary_label`, `label_policy_version`, `image_path`,
`image_sha256`, `split`, and `exclusion_reason`. Preserve conflicting source
records in a separate identity/label evidence table rather than overwriting them.
Store cutout and source metadata in a versioned dataset JSON file. Resolve image
paths against the dataset root, independent of the caller's working directory.
Represent long catalog IDs as strings, never floating-point numbers.

Proposed run directory:

```text
runs/<run-id>/
  config.toml
  provenance.json
  dataset.json
  splits.csv
  leakage-audit.json
  history.csv
  checkpoints/{best,last}.pt
  tensorboard/
  evaluation/<cohort>/{predictions.csv,metrics.json,report.md,figures/}
```

`provenance.json` records commit and dirty-tree identity, dependency lock/hash,
input and policy hashes, seeds, hardware, weights source/checksum, and settings.
Checkpoints include architecture, label mapping, transforms, partition hashes,
epoch, selection metric, model state, and states needed to resume. Evaluation
can recreate the architecture without downloading pretrained weights. CPU loading
of a GPU-trained checkpoint must work. Predictions retain record/group IDs,
truth, predicted class, and positive-class probability so metrics can be audited.
Losses must aggregate by sample count, including the final partial batch.

### Bulk membership scoring

#16 must expose the same checkpoint/preprocessing inference path for unlabeled
catalogs, independently of evaluation metrics. A `score` command processes bounded
batches without loading all images into memory, preserves identifiers/coordinates,
and writes positive membership probability, thresholded prediction, checkpoint
identity and per-object eligibility/failure status. Missing labels prevent metric
calculation, not scoring. Rejected/failed inputs remain visible in a separate
report and must not be silently classified as background. Thresholds are explicit
and chosen using validation data; ranking by probability lets astronomers adjust
follow-up priorities without re-running inference. Explain that scores require
calibration checks before being interpreted as calibrated membership probabilities.

## Choices to discuss before implementation

| Question | Proposed direction | Strength | Weakness / decision needed |
| --- | --- | --- | --- |
| Observational unit | Galaxy identity plus an explicit cutout-overlap rule | Handles aliases and repeated observations | Matching radius, angular footprint, neighboring galaxies and transitive groups need scientific agreement |
| Generalization claim | Explicitly define whether evaluation means new objects in Perseus or new sky regions | Makes the split answer the research question | Random grouped splits do not establish spatial/domain generalization; spatial blocks may reduce sample size and class balance |
| Labels | Versioned PCC mapping and spectroscopic policy with conflict reporting | Makes target construction inspectable | Merging systems, stars, uncertain members, redshift quality and boundary inclusivity need decisions |
| Model | ResNet18 baseline, explicit initialization in config; extra models deferred | Limits uncontrolled changes | Pretrained ResNet18 is a different experiment from the historical ResNet50; comparison must acknowledge that |
| Augmentation | Training-only online torchvision transforms | Avoids storing many correlated derivatives | Small rotations differ from the original angle set; physical invariances, interpolation and blank corners need review |
| Balancing | Preserve natural evaluation prevalence; training-only policy | Makes evaluation representative of its defined cohort | Overall accuracy can hide class failures; class weighting versus sampling requires a decision |
| Reproducibility | Frozen data/splits plus recorded deterministic settings | Audit and rerun do not depend on current service responses | Archival location and storage costs need agreement; cross-device numerical equivalence has limits |
| Validation | Fixed group-disjoint validation initially; preserve room for grouped cross-validation | Simple initial workflow | With a small sample, one split has high variance; uncertainty should account for independent groups |

No matching tolerance, split fractions, label changes, or new test cohort are
approved by this draft. Agents must not invent scientific defaults for unresolved
choices. Interfaces and synthetic fixtures can proceed while decisions are open.

## Delegation and issue plan

Published issue contracts: [subissue index](pytorch-pipeline-issues.md).

Use #8 as the parent. Create the following subissues as planning work; do not
start independent feature implementations until shared manifest/checkpoint
contracts and scientific policies are agreed. Every child PR targets
`reproducibility`; this design branch also targets `reproducibility`.

| Shard | Scope / suggested ownership | Depends on | Acceptance evidence |
| --- | --- | --- | --- |
| [#11](https://github.com/Json-To-String/JPAstro/issues/11) A Installation | `pyproject.toml`, dependency lock, installation docs | Agreed package layout | Clean CPU install; documented GPU setup; no TensorFlow; complete direct dependencies |
| [#12](https://github.com/Json-To-String/JPAstro/issues/12) B Acquisition | `download.py`, acquisition fixtures | Manifest contract; A for live run | Empty-directory download; Clean raw folder/manifest; basic response/decode checks, timeout and bounded retries; optional verified-file skipping; paths independent of cwd |
| [#13](https://github.com/Json-To-String/JPAstro/issues/13) C Scientific data preparation and isolation | `prepare.py`, `splits.py`, policies, manifest audit | Scientific decisions | Explicit exclusions/counts; catalog aliases, duplicates, overlapping cutouts, conflicting labels, transitive matches and external overlap tested; deterministic persisted splits |
| [#14](https://github.com/Json-To-String/JPAstro/issues/14) D PyTorch datasets and augmentation | `data.py`, transforms | [#13](https://github.com/Json-To-String/JPAstro/issues/13) C contracts | Red-pixel filter/report before split; RGB CHW tensors/labels; configurable training rotations after split; deterministic evaluation |
| [#15](https://github.com/Json-To-String/JPAstro/issues/15) E Training and checkpoints | `model.py`, `train.py`, TensorBoard | A, C, D | Config-driven model/optimizer; optional validation early stopping; best/last/graceful-quit checkpoint; weighted epoch history and provenance |
| [#16](https://github.com/Json-To-String/JPAstro/issues/16) F Independent evaluation | `evaluate.py`, reporting | C/E contracts; E checkpoint | Fresh-process CPU reload; overlap refusal; train/validation loss curves, confusion and probability-based ROC AUC; optional Brier; auditable report |
| [#17](https://github.com/Json-To-String/JPAstro/issues/17) G Integration and scientific acceptance | CLI, notebook wrapper, CI, workflow docs | A–F | Offline synthetic smoke chain and negative leakage cases; separate recorded live acquisition and real-data audit/run |

Agent handoff must include this document revision, issue number, allowed file
ownership, upstream artifact schemas, accepted scientific policies, validation
commands/results, and unresolved assumptions. Coordinate shared API/config changes
through the integration owner. Do not let agents change a split, label policy, or
metric definition merely to make a test or accuracy result pass.

## Verification and release gate

CI uses small synthetic catalogs/images, controlled HTTP fixtures and randomly
initialized small models; no SciServer credentials, live downloads, or pretrained
weight downloads. Negative tests must prove rejection of known aliases, identical
images under different names, spatial overlap under the approved policy,
conflicting labels, invalid manifest paths and incompatible evaluation provenance.
Offline fixtures verify code behavior, not scientific performance.

Before closing #8, run a separate documented real-data acceptance exercise:
install cleanly, acquire a small live subset then the selected dataset, rerun the
cache path, inspect selection/identity/exclusion reports, freeze partitions,
train, and evaluate in a new process. Preserve artifacts and commands. Report
sample/group counts and limitations. A reviewer must inspect the leakage report
and scientific policy alongside passing tests. New science decisions update this
document and affected issue contracts before implementation proceeds.

## Discussion log

- 2026-11-05: User requires a branch off PR #10, a shared Markdown design,
  agent-sized subissues under the existing PyTorch issue, back-and-forth design
  discussion, no data leakage, and reproducible, human-readable science.
- 2026-11-05: Initial proposal written; scientific policy questions remain open.
  No implementation agents have been dispatched.

- 2026-11-05: User supplied the six-stage mental map. Acquisition simplified;
  red-pixel filtering added explicitly; rotations moved after object-group splits;
  config, optimizer, optional early stopping, graceful-quit checkpoints, loss
  curves, confusion matrix and ROC AUC made explicit. Brier proposed as optional.

- 2026-11-05: User confirmed whole-image rejection for red contamination and
  cross-entropy loss. Preserve rejected raw images and exclusion records; do not
  repair pixels. Threshold/boundary and other scientific policies remain open.

- 2026-11-05: User requires CUDA when available, interchangeable models and a
  README walkthrough. Science target clarified as Perseus membership classification
  supporting dwarf candidate identification; other-cluster validation is optional
  and its benefit remains a hypothesis. Added device/factory contracts, dwarf
  selection/completeness caveats, and README acceptance requirements.

- 2026-11-05: User prioritizes membership classification and astronomer-led
  downstream selection; dwarf definitions/metrics no longer gate delivery.
  Brightness selection is configurable and can be disabled. Extra architectures
  deferred; ResNet18 selected as a modest-cost existing baseline. Added bounded
  unlabeled catalog scoring to make the data-sifting objective concrete.
