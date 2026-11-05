# PyTorch pipeline subissue drafts

Parent: [#8 Convert to Pytorch](https://github.com/Json-To-String/JPAstro/issues/8).
Design: [pytorch-pipeline.md](pytorch-pipeline.md).

These are planning contracts pending scientific design discussion. All seven are
attached as native GitHub subissues of #8. PRs target
`reproducibility`. Do not dispatch implementation agents with unresolved science
choices. A checklist link alone is not a native subissue relationship.

## A: Make the PyTorch pipeline installable and reproducible

GitHub: [#11](https://github.com/Json-To-String/JPAstro/issues/11). Native subissue of #8.

Own pyproject.toml, dependency locking, package layout setup and install documentation. Deliver a clean CPU install, documented GPU setup, all direct dependencies, and a supported workflow requiring no TensorFlow. Resolve missing license/package metadata and SciServer acquisition setup. Coordinate package layout with integration.

Document and verify CPU installation plus compatible CUDA setup, runtime verification and troubleshooting. Record software/runtime versions. CPU remains a fully supported path; do not require GPU hardware for normal CI.

Dependencies: Manifest/package layout agreement.

Handoff: owned files, agreed API/artifact contracts, design revision, validation
commands/results, and remaining assumptions. Coordinate shared schemas through
the integration owner. Preserve provenance and human-readable outputs.

## B: Download SDSS images into a clean reusable dataset folder

GitHub: [#12](https://github.com/Json-To-String/JPAstro/issues/12). Native subissue of #8.

Own a simple acquisition module and HTTP fixtures. Fix the downloader tuple
crash and notebook fresh-download path error. Download selected SDSS cutouts into
a clean raw-image folder, validate HTTP status and image decoding, use finite
timeouts and small bounded retries, and clearly report failed objects. Save a
source/image manifest with checksums; paths work independently of cwd. Subsequent
pipeline stages consume the existing folder without downloading again. Optional
reruns skip verified expected files. A download manager, concurrency, background
services and sophisticated recovery are outside scope.

Dependencies: Manifest contract; A for live execution.

Handoff: owned files, agreed API/artifact contracts, design revision, validation
commands/results, and remaining assumptions. Coordinate shared schemas through
the integration owner. Preserve provenance and human-readable outputs.

## C: Prepare auditable catalogs and enforce leakage-free partitions

GitHub: [#13](https://github.com/Json-To-String/JPAstro/issues/13). Native subissue of #8.

Own catalog preparation, identity grouping, versioned label policies, split manifests and leakage audits. Preserve raw sources; report unknown/conflicting labels and all exclusions/counts. Resolve aliases across catalogs, repeated observations, content duplicates, transitive matches and spatial cutout overlap under an agreed scientific policy BEFORE splitting. Persist deterministic group-disjoint partitions and fingerprints. Check external cohorts too. Consume #14's fixed red-pixel quality report before freezing splits; retain rejection reasons and class counts. Fit balancing/statistics only on training data. Negative tests must reject known leakage. Do not invent matching tolerances or scientific labels.

Primary science target: binary Perseus membership classification for astronomer-led downstream selection. Dwarf identification, definitions and dwarf metrics are deferred. Preserve source label confidence and coordinates; resolve foreground-star policy explicitly. Add brightness config: enabled/disabled, band, magnitude column, inclusive maximum magnitude and missing-value policy. Record photometric convention and exclusions/counts; no hardcoded 19.4 limit. Changed cuts create a new frozen dataset/audit rather than silently editing splits. Distinguish training brightness selection from scoring population.

Dependencies: Agreement on observational unit, spatial independence, labels and evaluation population; schema contract.

Handoff: owned files, agreed API/artifact contracts, design revision, validation
commands/results, and remaining assumptions. Coordinate shared schemas through
the integration owner. Preserve provenance and human-readable outputs.

## D: Implement PyTorch datasets and training-only augmentation

GitHub: [#14](https://github.com/Json-To-String/JPAstro/issues/14). Native subissue of #8.

Own deterministic red-pixel quality filtering as well as Dataset/DataLoader and
torchvision rotations. Deliver a pure filter/report function first; #13 consumes
its per-record scores, decisions and exclusions before freezing splits. Preserve
raw images. Document historical HSV thresholds, square-image denominator and
50-percent equality ambiguity; test actual height*width, channel ordering and
boundaries. Agreed: reject whole contaminated images from the eligible dataset,
retain original files and exclusion records, and do not repair or mask pixels.
The red-percentage threshold and equality boundary remain open.
Then deliver the split-aware Dataset/rotation integration. Consume frozen manifests; produce RGB CHW tensors and integer class labels. Training-only stochastic augmentation; deterministic validation/test transforms. Derived samples retain identity groups; no pre-split augmentation or held-out fitted statistics. Record transforms and seed worker/sampler behavior. Explain differences from historical rotation policy.

Support the resolved training device without coupling image-quality filtering to CUDA. The ResNet18 builder specifies input dimensions/normalization; record compatible transforms and filtering reports. Other model adapters are deferred.

Dependencies: Shared manifest/report schema; fixed filter policy. Rotation integration requires C splits and agreed augmentation settings; the pure filter can land first.

Handoff: owned files, agreed API/artifact contracts, design revision, validation
commands/results, and remaining assumptions. Coordinate shared schemas through
the integration owner. Preserve provenance and human-readable outputs.

## E: Implement reproducible PyTorch training and checkpointing

GitHub: [#15](https://github.com/Json-To-String/JPAstro/issues/15). Native subissue of #8.

Agreed loss: cross-entropy. For the proposed two-logit classifier use
`torch.nn.CrossEntropyLoss` on raw logits and integer class targets.
Own model/training modules, a validated model/hyperparameter config file and
TensorBoard integration. Config includes optimizer parameters, epoch limit,
batch size, initialization, seeds, best-checkpoint metric/direction and optional
validation-only early stopping with patience/minimum improvement. Save a resumable
checkpoint on Ctrl-C/supported graceful quit as well as normal completion and
early stopping; preserve best weights independently. Mid-epoch interruption may
resume at the last completed epoch, explicitly recorded; test that behavior and
safe checkpoint writes. Preserve early-stopping/scheduler state when enabled. Run short CPU training with explicit architecture/initialization config, seeded initialization and sample-weighted loss aggregation. Save best and resumable last checkpoints with model/optimizer/RNG state, label mapping, preprocessing, partition hashes, selection metric and provenance. Use validation for model selection; never use final test outcomes for tuning. Include machine-readable history beside TensorBoard. Make architecture changes from the historical ResNet50 explicit.

Required: config device auto/cpu/cuda, auto CUDA detection with CPU fallback, explicit unavailable-CUDA error, startup runtime check and recorded resolved hardware. No silent runtime/OOM fallback. Deliver one baseline: torchvision ResNet18 with explicit IMAGENET1K_V1 initialization and two output logits; allow weights=None for offline fixtures/recorded scratch runs. Keep a simple build_model(config) boundary, preprocessing and architecture/weights provenance, with checkpoint reconstruction requiring no weight download. Additional architectures, registries and custom-model plugins are deferred. Validate CPU round trip and record real CUDA execution when available. Measure representative training/scoring cost before throughput claims; no astronomy performance guarantee is inferred from ImageNet results.

Dependencies: A, C, D; agreed checkpoint contract and baseline policy.

Handoff: owned files, agreed API/artifact contracts, design revision, validation
commands/results, and remaining assumptions. Coordinate shared schemas through
the integration owner. Preserve provenance and human-readable outputs.

## F: Evaluate PyTorch checkpoints independently with auditable reports

GitHub: [#16](https://github.com/Json-To-String/JPAstro/issues/16). Native subissue of #8.

Own standalone evaluation and reports. Reload a checkpoint in a fresh CPU process without retraining, notebook state or pretrained-weight download. Verify dataset/policy compatibility and refuse forbidden training/validation overlaps with evaluation cohorts. Save record/group IDs, truth and class probabilities plus JSON metrics and a Markdown report/figures. Generate training/validation loss-versus-epoch plots from saved history; include
final test cross-entropy loss, class counts/names, confusion matrix, accuracy,
per-class precision/recall/F1 and ROC curve/AUC from positive-class probabilities.
Do not use per-epoch test results. Optional squared probability error is binary
Brier score, not MSE on hard classes. Keep PR reporting and define one-class
behavior. Record exploratory versus final evaluation provenance.

Membership classification is the deliverable; dwarf labels/definitions/metrics are not completion gates. Add bounded-batch scoring for unlabeled catalogs using the same checkpoint/preprocessing path: export original IDs/coordinates, membership score, thresholded class and checkpoint identity, with explicit rejection/failure reports. Missing truth disables metrics, not scoring. Never label filtered inputs as background. Keep threshold selection on validation data and report calibration limitations. Export candidate catalogs for astronomer-led selection. Other-cluster validation remains optional transfer testing; preserve CPU evaluation of CUDA checkpoints when hardware is available.

Dependencies: C/E artifact contracts and E checkpoint; approved evaluation protocol.

Handoff: owned files, agreed API/artifact contracts, design revision, validation
commands/results, and remaining assumptions. Coordinate shared schemas through
the integration owner. Preserve provenance and human-readable outputs.

## G: Integrate and verify the complete PyTorch research workflow

GitHub: [#17](https://github.com/Json-To-String/JPAstro/issues/17). Native subissue of #8.

Own CLI integration, thin notebook wrapper, CI and workflow documentation. Independently runnable stages must compose as download -> fixed red-pixel filter
-> independent-group train/validation/test split -> training-only rotations ->
config-driven fit -> evaluation. Installation and integration are supporting
work. Verify optional early stopping, graceful-quit resume, best weights and
fresh-process loss/metric reports. Offline synthetic smoke test covers acquisition fixtures, manifest/audit, augmentation, short training and fresh-process evaluation; negative leakage tests are mandatory. Separately record live acquisition and real-data acceptance with frozen data/splits and preserved artifacts. Human review of science policies and leakage report required before parent closure. Coordinate shared schemas and agent handoffs.

Root README must provide a tested copyable CPU walkthrough and CUDA verification, data prerequisites, existing-folder reuse, example brightness config (including disabling the cut and missing-value behavior), supported ResNet18 initialization, rejection reports, frozen splits, rotations, training, early stopping, quit/resume, evaluation and unlabeled bulk scoring. Explain score/catalog columns and membership limitations; astronomers perform downstream dwarf selection. A second architecture, registry and dwarf evaluation are not release requirements. Conditional real CUDA acceptance is separate from CPU CI; future commands stay labeled planned until implemented.

Dependencies: A–F; approved scientific policies.

Handoff: owned files, agreed API/artifact contracts, design revision, validation
commands/results, and remaining assumptions. Coordinate shared schemas through
the integration owner. Preserve provenance and human-readable outputs.

