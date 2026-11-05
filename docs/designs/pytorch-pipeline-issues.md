# PyTorch pipeline subissue drafts

Parent: [#8 Convert to Pytorch](https://github.com/Json-To-String/JPAstro/issues/8).
Design: [pytorch-pipeline.md](pytorch-pipeline.md).

These are planning contracts pending scientific design discussion. Each issue
must be attached as a native GitHub subissue of #8 when published. PRs target
`reproducibility`. Do not dispatch implementation agents with unresolved science
choices. A checklist link alone is not a native subissue relationship.

## A: Make the PyTorch pipeline installable and reproducible

GitHub: [#11](https://github.com/Json-To-String/JPAstro/issues/11). Native subissue of #8.

Own pyproject.toml, dependency locking, package layout setup and install documentation. Deliver a clean CPU install, documented GPU setup, all direct dependencies, and a supported workflow requiring no TensorFlow. Resolve missing license/package metadata and SciServer acquisition setup. Coordinate package layout with integration.

Dependencies: Manifest/package layout agreement.

Handoff: owned files, agreed API/artifact contracts, design revision, validation
commands/results, and remaining assumptions. Coordinate shared schemas through
the integration owner. Preserve provenance and human-readable outputs.

## B: Implement resumable and validated image acquisition

GitHub: [#12](https://github.com/Json-To-String/JPAstro/issues/12). Native subissue of #8.

Own acquisition module and HTTP fixtures. Fix the standalone downloader tuple crash and notebook fresh-download path error. Download from an empty directory, validate HTTP status and decoded images, apply timeouts/bounded retries and atomic writes, resume by expected files/checksums rather than directory counts, preserve source/query/release/cutout provenance. Paths must work independently of cwd. Demonstrate both fresh and cached runs.

Dependencies: Manifest contract; A for live execution.

Handoff: owned files, agreed API/artifact contracts, design revision, validation
commands/results, and remaining assumptions. Coordinate shared schemas through
the integration owner. Preserve provenance and human-readable outputs.

## C: Prepare auditable catalogs and enforce leakage-free partitions

GitHub: [#13](https://github.com/Json-To-String/JPAstro/issues/13). Native subissue of #8.

Own catalog preparation, identity grouping, versioned label policies, split manifests and leakage audits. Preserve raw sources; report unknown/conflicting labels and all exclusions/counts. Resolve aliases across catalogs, repeated observations, content duplicates, transitive matches and spatial cutout overlap under an agreed scientific policy BEFORE splitting. Persist deterministic group-disjoint partitions and fingerprints. Check external cohorts too. Fit balancing/statistics only on training data. Negative tests must reject known leakage. Do not invent matching tolerances or scientific labels.

Dependencies: Agreement on observational unit, spatial independence, labels and evaluation population; schema contract.

Handoff: owned files, agreed API/artifact contracts, design revision, validation
commands/results, and remaining assumptions. Coordinate shared schemas through
the integration owner. Preserve provenance and human-readable outputs.

## D: Implement PyTorch datasets and training-only augmentation

GitHub: [#14](https://github.com/Json-To-String/JPAstro/issues/14). Native subissue of #8.

Own Dataset/DataLoader and torchvision transform code. Consume frozen manifests; produce RGB CHW tensors and integer class labels. Training-only stochastic augmentation; deterministic validation/test transforms. Derived samples retain identity groups; no pre-split augmentation or held-out fitted statistics. Record transforms and seed worker/sampler behavior. Explain differences from historical rotation policy.

Dependencies: C manifest/schema and scientific augmentation decisions.

Handoff: owned files, agreed API/artifact contracts, design revision, validation
commands/results, and remaining assumptions. Coordinate shared schemas through
the integration owner. Preserve provenance and human-readable outputs.

## E: Implement reproducible PyTorch training and checkpointing

GitHub: [#15](https://github.com/Json-To-String/JPAstro/issues/15). Native subissue of #8.

Own model/training modules and TensorBoard integration. Run short CPU training with explicit architecture/initialization config, seeded initialization and sample-weighted loss aggregation. Save best and resumable last checkpoints with model/optimizer/RNG state, label mapping, preprocessing, partition hashes, selection metric and provenance. Use validation for model selection; never use final test outcomes for tuning. Include machine-readable history beside TensorBoard. Make architecture changes from the historical ResNet50 explicit.

Dependencies: A, C, D; agreed checkpoint contract and baseline policy.

Handoff: owned files, agreed API/artifact contracts, design revision, validation
commands/results, and remaining assumptions. Coordinate shared schemas through
the integration owner. Preserve provenance and human-readable outputs.

## F: Evaluate PyTorch checkpoints independently with auditable reports

GitHub: [#16](https://github.com/Json-To-String/JPAstro/issues/16). Native subissue of #8.

Own standalone evaluation and reports. Reload a checkpoint in a fresh CPU process without retraining, notebook state or pretrained-weight download. Verify dataset/policy compatibility and refuse forbidden training/validation overlaps with evaluation cohorts. Save record/group IDs, truth and class probabilities plus JSON metrics and a Markdown report/figures. Report class counts, confusion matrix, class metrics, ROC/PR with defined one-class behavior. Record exploratory versus final evaluation provenance.

Dependencies: C/E artifact contracts and E checkpoint; approved evaluation protocol.

Handoff: owned files, agreed API/artifact contracts, design revision, validation
commands/results, and remaining assumptions. Coordinate shared schemas through
the integration owner. Preserve provenance and human-readable outputs.

## G: Integrate and verify the complete PyTorch research workflow

GitHub: [#17](https://github.com/Json-To-String/JPAstro/issues/17). Native subissue of #8.

Own CLI integration, thin notebook wrapper, CI and workflow documentation. Independently runnable prepare/download/split/train/evaluate stages must compose. Offline synthetic smoke test covers acquisition fixtures, manifest/audit, augmentation, short training and fresh-process evaluation; negative leakage tests are mandatory. Separately record live acquisition and real-data acceptance with frozen data/splits and preserved artifacts. Human review of science policies and leakage report required before parent closure. Coordinate shared schemas and agent handoffs.

Dependencies: A–F; approved scientific policies.

Handoff: owned files, agreed API/artifact contracts, design revision, validation
commands/results, and remaining assumptions. Coordinate shared schemas through
the integration owner. Preserve provenance and human-readable outputs.

