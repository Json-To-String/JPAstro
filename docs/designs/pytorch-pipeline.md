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
| Split/validate | `splits.py` | Identity evidence, validated images, frozen split policy | Frozen membership, leakage audit, dataset fingerprint |
| Load/augment | `data.py` | Frozen manifest, transform config | Tensor/label batches; transform specification |
| Train | `train.py`, `model.py` | Training/validation membership, model/run config | Best and resumable checkpoints, history, TensorBoard logs |
| Evaluate | `evaluate.py` | Checkpoint, explicit evaluation membership | Predictions, metrics, figures, human report |

Download can run before final splitting so image-content duplicates and failed
acquisitions are resolved before a dataset is frozen. Preparation must preserve
the full original selection trail. Redownloading or changing exclusions after
freezing creates a new dataset version and requires a fresh audit.

Proposed commands: `python -m jpastro prepare`, `download`, `split`, `train`, and
`evaluate`. Each accepts explicit paths/configuration and has actionable failures.
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

## Choices to discuss before implementation

| Question | Proposed direction | Strength | Weakness / decision needed |
| --- | --- | --- | --- |
| Observational unit | Galaxy identity plus an explicit cutout-overlap rule | Handles aliases and repeated observations | Matching radius, angular footprint, neighboring galaxies and transitive groups need scientific agreement |
| Generalization claim | Explicitly define whether evaluation means new objects in Perseus or new sky regions | Makes the split answer the research question | Random grouped splits do not establish spatial/domain generalization; spatial blocks may reduce sample size and class balance |
| Labels | Versioned PCC mapping and spectroscopic policy with conflict reporting | Makes target construction inspectable | Merging systems, stars, uncertain members, redshift quality and boundary inclusivity need decisions |
| Model | One explicit baseline, with architecture/initialization in config | Limits uncontrolled changes | Pretrained ResNet18 is a different experiment from the historical ResNet50; comparison must acknowledge that |
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
| [#12](https://github.com/Json-To-String/JPAstro/issues/12) B Acquisition | `download.py`, acquisition fixtures | Manifest contract; A for live run | Empty-directory download; validated responses; retries/timeouts; atomic writes; resume by expected files/checksums; paths independent of cwd |
| [#13](https://github.com/Json-To-String/JPAstro/issues/13) C Scientific data preparation and isolation | `prepare.py`, `splits.py`, policies, manifest audit | Scientific decisions | Explicit exclusions/counts; catalog aliases, duplicates, overlapping cutouts, conflicting labels, transitive matches and external overlap tested; deterministic persisted splits |
| [#14](https://github.com/Json-To-String/JPAstro/issues/14) D PyTorch datasets and augmentation | `data.py`, transforms | [#13](https://github.com/Json-To-String/JPAstro/issues/13) C contracts | RGB CHW tensors and integer labels; training-only randomness; deterministic evaluation; no held-out statistics |
| [#15](https://github.com/Json-To-String/JPAstro/issues/15) E Training and checkpoints | `model.py`, `train.py`, TensorBoard | A, C, D | Short CPU run; weighted loss aggregation; best/last checkpoint; config/provenance; reproducible initialization; resumable state |
| [#16](https://github.com/Json-To-String/JPAstro/issues/16) F Independent evaluation | `evaluate.py`, reporting | C/E contracts; E checkpoint | Fresh-process CPU reload; overlap refusal; per-record probabilities; class metrics/ROC/PR/confusion; defined behavior for one-class cohorts; human-readable report |
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
