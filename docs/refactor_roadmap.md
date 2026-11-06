# Refactor Roadmap

This roadmap keeps the Captury/Motive comparison code maintainable while the
scientific workflow is still evolving. Each phase must start by adding or
strengthening focused tests, then refactor only the targeted ownership boundary,
then run a validation agent before commit.

## Validation Rule

For every phase:

1. Add or update tests that describe the current expected behavior.
2. Make the smallest refactor that improves ownership, naming or duplication.
3. Run the relevant unit tests, `py_compile` and formatting checks.
4. Ask an agent to review the phase for regressions, unclear CLI semantics and
   missing tests.
5. Document user-visible CLI or GUI changes in `README.md`.

## Repo-Wide Refactor Order

The current largest maintenance risks are concentrated in:

- `captury_biobuddy_gui.py`: GUI state, subprocess execution, auto-analysis,
  graph/viewer refresh and BioBuddy hooks are still tightly coupled.
- `compare_p6_motive_captury.py`: trial orchestration mixes discovery, model
  building, alignment, metrics, reports, caching, visualization and IK.
- `bvh_c3d_biobuddy_pyorerun_compare.py`: historical BVH/FBX processing still
  owns reusable C3D, alignment, model-source and pyorerun helpers.

The safest global order is:

1. Shared C3D/source/label/alignment contracts.
2. Metrics IO and aggregation contracts.
3. GUI graph/viewer data contracts.
4. GUI subprocess runner and state transitions.
5. BioBuddy model creation and IK service boundary.
6. P6 trial orchestration dataclasses and step functions.
7. Historical BVH/FBX source-run extraction.

## Phase 1 - Source-Explicit C3D Offset Diagnostic

Status: done.

Scope:

- `plot_c3d_initial_offset.py`
- `tests/test_plot_c3d_initial_offset.py`
- `README.md`

Purpose:

- Treat Motive and Captury independently in the diagnostic CLI.
- Make root-translation subtraction source-specific.
- Make point transforms source-specific.
- Preserve the old global `--subtract-root-offsets` flag only as hidden
  compatibility.

Validation:

- Unit tests cover independent source preparation and legacy flag behavior.
- The documented P6 command writes a diagnostic figure.

## Phase 2 - Shared Source Preparation Module

Status: done.

Scope:

- `c3d_source_preparation.py`
- `plot_c3d_initial_offset.py`
- `tests/test_c3d_source_preparation.py`
- `tests/test_plot_c3d_initial_offset.py`

Purpose:

- Extract source preparation helpers from `plot_c3d_initial_offset.py` into a
  small shared module.
- Reuse that module from plotting diagnostics and any future lightweight C3D
  inspection scripts.
- Establish the first repo-wide pattern: small scientific/data utilities are
  tested outside CLI and GUI code before larger orchestration refactors.

Tests first:

- Move the existing source-preparation tests to the new module tests.
- Add one CLI integration test that verifies parsed Motive/Captury options map
  to the expected preparation configs.

## Phase 3 - C3D, Labels, Units and Alignment Contracts

Status: started.

Candidate scope:

- `bvh_c3d_biobuddy_pyorerun_compare.py`
- `compare_p6_motive_captury.py`
- `gui_trial_viewer.py`
- new modules such as `c3d_io.py`, `mocap_labels.py`, `alignment.py`

Purpose:

- Centralize C3D point splitting, angle-channel filtering, duplicate label
  handling, unit conversion, Kabsch/rigid alignment and source colors/names.
- Prevent GUI/viewer/comparison scripts from reimplementing marker filtering or
  coordinate preparation differently.

Completed in this phase:

- Extracted shared marker-label helpers to `mocap_labels.py`.
- Reused these helpers from the Tk trial viewer, BioBuddy IK CLI and P6
  comparison workflow.
- Fixed duplicate display-label lookup so a unique clean label is not indexed
  twice.
- Extracted shared C3D point-unit conversion helpers to `mocap_units.py`.
- Reused unit conversion from the C3D viewer, capture-system comparison and
  historical BVH/FBX/C3D pipeline.
- Extracted shared row-vector rigid-alignment helpers to `mocap_alignment.py`.
- Reused Kabsch/transform composition from the Tk viewer, P6 comparison and
  capture-system comparison.

Completed in this phase:

- Extracted C3D point-channel classification and angle filtering to
  `c3d_point_channels.py`.
- Reused that contract from the historical BVH/FBX pipeline, capture-system
  comparison and angle-figure CLI.
- Added synthetic C3D tests covering explicit angle metadata, known Captury
  labels, extra labels and label/description regex matching.

Tests first:

- Add tests for angle-channel filtering, duplicate labels, `Skeleton_001_`
  prefix stripping, unit conversion to millimetres and Kabsch rows.
- Add an append/enriched C3D regression test that does not require a real GUI.

## Phase 4 - Metrics IO and Aggregation Contracts

Status: planned.

Candidate scope:

- `model_comparison_metrics.py`
- `compare_p6_motive_captury.py`
- `gui_graphs.py`

Purpose:

- Centralize `write_rows`, compact `.npz` table/time-series writers and
  `all_*` aggregation behavior.
- Keep fast `.npz` time-series readers separate from CSV summaries.
- Define explicit schemas for joint centres, marker correspondences, segments,
  dimensions and kinematics.

Tests first:

- Add empty/non-empty row writer tests.
- Add `.npz` roundtrip tests for each schema consumed by the GUI.
- Add aggregation tests for missing CSVs and mixed trials/participants.

## Phase 5 - GUI Graph and Viewer Data Contracts

Status: planned.

Candidate scope:

- `gui_graphs.py`
- `gui_run_report.py`
- `gui_trial_viewer.py`
- `captury_biobuddy_gui.py`

Purpose:

- Keep graph payload selection pure and independent of Tk widgets.
- Keep viewer layers, CoR chains and marker toggles synchronized from typed
  data payloads instead of direct GUI state.

Tests first:

- Add plotting-data selection tests without opening windows.
- Add fixture tests for missing `.npz`, empty CSV and source-reference
  combinations including BioBuddy fallback.

## Phase 6 - GUI Subprocess Runner and Option Semantics

Status: planned.

Candidate scope:

- `captury_biobuddy_gui.py`
- `gui_commands.py`
- new module such as `gui_runner.py`

Purpose:

- Keep scientific choices in CLI commands, not in Tk callbacks.
- Extract subprocess lifecycle, output queue draining, running-state buttons,
  pending auto-analysis and post-run hooks.
- Centralize display labels versus CLI values for options such as root
  translation policy, model source, segment-frame rotations and axis conversion.

Tests first:

- Add command-builder tests for Captury-specific and Motive-specific options.
- Add fake `Popen` runner tests for code return, stdout/stderr streaming,
  stop/terminate, button disabled/enabled and pending auto-analysis.

## Phase 7 - BioBuddy Model and IK Service Boundary

Status: planned.

Candidate scope:

- `create_biobuddy_c3d_model.py`
- `run_biobuddy_c3d_ik.py`
- `compare_p6_motive_captury.py`
- new module such as `biobuddy_c3d_service.py`

Purpose:

- Make generated BioBuddy models a stable third source for dimensions, centres,
  segment rotations and kinematics.
- Separate model creation, static nonlinear TRF IK and batch IK orchestration.
- Avoid the small IK script importing helpers from the historical mega-script.

Tests first:

- Add command-level tests that can run without BioBuddy internals by mocking the
  external model creation and IK calls.
- Add missing-marker and marker-prefix tests for Motive 57.
- Add one minimal integration test when the BioBuddy API is available.

## Phase 8 - P6 Trial Orchestration Context

Status: planned.

Candidate scope:

- `compare_p6_motive_captury.py`

Purpose:

- Split trial comparison into explicit steps: source discovery, model builds,
  static alignment, temporal cut/contact detection, metrics, outputs and report.
- Introduce dataclasses such as `TrialComparisonContext` and `TrialArtifacts`.

Tests first:

- Add golden mini-report tests with heavy helpers mocked.
- Add cache hit/miss tests preserving report keys and output paths.
- Add `--run-biobuddy-ik-batch` orchestration tests without running real IK.

## Phase 9 - Historical BVH/FBX Source Runs

Status: planned.

Candidate scope:

- `bvh_c3d_biobuddy_pyorerun_compare.py`

Purpose:

- Extract `process_model_source("bvh"|"fbx")` and a `SourceRunArtifacts`
  dataclass to reduce BVH/FBX duplication in `main`.
- Keep pyorerun animation, root policy reports and output filenames stable.

Tests first:

- Add source-run tests with model extraction mocked.
- Add expected output path/report tests for BVH-only, FBX-only and both-source
  runs.


## Available-source selection and viewer delivery — 2026-11-06

The GUI source checkboxes and CLI `--sources` share per-trial availability and
static-calibration orchestration. Paired recordings use the established paired
processor; available-source results retain native coordinates and explicit
registration limits. Canonical CoR edges and source-aware viewer refresh support
partial recordings. Compatible saved BioBuddy IK can be reused without invoking
the solver. No joint-angle conventions, templates or solver bounds change here.

Tests preceded extraction (18 tests: 3 failures, 15 errors). Independent review
required canonical chain edges and unconditional publication of cached BioBuddy
IK metadata in paired reports; both have regression coverage. The isolated branch
passes 380 unittest cases (3 desktop tests skipped in the headless suite),
`py_compile` and `git diff --check`. Actual Tk checks exercise source toggles,
empty selection, command generation, result refresh and missing-source reporting.
A P16 squat CLI smoke completes with Motive available and BioBuddy explicitly
unavailable when no compatible IK cache exists; data and outputs remain external
to the commit. Viewer navigation is also checked with desktop tests enabled.
