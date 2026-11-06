# captury_models

Small workspace for comparing Captury BVH/FBX skeleton exports with C3D marker data, generating biorbd `bioMod` models through BioBuddy, and checking whether C3D markers are stable in local segment frames.

## Contents

- `bvh_c3d_biobuddy_pyorerun_compare.py`: main BVH/FBX to `bioMod` pipeline.
- `captury_biobuddy_gui.py`: graphical launcher for the pipeline options.
- `gui_commands.py`: pure CLI command builders used by the GUI.
- `gui_graphs.py`: embedded Matplotlib graph configuration and drawing helpers.
- `gui_trial_viewer.py`: lightweight Tk C3D marker/CoR preview used inside the GUI.
- `c3d_source_preparation.py`: shared source-specific C3D point preparation helpers used by diagnostics and future comparison/viewer refactors.
- `mocap_labels.py`: shared marker-label helpers for Motive prefixes, duplicate display names, joint-centre labels and IK marker matching.
- `mocap_units.py`: shared C3D point-unit conversion helpers for millimetres/metres.
- `mocap_alignment.py`: shared row-vector rigid-alignment helpers for Kabsch, transform application and transform composition.
- `create_biobuddy_c3d_model.py`: CLI wrapper for BioBuddy's C3D-folder model creation workflow.
- `c3d_trial_viewer.py`: lightweight PySide/QPainter C3D marker viewer used from the trial-cutting tab.
- `compare_capture_systems.py`: compare Motive marker-based C3D trials against Captury markerless C3D trials.
- `compare_p6_motive_captury.py`: Captury/Motive model-centre comparison and C3D enrichment workflow for the `captury/` + `squelettes/` trial layout.
- `model_comparison_metrics.py`: agreement metrics used by the comparison script.
- `motive_captury_landmark_map.json`: editable Motive/Captury anatomical landmark correspondence map.
- `plot_c3d_initial_offset.py`: independent Motive/Captury C3D marker-cloud diagnostic for inspecting raw offsets before model registration.
- `plot_bvh_c3d_angle_comparisons.py`: optional plotting helper for BVH q versus C3D angle channels.
- `kinematic_conventions.json`: versioned registry of source, segment, joint and ISB D1-D6 convention statuses.
- `kinematic_conventions.py`: registry validation, final-comparison blockers and reproducibility-manifest helpers.
- `isb_segment_audit.py`: reproducible ISB D1-D3 segment-frame audit and BioBuddy Motive 57 template introspection.
- `isb_segment_frames.json`: versioned Wu 2002/2005 ISB target-frame registry used by the D1-D3 audit.
- `isb_joint_kinematics.json`: versioned joint sequences, signs and D4-D6 targets used by the matrix-first joint audit.
- `joint_kinematics.py`: corrected parent-child rotations, Euler/Cardan extraction, singularity flags and proximal-frame D5 translations.
- `isb_compliance_report.py`: normalized D1-D6 report and versioned curve-reproducibility manifest.
- `gui_isb_report.py`: presentation-only filtering and table formatting for the GUI ISB report.
- `docs/refactor_roadmap.md`: staged refactor plan with the test-first and agent-validation rule for each phase.
- `docs/scientific_kinematics_roadmap.md`: scientific work plan for harmonizing Captury, Motive and BioBuddy kinematics, including the ISB D1-D6 audit gates.
- `environment_bvh_c3d_biobuddy.yml`: conda environment definition.
- `data/unknown.bvh`, `data/unknown.fbx`, `data/unknown.c3d`: expected local Captury input files. They are ignored by git because they are data files.

Generated outputs are written to `out_biobuddy_bvh_c3d/` and are intentionally ignored by git.

## Environment

Create or update the conda environment:

```bash
conda env create -f environment_bvh_c3d_biobuddy.yml
conda activate captury_biobuddy
```

If the environment already exists:

```bash
conda env update -n captury_biobuddy -f environment_bvh_c3d_biobuddy.yml --prune
conda activate captury_biobuddy
```

The environment installs BioBuddy from:

```text
biobuddy @ git+https://github.com/mickaelbegon/biobuddy.git@codex/add-model-editor-gui
```

This corresponds to the BioBuddy branch:

```text
https://github.com/mickaelbegon/biobuddy/tree/codex/add-model-editor-gui
```

## C3D Point Classification

The C3D contains both marker trajectories and angle-like point channels. The pipeline excludes detected angle channels from the marker cloud used for animation and local marker placement. In addition to labels/descriptions matching the angle regex, these Captury joint labels are treated as angles by default:

```text
RHip LHip RKne LKne RAnk LAnk RSho LSho RElb LElb RWri LWri
```

During animation, the marker overlay contains the C3D marker points after this angle filtering plus the generated joint centres (`BVHJC_*` or `FBXJC_*`). Angle point channels are never sent to pyorerun as markers.

## Run

### Select available comparison sources

In **Données → Sources à inclure si disponibles**, select Captury, Motive and/or
BioBuddy. The same workflow uses whichever selected sources are available for
each trial; a missing Captury recording no longer hides Motive-only trials.
The CLI equivalent is `--sources motive biobuddy` (default: all three).
Reports identify requested, used and missing sources and explain unavailable data.

BioBuddy requires its model and Motive C3D marker recording, independently of
whether the Motive skeleton is selected. Lightweight analyses reuse compatible
saved IK; enable the existing BioBuddy IK batch option to compute missing IK.
Native q curves stay separate by source. Without paired capture registration,
Captury/BioBuddy spatial overlays and paired metrics remain unavailable.

The available-source adapter freezes Motive's existing marker-proxy alignment
from Static and leaves BioBuddy in its C3D fitting frame. Adding Captury later
uses the existing paired static registration in the same workflow. This feature
does not establish anatomical equivalence or change joint-angle conventions.
When no capture pair is available, disable the existing 180-degree segment-frame
rotation and ZXY reexpression options; this adapter displays native coordinates.


To launch the graphical interface:

```bash
conda activate captury_biobuddy
python captury_biobuddy_gui.py
```

For quick P6 debugging, launch the GUI with the local P6 preset already loaded:

```bash
conda run -n captury_biobuddy python captury_biobuddy_gui.py --p6-debug
```

Run the lightweight regression tests:

```bash
python -m unittest discover -s tests -v
```

The interface is a graphical launcher around the existing command line scripts. The scientific processing remains in the CLI scripts; the GUI builds the equivalent command with `sys.executable`, copies it to the clipboard, launches it in a background `subprocess.Popen`, streams stdout/stderr into a log popup, and lets the running process be stopped.

The small `Commande` button in the bottom-left corner opens a compact command popup. It can target three workflows:

- `Analyse Captury/Motive`: runs `compare_p6_motive_captury.py`.
- `Pipeline BVH/FBX/C3D`: runs `bvh_c3d_biobuddy_pyorerun_compare.py`.
- `Comparaison générique`: runs `compare_capture_systems.py`.

The GUI tabs are organized for the Captury/Motive analysis:

- `Données`: choose the flattened `Captury/` + `Motive/` data root, output folder, static trial, model source and model-to-C3D axis conversion. It also exposes the two opt-in Captury segment-frame corrections and the editable `captury_q_landmarks.json` alias map. The detected files are inventoried in a table, and the global trial menu in the top-right corner applies to every tab. The local P6 debug preset remains available from the CLI with `--p6-debug`.
- On exit, the GUI remembers the data root, output folder, BioBuddy model and selected trial in `~/.config/captury_models/gui_session.json`. On reopening, cached `joint_centre_timeseries.npz` results make the BioBuddy CoR checkbox available again when that layer exists; the checkbox remains unchecked until the user selects it.
- `BioBuddy`: create a `bioMod` directly from a folder of calibration C3D files with BioBuddy's `create_model_from_c3d_folder`, including the Motive 57 preset.
- `Occlusions`: analyze missing Motive marker trajectories in a sortable table with clean marker names.
- `Découpage`: estimate movement start/end and ground contacts from foot-marker kinematics, synchronize the Captury clock to Motive, normalize the selected phase to 0-100 %, and open the selected trial in the lightweight 3D C3D viewer.
- `Dimensions`: compare model dimensions with an embedded graph and hierarchical metric/component selectors.
- `Segments`: compare segment orientation matrices against a selectable reference model (`BioBuddy` by default, with documented fallback to Motive when no BioBuddy q time series is available).
- `Centres`: compare model joint-centre positions after alignment, including time curves for a selected joint.
- `Marqueurs`: compare reasonable Motive/Captury skin-marker correspondences.
- `Cinématiques`: compare available model q/angle channels, inspect DoF waveforms over time and optionally run batch IK.
- `Visualisation`: launch the enriched C3D/Rerun visualization or run headless.
- `Critique`: filter the D1-D6 table by source, deviation, segment/articulation target and status, inspect confidence/blockers, then review the sensitive algorithms and assumptions before interpreting distances or angles.
- `Avancé`: inspect the Python executable, script paths and compatibility options.

The metric tabs contain embedded Matplotlib graphs instead of PNG previews. Each graph panel has a hierarchical selector (`trial -> metric -> component`) so a metric can be plotted globally or narrowed to a specific marker, segment, joint, landmark or q component. In the `Segments` tab, selecting a segment displays global and X/Y/Z rotation-deviation curves over time from `segment_rotation_timeseries.npz`; selecting a broader metric displays absolute-deviation boxplots by segment/source. Segment deviations are computed from `R_ref.T @ R_source` with the rotation-vector log map, then displayed in degrees. In the `Centres` tab, selecting a metric displays one time-distribution boxplot per joint centre from `joint_centre_timeseries.npz`; selecting a single joint displays its error curves over time, with Euclidean distance and absolute X/Y/Z components. In the `Cinématiques` tab, selecting one DoF displays its Motive, Captury and difference waveforms over time; selecting one Captury C3D angle channel displays the exported Captury C3D angle waveform. Selecting a metric such as `bias_rad`, `mae_rad`, `rmse_rad` or `c3d_mean_deg` displays one boxplot per DoF/channel. Rotation metrics and rotation waveforms are converted to degrees for display, while the output files keep the raw radian values when they come from model q.

The model-centre workflow automatically handles the current P6 conventions by default: Captury BVH/FBX is treated as millimetres and its C3D remains in its native `+Y`-up laboratory frame; Motive BVH/FBX is treated as centimetres and its C3D is `+Z` up. With `--model-to-c3d-axis auto`, Captury model centres are therefore compared first with Captury C3D in `+Y`, while Motive model centres are compared with Motive C3D in `+Z`. Captury C3D is converted to the common `+Z` comparison frame only for Captury/Motive alignment and cross-system metrics. Before writing `CAPJC_*` and `MOTJC_*` channels into enriched C3D copies, the Motive model chain is also yaw/translation-aligned to the Motive C3D marker cloud from 57-marker anatomical proxies, with a horizontal PCA fallback when too few proxies are available. The bottom-left `Log` button opens the live process log when needed.

Each full `compare_p6_motive_captury.py` comparison batch now writes `provenance_manifest.json` before scientific processing. It records the exact selected C3D/BVH/FBX/bioMod inputs, SHA-256 hashes, command, Python executable, dependency versions, matrix convention and unresolved convention blockers. It also records the generated `isb_d1_d3_audit.json` and `isb_d1_d3_audit.npz` artifacts. The specialized `--occlusions-only` path intentionally skips BioBuddy import and D1-D3 audit generation. A `comparison_readiness.status` of `diagnostic_only` means that the figures remain useful for diagnosis but must not be interpreted as final biomechanical agreement. The manifest does not yet recover every C3D sampling rate or proprietary exporter convention; these remain explicit G0/G2 tasks in the scientific roadmap.

The D1-D3 audit inventories every source and segment. `isb_segment_frames.json` versions the selected Wu et al. (2002/2005) origins, axes, landmarks, option and DOI for pelvis, thorax, thigh, shank, foot, upper arm, forearm and the global-wrist hand surrogate. For the active BioBuddy Motive 57 runtime, the audit extracts the declared origin, raw axes, retained orthogonalization axis, functional SCoRE/SARA trials and static fallbacks, and records the exact imported template path and SHA-256 hash. Its symbolic comparison can demonstrate D2/D3 differences but never declares D1 conformity from marker names alone. Captury and Motive exporter frames remain `inconnu` when their construction is proprietary.

Creating a Motive 57 model now writes two SHA-bound evidence sidecars beside the model: `<model>.roundtrip.json` verifies all serialized parent-local `RT` matrices against the BioBuddy real model, and `<model>.isb_static.json` records static angular deviation, origin deviation in millimetres, determinants and orthogonality. On P6, 7 of 15 BioBuddy segment frames are numerically evaluable without undocumented substitutions. Thorax is withheld because Wu requires T8 while Motive 57 provides TV7; head has no selected target; both hands are withheld because HM2 is not silently substituted for the selected third-metacarpal surrogate. Feet and upper arms are withheld because the neutral-ankle and option-2 humerus calibration postures prescribed by Wu are not validated in P6 metadata. These missing evaluations remain explicit and do not mean zero error. When `compare_p6_motive_captury.py` receives the same model through `--biobuddy-biomod`, it embeds matching sidecars in `isb_d1_d3_audit.json`; stale sidecars are rejected by SHA-256.

Each trial also writes `joint_kinematics_d4_d6.json` and `joint_kinematics_d4_d6.npz`. The JSON states proximal/distal segments, applied correction matrices, target sequence, anatomical components, signs, D4-D6 status, singular-frame count, roundtrip error and SHA-256 evidence; the compressed NPZ preserves relative SO(3) matrices, JCS axes, quaternions, rotation magnitudes and available Euler series. Native Captury/Motive relative matrices remain diagnostic because their source-to-anatomical corrections are unknown. With the current P6 BioBuddy static evidence, both hips and knees can be expressed in the target `ZXY` convention as a diagnostic. This is not final ISB conformity because the underlying D1-D3 frames remain diagnostically, rather than independently anatomically, validated. The remaining BioBuddy joints stay unavailable when either segment frame is not justified. Shoulder motion is labelled `thoracohumeral` and D6 deviation; no glenohumeral angle is emitted because the model has no scapula. D5 remains non-applicable until a common joint point is reconstructed independently in proximal and distal frames.

At the end of a full batch, `isb_d1_d6_report.json` and
`isb_d1_d6_table.npz` consolidate D1-D3 segment evidence and per-trial D4-D6
joint evidence. Captury BVH/FBX, Captury C3D angles, Motive BVH/FBX and
BioBuddy remain four distinct sources: an unavailable source produces explicit
blocking rows and never falls back to another model. The report records the raw
status, normalized status, confidence, evidence and blocker codes. The
`Critique` tab renders this table side by side and keeps global D1-D3 rows
visible while filtering a selected trial.

`comparison_reproducibility_manifest.json` fingerprints every JSON/NPZ
scientific artifact exposed by a trial report. It records paths relative to the
output root, SHA-256 hashes, NPZ keys/shapes/dtypes, transformation references
and curve recipes. Table-like NPZ files and hierarchical D4-D6 angle series can
therefore be reloaded without GUI code. The manifest reproduces stored curves;
it does not turn unknown conventions into anatomically harmonized data.

The detected-file tables show the vertical-axis convention used by the GUI: BVH/FBX model files are treated as `+Y modèle`; Captury C3D is displayed as `+Y labo`, whereas Motive C3D is displayed and written as `+Z labo`. The embedded viewer converts the Captury marker layer to the common frame before its Captury-to-Motive rigid overlay, while preserving the raw source file on disk.

The selected trial's Motive and Captury C3D files are loaded as separate marker layers in the right-hand 3D viewer panel whenever both are available. C3D marker coordinates are converted automatically to millimetres from the C3D `POINT:UNITS` field before display, matching the CoR chains written in `joint_centre_timeseries.npz`. Captury C3D angle channels stored in the POINT section, such as `RHip`, `LKne` or labels matching `angle`, are excluded from marker layers and marker comparisons. They are extracted separately as kinematic channels. Marker colors use lighter source-code nuances, while CoR chains use the stronger Captury orange, Motive cyan and BioBuddy green colors. Marker layers and CoR chains have independent checkboxes, so Captury/Motive markers and Captury/Motive/BioBuddy kinematic chains can be toggled separately. Selecting a trial starts a lightweight cached analysis for that movement by default, using no meshes, no figures, no Rerun and no batch IK; this refreshes the metric tables and graphs without launching the heaviest options.

The BioBuddy tab mirrors this CLI command for Motive 57 model creation:

```bash
python create_biobuddy_c3d_model.py \
  /Users/mickaelbegon/Downloads/data/Motive \
  --preset motive_57 \
  --motive-57-mapping-json /Users/mickaelbegon/Downloads/data/Motive/.motive_57_c3d_mapping.json \
  --output /tmp/motive_57.bioMod
```

The wrapper imports `biobuddy.gui.c3d_model_creation` at runtime. If that module
is missing, update the `captury_biobuddy` environment to a BioBuddy branch that
contains `create_model_from_c3d_folder`.

For the Motive 57 preset, the GUI inventories all `.c3d` files in the Motive
folder and stores the selected calibration roles in
`.motive_57_c3d_mapping.json`. The expected roles are:

- `static`: static/anatomical C3D used for segment frames and static virtual
  shoulder centers `LGJC`/`RGJC`.
- `left_hip_score`: left hip SCoRE trial, expected by BioBuddy as
  `*Func_LHip.c3d`.
- `left_knee_sara`: left knee SARA axis trial, expected as `*Func_LKnee.c3d`.
- `left_ankle_score`: left ankle SCoRE trial, expected as `*Func_LAnkle.c3d`.
- `right_hip_score`: right hip SCoRE trial, expected as `*Func_RHip.c3d`.
- `right_knee_sara`: right knee SARA axis trial, expected as `*Func_RKnee.c3d`.
- `right_ankle_score`: right ankle SCoRE trial, expected as `*Func_RAnkle.c3d`.

The JSON lets files such as `P6_LHip.c3d` be selected explicitly even though the
BioBuddy template searches for `*Func_LHip.c3d`. At launch time the wrapper
creates a temporary calibration folder with template-compatible symlinks/copies,
then calls BioBuddy on that prepared folder.

Below the D1-D6 table, the `Critique` tab lists the main assumptions that should
be checked before interpreting results: FBX/BVH-to-C3D registration,
Captury/Motive/BioBuddy model coherence, vertical-axis orientation, unit
scaling and joint-angle extraction.

In the `Découpage` tab, drag horizontally on a contact/movement graph to define the manual phase of interest. The selected time span is shaded on the graph and copied into `Début manuel (s)` / `Fin manuelle (s)`.

The same tab exposes `auto`, `manual`, and `none` temporal synchronization. Motive is always the reference clock. The sign convention is explicit: `captury_corrected_time = captury_original_time + lag_s`. Automatic mode estimates one constant lag from a robust composite of translation-invariant common joint-centre speeds; it is skipped for the Static trial and refused when motion is insufficient, correlation is below 0.5, the correlation gain is below 0.01, the optimum reaches the search boundary, or separated periodic peaks are ambiguous. The estimated lag remains in the report when it is refused. It does not estimate clock drift or deform time, and it leaves samples outside temporal overlap missing rather than repeating endpoint postures.

The C3D viewer is a lightweight PySide/QPainter widget. It uses orthographic projection, drag rotation, wheel zoom, double-click reset, a right-click view menu (`XY`, `YZ`, `XZ`, `Face`, `Dos`, `Côté`), a frame slider, playback, marker-table selection highlighting, a whole-body fit toggle and an RGB triad. Launch it directly with:

```bash
python c3d_trial_viewer.py local_trials/2026-06-30_P6_flat/Motive/P6_Static.c3d
```

For direct command line use:

```bash
python bvh_c3d_biobuddy_pyorerun_compare.py \
  --bvh data/unknown.bvh \
  --fbx data/unknown.fbx \
  --c3d data/unknown.c3d \
  --out-dir out_biobuddy_bvh_c3d
```

To also launch pyorerun animations:

```bash
python bvh_c3d_biobuddy_pyorerun_compare.py \
  --bvh data/unknown.bvh \
  --fbx data/unknown.fbx \
  --c3d data/unknown.c3d \
  --out-dir out_biobuddy_bvh_c3d \
  --animate
```

To display the BVH model, the FBX surface model and filtered C3D markers superposed in one scene:

```bash
python bvh_c3d_biobuddy_pyorerun_compare.py \
  --bvh data/unknown.bvh \
  --fbx data/unknown.fbx \
  --c3d data/unknown.c3d \
  --out-dir out_biobuddy_bvh_c3d \
  --animate-superposed \
  --hide-extremities-in-rerun
```

To compute inverse kinematics from the C3D markers with biorbd nonlinear least squares, add:

```bash
python bvh_c3d_biobuddy_pyorerun_compare.py \
  --bvh data/unknown.bvh \
  --fbx data/unknown.fbx \
  --c3d data/unknown.c3d \
  --out-dir out_biobuddy_bvh_c3d \
  --inverse-kinematics \
  --inverse-kinematics-solver least_squares
```

To use biorbd's marker Kalman reconstruction instead:

```bash
python bvh_c3d_biobuddy_pyorerun_compare.py \
  --bvh data/unknown.bvh \
  --fbx data/unknown.fbx \
  --c3d data/unknown.c3d \
  --out-dir out_biobuddy_bvh_c3d \
  --inverse-kinematics \
  --inverse-kinematics-solver kalman
```

For a quick smoke test, limit inverse kinematics to the first few frames:

```bash
python bvh_c3d_biobuddy_pyorerun_compare.py \
  --bvh data/unknown.bvh \
  --fbx data/unknown.fbx \
  --c3d data/unknown.c3d \
  --out-dir out_biobuddy_bvh_c3d \
  --inverse-kinematics \
  --inverse-kinematics-max-frames 5
```

Useful generated files include:

- `model_from_bvh_biobuddy.bioMod`
- `model_from_fbx_biobuddy.bioMod`
- `meshes/*.ply`
- `bvh_q_biorbd_order.npz`
- `fbx_q_biorbd_order.npz`
- `bvh_c3d_local_markers.csv`
- `fbx_c3d_local_markers.csv`
- `bvh_c3d_marker_error_norm_boxplot.png`
- `fbx_c3d_marker_error_norm_boxplot.png`
- `bvh_fbx_c3d_marker_error_norm_overall_boxplot.png`
- `bvh_animation_markers_no_angles_with_joint_centres.npz`
- `fbx_animation_markers_no_angles_with_joint_centres.npz`
- `bvh_inverse_kinematics_from_c3d_markers.npz`
- `fbx_inverse_kinematics_from_c3d_markers.npz`
- `run_report.json`

## Root Translation Policy

Captury exports may store a static root offset in the skeleton while also storing root position channels in laboratory coordinates. The generic policy remains `--root-offset-mode auto`: it builds both interpretations of the root translation q, with and without subtracting the static root offset, then keeps the better overlay. In the Captury/Motive comparison, Captury explicitly defaults to `--captury-root-offset-mode keep`, while Motive inherits the generic `auto` policy. The single-trial BVH/FBX pipeline scores in native model units. The Captury/Motive P6 pipeline first converts each model into its *own* C3D basis with `--model-to-c3d-axis`, scores automatic interpretations in millimetres against that system's matching C3D marker cloud, and writes the chosen policy in each trial report.

The selected policy is written to:

- `bvh_root_translation_policy.json`
- `fbx_root_translation_policy.json`
- `out_p6_motive_captury_comparison/<trial>/<system>/<source>/<system>_<source>_root_translation_policy.json`

Use `--root-offset-mode subtract` or `--root-offset-mode keep` to force either convention. In the GUI this is the `Offset racine` selector with explicit labels: choose the best C3D overlay automatically, subtract the static root offset, or keep the file root translations. The automatic mode is preferred for debugging because it documents both scores instead of silently assuming one convention.

For the Captury/Motive comparison, the policies can be controlled independently
with `--captury-root-offset-mode` and `--motive-root-offset-mode`. Captury defaults
to `keep`; Motive defaults to the generic `auto` policy. Their selected values
are frozen from the static calibration and reused on dynamic trials.

The default spatial calibration is non-circular: it reserves `Hips`, `Head`,
`LeftShoulder`, and `RightShoulder` for the Static Captury-to-Motive rigid fit,
then excludes them from primary joint-centre metrics. Override the reserved set
by repeating `--alignment-calibration-centre NAME`. The legacy fit on all common
centres is available only as an explicit diagnostic with
`--spatial-alignment-mode legacy_all_centres`.

`plot_c3d_initial_offset.py` is a separate raw-C3D diagnostic and does not use
`--root-offset-mode`. It treats Motive and Captury independently so that offsets
and axis hypotheses can be tested without hiding where a discrepancy comes from.
The operation order is explicit: optional source root-translation subtraction
first, then optional source axis transform. For the current P6 raw Captury C3D
marker cloud, do not subtract the FBX/BVH root offset; test the vertical
conversion with Captury `R(x,+90 deg)` only:

```bash
MPLCONFIGDIR=/tmp/mplconfig python plot_c3d_initial_offset.py \
  --captury-transform rx_plus_90 \
  --output out_c3d_initial_offset/static_initial_offset_captury_rx_plus_90.png
```

When you need to test an offset on Motive only, keep it source-specific:

```bash
MPLCONFIGDIR=/tmp/mplconfig python plot_c3d_initial_offset.py \
  --motive-subtract-root-offset \
  --motive-root-offset-mm X_MM Y_MM Z_MM \
  --captury-transform rx_plus_90 \
  --output out_c3d_initial_offset/static_initial_offset_motive_offset_test.png
```

The symmetrical options are `--motive-transform`, `--captury-transform`,
`--motive-subtract-root-offset` and `--captury-subtract-root-offset`. The hidden
legacy `--subtract-root-offsets` flag still enables both subtractions for old
commands, but new debugging commands should use the per-source flags.

## Generalized Coordinate Units

The exported `*_q_biorbd_order.npz` files are now populated from BioBuddy's `to_q()` output for both BVH and FBX, so they follow the DOF order expected by the generated `bioMod` (`*_transX`, `*_rotZ`, etc.). Translation channels remain in the native length unit of the BVH/FBX file so they match the `RT` offsets written in the `bioMod`. Rotation channels are returned by BioBuddy in radians, then unwrapped per Euler channel before saving and animation. The `.npz` files include `q_units`, and `run_report.json` includes an unwrap summary.

## FBX Mesh

The FBX mesh is handled by the BioBuddy branch `codex/add-model-editor-gui`: the skinned visual mesh is split into per-segment `.ply` files for the FBX `bioMod`. The script also converts those generated `.ply` files to per-segment `.vtp` files before animation, because pyorerun accepts `.stl`/`.vtp` mesh files for rendering surfaces while biorbd keeps reading the `.ply` paths from the `bioMod`. Writing raw `mesh x y z` points into a `bioMod` only provides vertices and typically appears as a line/point cloud in the viewer.

The pyorerun display uses millimetre-scale marker radii by default (`--rerun-marker-radius 15`) and keeps the FBX mesh opaque so the scene is visible immediately in Rerun.

Use `--hide-hands-in-rerun`, `--hide-feet-in-rerun`, or `--hide-extremities-in-rerun` to hide hand/wrist/finger and/or foot/ankle/toe markers and meshes from pyorerun animations without changing the numerical outputs.

The script declares `Y` as the vertical axis in Rerun by default (`--rerun-up-axis y`). Use `--rerun-up-axis z`, `x`, or `none` if the viewer orientation should follow another convention.

## Motive vs Captury Comparison

The local Motive/Captury trial archive can be extracted into `local_trials/`, which is ignored by git:

```bash
mkdir -p local_trials
unzip -oq /Users/mickaelbegon/Downloads/data.zip -d local_trials
```

Run the comparison across all discovered trial pairs:

```bash
python compare_capture_systems.py \
  --data-root local_trials/data \
  --reference-system Motive \
  --test-system Captury \
  --landmark-map motive_captury_landmark_map.json \
  --out-dir out_capture_system_comparison
```

The comparison script is prepared for C3D, BVH and FBX on both systems. The current single-participant layout remains valid. It discovers either flat files:

```text
local_trials/data/Motive/P5_Marche_001.c3d
local_trials/data/Motive/P5_Marche_001.bvh
local_trials/data/Motive/P5_Marche_001.fbx
```

or one folder per trial:

```text
local_trials/data/Captury/P5_Marche_001/unknown.c3d
local_trials/data/Captury/P5_Marche_001/unknown.bvh
local_trials/data/Captury/P5_Marche_001/unknown.fbx
```

For population studies, use one directory per participant, with the same trial naming convention under each system:

```text
local_trials/data/P01/Motive/P01_Marche_001.c3d
local_trials/data/P01/Motive/P01_Marche_001.bvh
local_trials/data/P01/Motive/P01_Marche_001.fbx
local_trials/data/P01/Captury/P01_Marche_001/unknown.c3d
local_trials/data/P01/Captury/P01_Marche_001/unknown.bvh
local_trials/data/P01/Captury/P01_Marche_001/unknown.fbx
local_trials/data/P02/Motive/P02_Marche_001.c3d
local_trials/data/P02/Captury/P02_Marche_001/unknown.c3d
```

The GUI exposes participant and trial filters. From the command line, use repeated regex filters when needed:

```bash
python compare_capture_systems.py \
  --data-root local_trials/data \
  --participant-filter "P0[1-5]" \
  --trial-filter "Marche" \
  --landmark-map motive_captury_landmark_map.json \
  --out-dir out_capture_system_comparison
```

The current numerical comparison derives comparable anatomical landmarks from Motive markers and Captury `Q_*` points, time-normalizes each trial, applies a global rigid alignment by default, and writes raw/aligned landmark errors plus C3D and model-file inventories. Main outputs:

- `out_capture_system_comparison/all_landmark_metrics.csv`
- `out_capture_system_comparison/all_angle_metrics.csv`
- `out_capture_system_comparison/all_model_inventory.csv`
- `out_capture_system_comparison/population_landmark_summary.csv`
- `out_capture_system_comparison/population_angle_summary.csv`
- `out_capture_system_comparison/run_report.json`

For the current sample archive, Motive C3D files contain marker trajectories but no joint-angle POINT channels. Captury C3D files contain markerless `Q_*` points and joint-angle channels listed in `POINT:ANGLES`; therefore the script reports Captury angles in the inventory, but only computes angle agreement when both systems provide matching angle channels.

## Motive/Captury Kinematic Model-Centre Comparison

The original P6 folder can be flattened into a simpler local dataset with only `Captury/` and `Motive/` subfolders:

```bash
python prepare_kinematic_dataset.py \
  --source-root /Users/mickaelbegon/Downloads/2026-06-30_P6 \
  --output-root local_trials/2026-06-30_P6_flat
```

The flattened folder is ignored by git and has this structure:

```text
local_trials/2026-06-30_P6_flat/Captury/Static_P6.bvh
local_trials/2026-06-30_P6_flat/Captury/Static_P6.fbx
local_trials/2026-06-30_P6_flat/Captury/Static_P6.c3d
local_trials/2026-06-30_P6_flat/Motive/P6_Static_Skeleton 001.bvh
local_trials/2026-06-30_P6_flat/Motive/P6_Static.fbx
local_trials/2026-06-30_P6_flat/Motive/P6_Static.c3d
```

List the detected trials:

```bash
python compare_p6_motive_captury.py \
  --data-root local_trials/2026-06-30_P6_flat \
  --list-trials
```

Run the example batch for `Static`, `LKnee` and `Marche_001`:

```bash
python compare_p6_motive_captury.py \
  --data-root local_trials/2026-06-30_P6_flat \
  --trial Static \
  --trial LKnee \
  --trial Marche_001 \
  --joint-filter "Hip|Knee|Ankle|Leg|Foot" \
  --static-trial Static \
  --model-source bvh \
  --no-mesh \
  --out-dir out_p6_motive_captury_comparison
```

Use trial cutting bounds when only part of the trial should be compared. The default `manual` mode uses `--time-start` and `--time-end` when they are provided:

```bash
python compare_p6_motive_captury.py \
  --data-root local_trials/2026-06-30_P6_flat \
  --trial Marche_001 \
  --cut-mode manual \
  --time-start 0.75 \
  --time-end 4.25 \
  --no-mesh \
  --no-figures \
  --out-dir out_p6_motive_captury_cut_check
```

The enriched C3D remains a full-trial visualization copy, while joint-centre metrics, q metrics and `trial_events_contacts.csv` are restricted to the requested time window. The report records the manual and used bounds under `time_window` and `trial_events`.

To use the movement-onset detector instead of manual bounds:

```bash
python compare_p6_motive_captury.py \
  --data-root local_trials/2026-06-30_P6_flat \
  --trial Marche_001 \
  --cut-mode movement \
  --no-mesh \
  --no-figures \
  --out-dir out_p6_motive_captury_detected_cut_check
```

Use `--cut-mode full` to explicitly ignore manual and detected bounds.

Synchronize a dynamic trial automatically and normalize its selected window to
101 points:

```bash
python compare_p6_motive_captury.py \
  --data-root local_trials/2026-06-30_P6_flat \
  --trial Marche_001 \
  --temporal-sync-mode auto \
  --max-lag-s 0.5 \
  --phase-normalization-points 101 \
  --cut-mode movement \
  --no-mesh --no-figures \
  --out-dir out_p6_motive_captury_comparison
```

For an externally measured offset, use `--temporal-sync-mode manual
--manual-lag-s VALUE`. A positive value places Captury samples later on the
Motive timeline. The corrected time vector is used consistently for Captury
model centres, q, segment rotations, C3D angle channels and skin markers;
spatial samples are not modified. `temporal_synchronization.json` records the
method, lag, correlations, peak prominence, lag resolution, normalized residual
RMSE and analyzed window. Automatic synchronization is not applied when the
signal is static, correlation is weak, improvement is negligible, the best
peak lies on the search boundary, or multiple periodic peaks are ambiguous.
`temporal_synchronization_timeseries.npz` stores the inspectable alignment
signal, and `temporal_phase_normalized.npz` stores the selected phase on
0-100 %. `phase_normalized_scientific_timeseries.npz` extends the same phase
window to each available family among joint centres, q, segment rotations and
paired skin markers; unavailable families remain absent instead of being
silently synthesized. Every normalized signal records its finite coverage,
largest interpolated gap, accepted maximum gap and status; long internal gaps
are refused and written as missing values.
`contact_cycles.json` lists foot-contact cycles detected from kinematics; they
remain diagnostic and are explicitly marked as not force-plate validated.
Missing foot-marker groups are marked unavailable, never converted to contact,
and cycles shorter than 0.2 s are rejected.

The script builds BioBuddy/biorbd models for both systems from BVH by default. Use `--model-source fbx` to force FBX. `--model-source auto` now builds and compares both exports when both are available: automatic selection is rejected unless every mapped segment stays below `--bvh-fbx-max-p95-geodesic-deg` (default `5` degrees at p95). This numerical gate prevents silent source substitution; it is not an ISB tolerance or proof of anatomical equivalence. Captury BVH/FBX is treated as millimetres; Motive BVH/FBX is treated as centimetres unless overridden with `--captury-unit-scale-to-m` or `--motive-unit-scale-to-m`.

Audit BVH and FBX while keeping an explicit BVH result:

```bash
python compare_p6_motive_captury.py \
  --data-root local_trials/2026-06-30_P6_flat \
  --out-dir out_p6_motive_captury_debug \
  --trial Static \
  --model-source bvh \
  --audit-bvh-fbx-rotations \
  --bvh-fbx-max-p95-geodesic-deg 5 \
  --no-mesh --no-figures
```

Each system writes `bvh_fbx_rotation_audit.json` plus a compressed `bvh_fbx_rotation_audit.npz` under the trial's `captury/` or `motive/` directory. The JSON records canonical segment coverage, SO(3) quality corrections, per-segment median/p95/max/RMS geodesic deviations and the automatic-selection verdict. On the current P6 Static files, both systems fail the conservative 5-degree gate, so BVH or FBX must be selected explicitly.

The within-system BVH/FBX rotation audit still compares elapsed timestamps without a separate lag estimate because both files are expected to originate from the same export. Cross-system Captury/Motive comparisons use the explicit synchronization policy above before interpolation. No path currently compensates clock drift.

## Captury Coordinate Frames and Optional Q_* Corrections

The current FBX/BVH model convention is `+Y` vertical. `--model-to-c3d-axis
auto` keeps a Captury model in its own Captury C3D `+Y` frame, and maps a
Motive model from `+Y` to its Motive C3D `+Z` frame:

```text
Captury model (+Y) -> Captury C3D (+Y): identity
Motive  model (+Y) -> Motive  C3D (+Z): x'=x, y'=-z, z'=y
Captury C3D (+Y) -> common comparison frame (+Z): x'=x, y'=-z, z'=y
```

The final Captury-to-Motive alignment uses only the last, comparison-stage
conversion. It is deliberately not injected into Captury root-offset scoring
or Captury-only segment-frame calibration. This distinction is recorded in
`run_report.json -> axis_conversion` and `spatial_calibration.json` so an
apparent offset can be traced to the correct stage.

For a manual source-specific diagnostic, use
`--captury-model-to-c3d-axis` and `--motive-model-to-c3d-axis` separately.
The legacy shared `--model-to-c3d-axis` must remain `auto` for a two-system
comparison: applying one manual basis to both systems would incorrectly rotate
either Captury or Motive relative to its own C3D. The GUI exposes the same two
source-specific menus in `Données` and `Modèles`.

The GUI `Données` tab exposes two mutually exclusive, **unchecked by default**
Captury options. They are also available from the CLI:

```bash
# Diagnostic baseline: a single local-axis rename for every Captury segment.
python compare_p6_motive_captury.py ... --captury-rename-axes

# Participant/source-specific static calibration from Captury data only.
python compare_p6_motive_captury.py ... \
  --captury-calibrate-segment-frames \
  --captury-q-landmarks-json captury_q_landmarks.json
```

`--captury-rename-axes` right-multiplies every Captury local frame by the
single matrix `R_y(+90 deg)`, which merely renames the local axes. It uses no
marker or joint-centre data. It is kept as an explicit diagnostic baseline:
when it does not resolve segment-specific discrepancies, that result shows
why a global relabelling is not an anatomical calibration and must not be
interpreted as ISB conformity.

`--captury-calibrate-segment-frames` is the opt-in alternative. From the
selected participant's **Captury static trial only**, it derives the pelvis
target frame from the accepted `Q_Wa#1..#5` aliases and the left/right thigh
target frames from the Captury hip-to-knee model-centre direction plus the
accepted `Q_LK#1/#2` or `Q_RK#1/#2` transverse marker pair. Per-frame
matrices are fitted as `C = R_source.T @ R_target`, robustly averaged on
SO(3), and then applied as `R_target = R_source @ C` on that participant's
dynamic trials. No Motive marker, Motive centre, Motive transform, or
functional centre is used in this calibration. The matrices are frozen in
`spatial_calibration.json`, reused only for the same `bvh` or `fbx` source
kind, and recomputed for each new participant/static trial. Segments without
a reviewed Captury landmark construction remain explicitly uncalibrated.

`captury_q_landmarks.json` is the editable evidence map. In the GUI, it
renames raw duplicates for review, for example `CAP_RASIS (Q_Wa#1)` and
`CAP_LK_medial (Q_LK#1)`, while correspondence files retain the stable raw
occurrence keys (`Q_Wa#1`, `Q_LK#1`, ...). Review this map before enabling
the calibration on a new Captury export because duplicate occurrence order is
an export property, not a universal anatomical guarantee. These frames are
anatomical surrogates for comparison; they do not, on their own, certify ISB
D1-D3 compliance.

BioBuddy exports the generated biorbd segments with `translations xyz` and
`rotations zyx` in the `bioMod`. The saved `q` arrays still expose readable
coordinate names such as `Hips_transX`, then `Hips_rotX`, `Hips_rotY`,
`Hips_rotZ`; always use the `q_names`/generated `bioMod` order rather than
assuming a generic FBX Euler order.

Main outputs:

- `out_p6_motive_captury_comparison/<trial>/<trial>_motive_with_capjc_motjc.c3d`
- `out_p6_motive_captury_comparison/<trial>/joint_centre_metrics.csv`
- `out_p6_motive_captury_comparison/<trial>/kinematics_q_metrics.csv`
- `out_p6_motive_captury_comparison/<trial>/joint_centre_timeseries.npz`
- `out_p6_motive_captury_comparison/<trial>/alignment_calibration_centre_metrics.csv`
- `out_p6_motive_captury_comparison/<trial>/alignment_calibration_centre_timeseries.npz`
- `out_p6_motive_captury_comparison/<trial>/spatial_calibration.json`
- `out_p6_motive_captury_comparison/<trial>/kinematics_q_timeseries.npz`
- `out_p6_motive_captury_comparison/<trial>/captury_c3d_angle_metrics.csv`
- `out_p6_motive_captury_comparison/<trial>/captury_c3d_angle_timeseries.npz`
- `out_p6_motive_captury_comparison/<trial>/captury_c3d_angle_decode.json`
- `out_p6_motive_captury_comparison/<trial>/<system>/<source>/<system>_<source>_root_translation_policy.json`
- `out_p6_motive_captury_comparison/<trial>/motive_marker_occlusions.csv`
- `out_p6_motive_captury_comparison/<trial>/trial_events_contacts.csv`
- `out_p6_motive_captury_comparison/<trial>/temporal_synchronization.json`
- `out_p6_motive_captury_comparison/<trial>/temporal_synchronization_timeseries.npz`
- `out_p6_motive_captury_comparison/<trial>/temporal_phase_normalized.npz`
- `out_p6_motive_captury_comparison/<trial>/phase_normalized_scientific_timeseries.npz`
- `out_p6_motive_captury_comparison/<trial>/contact_cycles.json`
- `out_p6_motive_captury_comparison/<trial>/model_dimensions.csv`
- `out_p6_motive_captury_comparison/<trial>/skin_marker_correspondence_metrics.csv`
- `out_p6_motive_captury_comparison/all_joint_centre_metrics.csv`
- `out_p6_motive_captury_comparison/all_kinematics_q_metrics.csv`
- `out_p6_motive_captury_comparison/all_motive_marker_occlusions.csv`
- `out_p6_motive_captury_comparison/all_model_dimensions.csv`
- `out_p6_motive_captury_comparison/all_skin_marker_correspondence_metrics.csv`
- `out_p6_motive_captury_comparison/population_kinematics_summary.csv`
- `out_p6_motive_captury_comparison/population_joint_centre_summary.csv`
- `out_p6_motive_captury_comparison/population_segment_summary.csv`
- `out_p6_motive_captury_comparison/population_aggregation.json`
- `out_p6_motive_captury_comparison/isb_d1_d3_audit.json`
- `out_p6_motive_captury_comparison/isb_d1_d3_audit.npz`
- `out_p6_motive_captury_comparison/run_report.json`

Trial-level results are cached in each trial's `run_report.json`. A trial is reused
when the source C3D/BVH/FBX files, the relevant options, the comparison script and
the static Captury -> Motive alignment all match the previous run. This avoids
rebuilding BioBuddy models and recomputing metrics during GUI exploration. Use
`--no-cache`, or the GUI's `Avancé -> Ignorer le cache` checkbox, to force a full
recompute:

```bash
python compare_p6_motive_captury.py \
  --data-root local_trials/2026-06-30_P6_flat \
  --trial Static \
  --out-dir out_p6_motive_captury_debug \
  --no-cache
```

The GUI reads compact metric CSV outputs for summary tables and fast `.npz` outputs for time series. Occlusions are shown as a sortable table with marker names stripped of prefixes such as `Skeleton_001_`; the table can be sorted by clicking the column headers. The other metric tabs render embedded graphs with hierarchical menus for `trial -> metric -> component`, covering joint-centre signed bias and limits of agreement, SO(3) geodesic segment deviations, bias/MAE/RMSE, waveform limits of agreement, ROM, gain, extrema timing, correlation/CCC, Captury C3D angle summaries and contact-detection signals. Correlation, CCC and range-normalized RMSE are hidden when either waveform has less than 1 degree of rotational amplitude, less than 80% paired coverage, or fewer than 10 paired samples. Frame-level limits of agreement are descriptive within-trial diagnostics, not population confidence intervals. Segment metric selections use `segment_rotation_timeseries.npz` to show time curves or absolute-deviation boxplots. Joint-centre metric selections are shown as one boxplot per centre using frame-by-frame errors, while model dimensions use grouped source-colored bars for single-metric length comparisons. In the joint-centre graph, a selected joint uses `joint_centre_timeseries.npz` to show error curves over time. In the kinematics graph, a single selected DoF or Captury C3D angle channel uses `kinematics_q_timeseries.npz` to show time curves; broader metric selections use summary boxplots. The integrated 3D trial viewer can overlay Captury, Motive and BioBuddy joint-centre chains and optional compact RGB local triads on those chains. Results refresh automatically when a selected movement finishes its lightweight analysis.

Population summaries keep exact trial names separate, then aggregate participant values with a participant-cluster bootstrap; grouping named repetitions such as `Marche_001` and `Marche_002` into one movement family is not yet inferred automatically. They refuse rows without an explicit participant identifier, report `insufficient_participants` below two participants and never treat frames as independent subjects. Frame-level limits of agreement, Pearson correlations and CCC are deliberately excluded from population aggregation because they require dedicated repeated-measures or transformed estimators. Native translation coordinates retain absolute bias/MAE/RMSE, but normalized/shape metrics are blocked until their physical unit and a defensible amplitude threshold are declared. Trial reports also write `metric_quality.json` and `metric_sensitivity.json`. The latter exposes only counterfactuals actually computed (root-translation policy, temporal lag, calibration versus held-out centres, and optional BVH/FBX rotation audit); unavailable anatomical convention variants remain explicitly `not_computed`.

The enriched Motive C3D copies contain generated model joint centres:

- `CAPJC_*`: Captury centres after static rigid alignment Captury -> Motive, then Motive-model -> Motive-marker C3D yaw/translation alignment.
- `MOTJC_*`: Motive model centres after Motive-model -> Motive-marker C3D yaw/translation alignment.

Open/validate a visualization for one trial without launching the Rerun viewer:

```bash
PYORERUN_HEADLESS=1 python compare_p6_motive_captury.py \
  --data-root local_trials/2026-06-30_P6_flat \
  --trial Static \
  --visualize \
  --visualize-trial Static \
  --headless \
  --rerun-wait-seconds 0 \
  --out-dir out_p6_motive_captury_visual_check
```

For an interactive Rerun view, remove `PYORERUN_HEADLESS=1` and `--headless`. The current visualization displays the enriched C3D joint-centre channels. FBX meshes are generated when `--model-source fbx` and mesh extraction is enabled, but Motive FBX files may not contain usable geometry; this is reported under each trial's `run_report.json`.

Run BioBuddy inverse kinematics in batch with bounded nonlinear TRF:

```bash
python compare_p6_motive_captury.py \
  --data-root local_trials/2026-06-30_P6_flat \
  --trial LKnee \
  --run-biobuddy-ik-batch \
  --ik-max-frames 50 \
  --out-dir out_p6_motive_captury_ik_check
```

Kinematic comparisons in `kinematics_q_metrics.csv` are intentionally conservative: they compare matching generalized-coordinate names from the generated BioBuddy models. Translation channels are useful for gross motion checks. Rotation channels are written in radians in the CSV outputs, then converted to degrees in the GUI for readability. Captury and Motive BVH/FBX exports may use different local segment frames, Euler sequences or axis signs, so angular differences should be interpreted as diagnostic rather than direct biomechanical agreement.

Captury C3D angle channels are excluded from marker processing, stored in `captury_c3d_angle_timeseries.npz`, and appended to the kinematics GUI as `CapturyC3D_*` channels. The versioned `captury_c3d_angles.json` registry maps the 13 abbreviated `POINT:LABELS` channels to their `POINT:ANGLES` names and anatomical articulations. This decodes channel identity only: Captury P6 files do not provide a dedicated angle-unit parameter, Euler sequence, signs, or anatomical meaning for X/Y/Z. The GUI/CLI `deg` or `rad` selection is therefore recorded as an explicit assumption, and every raw component remains `eligible_for_anatomical_agreement=false`. `captury_c3d_angle_decode.json` reports constants, exact duplicate curves, discontinuities and out-of-range values for each trial. In P6 Static, the decoder finds exact copies across several shoulder/elbow/wrist components and constant `left_elbow/Y` and `left_wrist/Z`; these channels must not be interpreted as independent anatomical DoFs. The Motive C3D files inspected here do not expose matching C3D angle channels. Captury duplicate marker labels are inventoried separately in `run_report.json`; current marker correspondences average duplicate labels until they are renamed more explicitly.

## Local Marker Test

For each C3D marker, the script uses biorbd segment rototranslations to express the marker in every segment's local frame. It assigns the marker to the segment where that local position varies least across frames, writes the local mean position into the corresponding `bioMod`, and reports stability statistics in the local marker CSV files.

The script then recomputes each local marker position in the global frame while the BVH or FBX model is animated. The Euclidean norm between this model marker and the measured C3D marker is saved in `*_c3d_marker_error_norm_mm.csv`, summarized in JSON, and displayed as per-marker and overall boxplots in millimetres. If several C3D channels have the same visible marker label, their channel indices are retained so every physical marker remains distinct. This measures residual fit on the same trial used to attach markers to segments; it is not an independent validation trial.

## Inverse Kinematics

The BioBuddy batch now uses only the generated BioBuddy bioMod and the Motive C3D markers. It no longer launches the historical BVH/FBX inverse-kinematics pipeline. The bounded solver is reported as `scipy.optimize.least_squares:trf`; `--run-ik-batch` remains a compatibility alias for `--run-biobuddy-ik-batch`. Results are cached by SHA-256 of the BioMod and C3D plus the solver, marker mapping, unit scale, frame limit and tolerance. The static result calculated after model creation is therefore reused by a later batch with identical parameters.

Each cached summary reports elapsed time, `nfev`, successful frames and marker residuals globally, per marker and per parent segment. The NPZ stores `q`, `qdot`, `qddot`, residuals, `nfev` and timing arrays. The cache key includes the IK implementation hash, and the NPZ digest and minimal schema are checked before reuse. On P6 Static, the corrected technical-marker indexing uses 47/47 markers and reduces the 10-frame diagnostic from the historical approximately 58.7 mm to 0.350 mm mean marker residual; the complete 773-frame static trial reaches a 0.192 mm in-sample residual in about 1.0 s. This is goodness of fit on the markers optimized by IK, not independent validation error. The former value came primarily from mixing indices of all BioMod markers with indices of technical markers after interleaved anatomical markers; it was not evidence that TRF tolerances should be relaxed.

P6 `Marche_001` remains a scientific warning rather than a solver-performance target: the complete trial converges in about 8.8 s but retains 34.95 mm mean marker error. At its first frame, independent rigid fits of marker clusters from the static model are only 0.23-6.15 mm RMSE, while the linked BioBuddy chain produces its largest residuals on upper arms and hands. This points toward inter-segment geometry, functional joint centres/axes or model constraints, rather than a global unit error. No acceptance threshold is silently applied: converged frames remain `diagnostic_only` until this dynamic inconsistency is resolved. BioBuddy DoF names are not assumed homologous to Motive/Captury export names; cross-model kinematics are compared through segment rotation matrices and audited joint conventions until an explicit anatomical DoF map exists.

The “Visu 3D essai” preview supports Shift-drag panning, ordinary drag rotation, wheel zoom, and −/+/Recentrer buttons. “Grande fenêtre” opens an independent snapshot of the loaded trial with the same marker and CoR layers, selected frame, camera, and visibility settings; it includes layer toggles, a frame slider, and playback. Reopen it to load a different trial or refreshed analysis.
