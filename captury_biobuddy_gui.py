"""Tkinter GUI launcher for the Captury/BioBuddy comparison pipelines.

This module deliberately keeps the scientific work in the existing command line
scripts. The GUI owns only interaction concerns:

* collecting file paths and options;
* building the equivalent CLI command with ``sys.executable``;
* launching long tasks in a background subprocess;
* streaming stdout/stderr into a log popup;
* reading generated CSV outputs for tables, plots and the lightweight 3D view.

That separation is important for reproducibility: every GUI action can be copied
as a CLI command and rerun outside Tkinter. New analysis logic should therefore
go in ``compare_p6_motive_captury.py`` or the lower-level helpers, not in this
file.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Iterable

import numpy as np

from c3d_trial_viewer import load_c3d_marker_data
from captury_frame_calibration import (
    captury_display_labels,
    load_captury_q_landmark_config,
)
from gui_commands import (
    C3D_VIEWER_SCRIPT,
    COMMAND_MODES,
    MODEL_EDITOR_SCRIPT,
    PIPELINE_SCRIPT,
    PROJECT_DIR,
    ROOT_OFFSET_MODE_CHOICES,
    ROOT_OFFSET_MODE_LABELS,
    build_biobuddy_c3d_ik_args,
    build_biobuddy_c3d_model_args,
    build_comparison_args,
    build_p6_args,
    build_p6_auto_analysis_args,
    build_p6_occlusions_args,
    build_pipeline_args,
    split_extra_labels,
    split_lines,
)
from gui_graphs import (
    EVENT_METRICS,
    GRAPH_CONFIGS,
    KINEMATIC_TIMESERIES_COLUMNS,
    draw_dimension_metric_graph,
    draw_segment_rotation_timeseries,
    joint_centre_error_boxplot_series_between,
    draw_joint_centre_error_timeseries_between,
    draw_metric_boxplot,
    graph_metric_columns,
    is_rotation_q_name,
    metric_display_name,
    read_table_npz,
    segment_rotation_boxplot_series,
    values_for_display,
)
from gui_marker_correspondence import (
    marker_pair_key,
    marker_pair_to_payload,
    payload_to_tree_values,
    save_marker_correspondence_payload,
    tree_values_to_payload,
)
from gui_run_report import summarize_run_report
from gui_state import (
    default_gui_session_state_path,
    load_gui_session_state,
    save_gui_session_state,
)
from gui_isb_report import (
    SOURCE_IDS_BY_LABEL,
    SOURCE_LABELS,
    filter_isb_rows,
    isb_filter_values,
    isb_table_values,
)
from gui_trial_viewer import (
    COR_LAYER_LABELS,
    DATA_SOURCE_COLORS,
    C3DMarkerData,
    JointCentreChainData,
    TkC3DTrialCanvas,
    available_cor_layers,
    captury_marker_transform_from_c3d_layers,
    captury_marker_transform_from_report,
    data_source_color,
    data_source_marker_color,
    display_marker_name,
    is_joint_centre_marker_label,
    joint_chain_edges,
    load_joint_centre_chain_data,
    marker_display_labels,
    transformed_marker_data,
    vertical_axis_label,
)
from motive57_c3d_mapping import (
    MOTIVE_57_C3D_ROLES,
    assignments_from_payload,
    discover_c3d_files,
    infer_motive57_role_assignments,
    load_motive57_mapping,
    motive57_mapping_path,
    motive57_mapping_payload,
    save_motive57_mapping,
)
from mocap_coordinate_frames import resolve_system_coordinate_frames
from gui_gait import (
    copy_gait_exports,
    draw_gait_cycles,
    cycle_selection,
    save_cycle_selection,
    cycle_options,
    selected_cycle_keys,
)

try:
    import pandas as pd
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    EMBEDDED_GRAPHS_AVAILABLE = True
except Exception:  # pragma: no cover - optional GUI plotting dependency
    pd = None
    Figure = None
    FigureCanvasTkAgg = None
    EMBEDDED_GRAPHS_AVAILABLE = False

ALL_TRIALS_LABEL = "Tous les essais"
P6_AUTO_ANALYSIS_DEBOUNCE_MS = 600
CRITICAL_METHOD_NOTES = (
    {
        "title": "Recalage FBX/BVH -> C3D",
        "algorithm": (
            "Pour chaque modèle BVH/FBX, les deux conventions de translation racine "
            "sont générées: conserver q tel qu'exporté ou soustraire l'offset statique "
            "de la racine. Le mode auto convertit ensuite les centres en repère C3D, "
            "score les deux superpositions contre le nuage de marqueurs, puis garde "
            "la meilleure avant les recalages Captury -> Motive et yaw/translation."
        ),
        "risk": (
            "Si le meilleur score reste élevé, l'offset racine n'explique pas tout: "
            "la pose statique, les correspondances anatomiques, les unités ou le choix "
            "yaw-only peuvent encore masquer un mauvais placement segmentaire."
        ),
        "check": (
            "Inspecter les fichiers *_root_translation_policy.json, la visu 3D, les "
            "distances CAPJC/MOTJC, la cohérence verticale sur Z et les repères XYZ "
            "locaux avant d'interpréter les erreurs."
        ),
    },
    {
        "title": "Cohérence des modèles",
        "algorithm": (
            "Les modèles Captury, Motive et BioBuddy sont construits séparément depuis "
            "les sources disponibles. Les comparaisons de q se font par noms de DoF "
            "communs et les dimensions par segments ou centres disponibles."
        ),
        "risk": (
            "Deux segments portant le même nom peuvent avoir des repères locaux, "
            "longueurs, offsets ou conventions Euler différents."
        ),
        "check": (
            "Comparer les dimensions, afficher les trois chaînes CoR et éviter de "
            "traiter une différence angulaire comme un écart biomécanique direct."
        ),
    },
    {
        "title": "Orientation des données",
        "algorithm": (
            "Le mode auto suppose FBX/BVH en +Y vertical et C3D Motive en +Z vertical. "
            "Les vues face/dos/côté utilisent des estimations géométriques/PCA quand "
            "la caméra doit être orientée automatiquement."
        ),
        "risk": (
            "Une mauvaise hypothèse d'axe vertical, une inversion gauche-droite ou "
            "une ambiguïté de signe PCA peut faire paraître les chaînes incohérentes."
        ),
        "check": (
            "Vérifier l'axe vertical dans l'onglet Données, utiliser les vues XY/YZ/XZ "
            "et confirmer que les marqueurs et chaînes suivent le même mouvement."
        ),
    },
    {
        "title": "Mise à l'échelle",
        "algorithm": (
            "Les C3D sont normalisés à partir de POINT:UNITS. Pour P6, Captury BVH/FBX "
            "est traité en millimètres et Motive BVH/FBX en centimètres par défaut."
        ),
        "risk": (
            "Un metadata C3D erroné ou une unité modèle mal déclarée crée des erreurs "
            "d'un facteur 10 ou 100 qui ressemblent à un problème de recalage."
        ),
        "check": (
            "Lire les dimensions de modèles, vérifier les amplitudes verticales et "
            "confirmer les échelles avant de comparer les distances en millimètres."
        ),
    },
    {
        "title": "Angles articulaires et q",
        "algorithm": (
            "Les rotations BioBuddy sont sauvegardées en radians dans les CSV, puis "
            "affichées en degrés dans le GUI. Les DoF suivent l'ordre q_names/bioMod; "
            "les segments FBX générés indiquent typiquement rotations zyx."
        ),
        "risk": (
            "Les exports Captury/Motive peuvent utiliser des frames locaux, axes de "
            "rotation, offsets et séquences Euler non équivalents."
        ),
        "check": (
            "Comparer d'abord les formes temporelles DoF par DoF, puis documenter les "
            "conventions avant toute conclusion quantitative sur les angles."
        ),
    },
)


def safe_trial_dir_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("_") or "trial"


def captury_trial_name_from_path(path: Path) -> str:
    name = path.stem
    return name[: -len("_P6")] if name.endswith("_P6") else name


def motive_trial_name_from_path(path: Path) -> str:
    name = path.stem
    if name.startswith("P6_"):
        name = name[3:]
    if name.endswith("_Skeleton 001"):
        name = name[: -len("_Skeleton 001")]
    return name


def inventory_p6_dataset(data_root: Path) -> dict[str, dict[str, dict[str, Path]]]:
    inventory: dict[str, dict[str, dict[str, Path]]] = {}
    captury_dir = data_root / "Captury"
    motive_dir = data_root / "Motive"
    if motive_dir.is_dir() or captury_dir.is_dir():
        for path in sorted(captury_dir.glob("*")):
            if path.suffix.lower() not in {".bvh", ".fbx", ".c3d"}:
                continue
            trial = captury_trial_name_from_path(path)
            kind = path.suffix.lower().lstrip(".")
            inventory.setdefault(trial, {}).setdefault("Captury", {})[kind] = path
        for path in sorted(motive_dir.glob("*")):
            if path.suffix.lower() not in {".bvh", ".fbx", ".c3d"}:
                continue
            trial = motive_trial_name_from_path(path)
            kind = path.suffix.lower().lstrip(".")
            inventory.setdefault(trial, {}).setdefault("Motive", {})[kind] = path
        return inventory

    for trial_dir in sorted(path for path in data_root.glob("*") if path.is_dir()):
        captury = trial_dir / "captury"
        motive = trial_dir / "squelettes"
        if captury.is_dir():
            for path in sorted(captury.glob("P6.*")):
                if path.suffix.lower() not in {".bvh", ".fbx", ".c3d"}:
                    continue
                kind = path.suffix.lower().lstrip(".")
                inventory.setdefault(trial_dir.name, {}).setdefault("Captury", {})[
                    kind
                ] = path
        if motive.is_dir():
            for path in sorted(motive.glob("*")):
                if path.suffix.lower() not in {".bvh", ".fbx", ".c3d"}:
                    continue
                kind = path.suffix.lower().lstrip(".")
                inventory.setdefault(trial_dir.name, {}).setdefault("Motive", {})[
                    kind
                ] = path
    return inventory


class CapturyBioBuddyGui(tk.Tk):
    """Main Captury/Motive analysis window.

    The class is intentionally a GUI coordinator rather than a numerical
    analysis engine. Its responsibilities are split into five local regions:

    * Tk variable creation and command preview synchronization;
    * tab construction for data loading, occlusions, cutting, dimensions,
      centres, markers, kinematics, visualization and advanced settings;
    * non-blocking subprocess execution and log/status updates;
    * lightweight CSV exploration through sortable tables and Matplotlib
      canvases;
    * synchronized right-hand 3D trial preview for C3D markers and CoR chains.

    The generated command is the source of truth for computation. GUI plotting
    code only reads existing CSV files from the selected output directory. This
    keeps batch behavior testable from the CLI and makes the GUI safe to use as
    an exploratory layer over cached results.
    """

    def __init__(self) -> None:
        super().__init__(baseName="captury_biobuddy", className="CapturyBioBuddy")
        self.title("Captury BioBuddy")
        self.geometry("1180x780")
        self.minsize(980, 680)

        self.process: subprocess.Popen[str] | None = None
        self.running_command_mode: str | None = None
        self.output_queue: queue.Queue[str | tuple[str, int]] = queue.Queue()
        self.analysis_buttons: list[ttk.Button] = []
        self.trial_inventory: dict[str, dict[str, dict[str, Path]]] = {}
        self.command_text: tk.Text | None = None
        self.command_window: tk.Toplevel | None = None
        self.log_text: tk.Text | None = None
        self.log_window: tk.Toplevel | None = None
        self.log_buffer = ""
        self.graph_panels: dict[str, dict[str, object]] = {}
        self.graph_payloads: dict[str, dict[str, dict[str, object]]] = {}
        self.graph_drag_selection: dict[str, object] = {}
        self.c3d_marker_cache: dict[tuple[str, int, int, str], C3DMarkerData] = {}
        self.joint_chain_cache: dict[
            tuple[str, int, int], JointCentreChainData | None
        ] = {}
        self.viewer_play_after_id: str | None = None
        self.occlusion_sort_column = "marker_order"
        self.occlusion_sort_descending = False
        self.pending_auto_analysis = False
        self.auto_analysis_after_id: str | None = None
        self.auto_enable_biobuddy_cor_after_refresh = False
        self.motive57_c3d_files: list[str] = []
        self.motive57_role_combos: dict[str, ttk.Combobox] = {}
        self.motive57_inventory_tree: ttk.Treeview | None = None

        self.vars: dict[str, tk.Variable] = {}
        self._create_variables()
        self.gui_session_state_path = default_gui_session_state_path()
        self._restore_gui_session_state()
        self._configure_style()
        self._build_layout()
        self._bind_command_preview()
        self._update_biobuddy_biomod_status()
        self._update_command_preview()
        self._update_embedded_trial_viewer()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._drain_output_queue)

    def _register_analysis_button(self, button: ttk.Button) -> ttk.Button:
        """Track buttons that start subprocess-backed analysis commands."""

        self.analysis_buttons.append(button)
        if self.process is not None:
            button.configure(state=tk.DISABLED)
        return button

    def _create_variables(self) -> None:
        defaults: dict[str, str | bool] = {
            "bvh": "data/unknown.bvh",
            "fbx": "data/unknown.fbx",
            "c3d": "data/unknown.c3d",
            "out_dir": "out_biobuddy_bvh_c3d",
            "bvh_unit_scale_to_m": "0.001",
            "fbx_unit_scale_to_m": "0.001",
            "c3d_angle_unit": "deg",
            "angle_label_regex": r"(?i)(^.*angles?$|^.*_angle[s]?$|angle)",
            "extra_angle_labels": "",
            "comparison_map": "",
            "model_explorer_path": "out_biobuddy_bvh_c3d/model_from_fbx_biobuddy.bioMod",
            "no_biomod_joint_centre_markers": False,
            "no_root_offset_correction": False,
            "root_offset_mode": ROOT_OFFSET_MODE_LABELS["auto"],
            "no_fbx_mesh": False,
            "max_fbx_mesh_points": "0",
            "animate": False,
            "animate_superposed": False,
            "display_q_in_rerun": False,
            "rerun_marker_radius": "15",
            "rerun_wait_seconds": "2",
            "rerun_up_axis": "y",
            "hide_hands_in_rerun": False,
            "hide_feet_in_rerun": False,
            "hide_extremities_in_rerun": False,
            "headless": False,
            "inverse_kinematics": False,
            "inverse_kinematics_solver": "least_squares",
            "inverse_kinematics_method": "trf",
            "inverse_kinematics_max_frames": "0",
            "kalman_noise_factor": "1e-10",
            "kalman_error_factor": "1e-5",
            "inverse_dynamics": False,
            "inverse_dynamics_method": "",
            "inverse_dynamics_max_frames": "",
            "compare_data_root": "local_trials/data",
            "compare_reference_system": "Motive",
            "compare_test_system": "Captury",
            "compare_reference_c3d": "",
            "compare_reference_bvh": "",
            "compare_reference_fbx": "",
            "compare_test_c3d": "",
            "compare_test_bvh": "",
            "compare_test_fbx": "",
            "compare_trial_name": "",
            "compare_participant_filter": "",
            "compare_trial_filter": "",
            "compare_out_dir": "out_capture_system_comparison",
            "compare_landmark_map": "",
            "compare_resample_points": "101",
            "compare_alignment": "global_rigid",
            "p6_data_root": "local_trials/2026-06-30_P6_flat",
            "p6_out_dir": "out_p6_motive_captury_comparison",
            "p6_trials": "",
            "selected_trial": ALL_TRIALS_LABEL,
            "p6_static_trial": "Static",
            "p6_cut_mode": "manual",
            "p6_temporal_sync_mode": "auto",
            "p6_manual_lag_s": "",
            "p6_max_lag_s": "0.5",
            "p6_phase_normalization_points": "101",
            "p6_time_start": "",
            "p6_time_end": "",
            "p6_joint_filter": "",
            "p6_joint_centre_reference": "biobuddy",
            "p6_auto_analyze": True,
            "p6_normalize_gait": True,
            "p6_cycle_task": "gait",
            "p6_cycle_detection": "automatic",
            "p6_squat_standing_deg": "15",
            "p6_squat_peak_deg": "40",
            "p6_jump_height_mm": "30",
            "p6_gait_events_csv": "",
            "p6_gait_min_cycle_s": "0.4",
            "p6_gait_max_cycle_s": "2.5",
            "p6_gait_max_gap_s": "0.05",
            "p6_gait_velocity_threshold_m_s": "0.5",
            "p6_gait_filter_hz": "7.0",
            "p6_no_figures": True,
            "p6_no_cache": False,
            "p6_model_source": "bvh",
            "p6_source_captury": True,
            "p6_source_motive": True,
            "p6_source_biobuddy": True,
            "p6_model_to_c3d_axis": "auto",
            "p6_captury_model_to_c3d_axis": "auto",
            "p6_motive_model_to_c3d_axis": "auto",
            "p6_captury_q_landmarks_json": "captury_q_landmarks.json",
            "p6_segment_reference": "biobuddy",
            "p6_captury_reorient_thigh_y_from_cor": False,
            "p6_captury_rename_axes": False,
            "p6_captury_calibrate_segment_frames": False,
            "p6_rotate_body_segments_180_x": False,
            "p6_reexpress_rotations_zxy": False,
            "p6_disable_static_model_alignment": False,
            "p6_disable_motive_marker_alignment": False,
            "p6_no_mesh": False,
            "p6_max_mesh_points": "0",
            "p6_run_ik_batch": False,
            "p6_ik_max_frames": "0",
            "p6_visualize": False,
            "p6_visualize_trial": "",
            "p6_headless": False,
            "p6_rerun_wait_seconds": "1",
            "biobuddy_c3d_folder": "",
            "biobuddy_c3d_preset": "motive_57",
            "biobuddy_c3d_output": "/tmp/motive_57.bioMod",
            "biobuddy_c3d_mapping_json": "",
            "biobuddy_c3d_with_mesh": False,
            "biobuddy_c3d_no_default_virtual_points": False,
            "command_mode": COMMAND_MODES["kinematic"],
        }
        for role in MOTIVE_57_C3D_ROLES:
            defaults[f"motive57_role_{role.key}"] = ""
        for name, value in defaults.items():
            var_cls = tk.BooleanVar if isinstance(value, bool) else tk.StringVar
            self.vars[name] = var_cls(value=value)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background="#f6f7f8")
        style.configure("TLabelframe", background="#f6f7f8")
        style.configure("TLabelframe.Label", background="#f6f7f8", foreground="#1f2933")
        style.configure("TLabel", background="#f6f7f8", foreground="#1f2933")
        style.configure("TButton", padding=(10, 6))
        style.configure("Primary.TButton", padding=(12, 7))
        style.configure("Danger.TButton", padding=(12, 7))
        style.configure("Status.TLabel", foreground="#475569")
        for source, color in DATA_SOURCE_COLORS.items():
            style.configure(
                f"{source.title()}.TCheckbutton",
                background="#f6f7f8",
                foreground=color,
            )

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        root.rowconfigure(2, weight=0)

        header = ttk.Frame(root)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=0)
        ttk.Label(
            header, text="Captury BioBuddy", font=("TkDefaultFont", 18, "bold")
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Génération bioMod, comparaison C3D, visualisation Rerun et cinématique inverse",
            style="Status.TLabel",
        ).grid(row=1, column=0, sticky="w")
        trial_selector = ttk.Frame(header)
        trial_selector.grid(row=0, column=1, rowspan=2, sticky="e")
        ttk.Label(trial_selector, text="Essai").grid(row=0, column=0, sticky="w")
        self.trial_combobox = ttk.Combobox(
            trial_selector,
            textvariable=self.vars["selected_trial"],
            values=(ALL_TRIALS_LABEL,),
            state="readonly",
            width=24,
        )
        self.trial_combobox.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        body = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        body.grid(row=1, column=0, sticky="nsew")

        left = ttk.Frame(body, padding=(0, 0, 10, 0))
        right = ttk.Frame(body)
        body.add(left, weight=3)
        body.add(right, weight=2)

        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        notebook = ttk.Notebook(left)
        self.main_notebook = notebook
        notebook.grid(row=0, column=0, sticky="nsew")

        self._build_loading_matching_tab(notebook)
        self._build_biobuddy_c3d_model_tab(notebook)
        self._build_occlusions_tab(notebook)
        self._build_trial_cutting_tab(notebook)
        self._build_dimensions_tab(notebook)
        self._build_segments_tab(notebook)
        self._build_joint_centres_tab(notebook)
        self._build_skin_markers_tab(notebook)
        self._build_kinematics_compare_tab(notebook)
        self._build_gait_tab(notebook)
        self._build_visualization_tab(notebook)
        self._build_critical_methods_tab(notebook)
        self._build_advanced_tab(notebook)

        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        self._build_actions(right)
        self._build_embedded_trial_viewer(right)
        self._build_footer_tools(root)
        self._refresh_trial_inventory()
        self._refresh_results()

    def _tab(self, notebook: ttk.Notebook, title: str) -> ttk.Frame:
        frame = ttk.Frame(notebook, padding=12)
        frame.columnconfigure(0, weight=1)
        notebook.add(frame, text=title)
        return frame

    def _analysis_action_row(self, parent: ttk.Widget, row: int) -> None:
        actions = ttk.Frame(parent)
        actions.grid(row=row, column=0, sticky="ew", pady=(12, 0))
        for column in range(2):
            actions.columnconfigure(column, weight=1)
        run_button = self._register_analysis_button(
            ttk.Button(
                actions,
                text="Lancer analyse",
                style="Primary.TButton",
                command=self._run_p6_analysis,
            )
        )
        run_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(actions, text="Copier commande", command=self._copy_p6_command).grid(
            row=0, column=1, sticky="ew", padx=(6, 0)
        )

    def _build_loading_matching_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Données")
        data = ttk.LabelFrame(tab, text="Chargement et correspondance")
        data.grid(row=0, column=0, sticky="ew")
        data.columnconfigure(1, weight=1)
        self._path_row(data, 0, "Dossier racine", "p6_data_root", directory=True)
        self._path_row(data, 1, "Sortie", "p6_out_dir", directory=True)
        self._path_row(
            data,
            2,
            "Modèle biorbd",
            "biobuddy_c3d_output",
            [("Modèles biorbd", "*.bioMod *.bioModx"), ("Tous les fichiers", "*")],
        )
        self.biobuddy_biomod_status_var = tk.StringVar(value="")
        ttk.Label(
            data,
            textvariable=self.biobuddy_biomod_status_var,
            style="Status.TLabel",
        ).grid(row=3, column=1, columnspan=2, sticky="w", padx=10, pady=(0, 6))
        self._entry_row(data, 4, "Essai statique", "p6_static_trial")
        self._combo_row(
            data, 5, "Source modèle", "p6_model_source", ("bvh", "fbx", "auto")
        )
        self._combo_row(
            data,
            6,
            "Captury modèle -> C3D",
            "p6_captury_model_to_c3d_axis",
            ("auto", "y_up_to_z_up", "identity"),
        )
        self._combo_row(
            data,
            7,
            "Motive modèle -> C3D",
            "p6_motive_model_to_c3d_axis",
            ("auto", "y_up_to_z_up", "identity"),
        )
        self._combo_row(
            data,
            8,
            "Offset trans. racine",
            "root_offset_mode",
            ROOT_OFFSET_MODE_CHOICES,
        )
        self._check(
            data,
            9,
            "R(x,180°)R(y,180°)",
            "p6_rotate_body_segments_180_x",
        )
        self._check(
            data,
            10,
            "Captury : renommer axes R(y,90°) (diagnostic)",
            "p6_captury_rename_axes",
        )
        self._check(
            data,
            11,
            "Captury : calibration repères Q_* + CoR (statique)",
            "p6_captury_calibrate_segment_frames",
        )
        self._path_row(
            data,
            12,
            "Config Q_* Captury",
            "p6_captury_q_landmarks_json",
            [("JSON", "*.json"), ("Tous les fichiers", "*")],
        )
        self._check(
            data,
            13,
            "Désactiver recalage statique Captury -> Motive",
            "p6_disable_static_model_alignment",
        )
        self._check(
            data,
            14,
            "Désactiver recalage Motive -> marqueurs C3D",
            "p6_disable_motive_marker_alignment",
        )
        source_panel = ttk.LabelFrame(data, text="Sources à inclure si disponibles")
        source_panel.grid(row=15, column=0, columnspan=3, sticky="ew", padx=10, pady=8)
        self.source_selection_checks = {}
        for column, (source, label) in enumerate(
            (("captury", "Captury"), ("motive", "Motive"), ("biobuddy", "BioBuddy"))
        ):
            check = ttk.Checkbutton(
                source_panel, text=label, variable=self.vars[f"p6_source_{source}"]
            )
            check.grid(row=0, column=column, sticky="w", padx=8, pady=4)
            self.source_selection_checks[source] = check
        ttk.Label(
            source_panel,
            text="Chaque essai utilise les sources disponibles. Vous pouvez ajouter Captury plus tard.",
            wraplength=500,
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 4))
        inventory = ttk.LabelFrame(tab, text="Fichiers détectés")
        inventory.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        inventory.rowconfigure(0, weight=1)
        inventory.columnconfigure(0, weight=1)
        inventory.columnconfigure(1, weight=1)
        self.inventory_trees: dict[str, ttk.Treeview] = {}
        for column, system in enumerate(("Captury", "Motive")):
            system_frame = ttk.LabelFrame(inventory, text=system)
            system_frame.grid(
                row=0,
                column=column,
                sticky="nsew",
                padx=(8, 4) if column == 0 else (4, 8),
                pady=8,
            )
            system_frame.rowconfigure(0, weight=1)
            system_frame.columnconfigure(0, weight=1)
            tree = ttk.Treeview(
                system_frame,
                columns=("kind", "vertical_axis", "path"),
                show="headings",
                height=6,
                selectmode="browse",
            )
            tree.heading("kind", text="Type")
            tree.heading("vertical_axis", text="Axe vertical")
            tree.heading("path", text="Fichier")
            tree.column("kind", width=60, stretch=False)
            tree.column("vertical_axis", width=110, stretch=False)
            tree.column("path", width=360, stretch=True)
            tree.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=8)
            scrollbar = ttk.Scrollbar(
                system_frame, orient=tk.VERTICAL, command=tree.yview
            )
            tree.configure(yscrollcommand=scrollbar.set)
            scrollbar.grid(row=0, column=1, sticky="ns", pady=8)
            self.inventory_trees[system] = tree

        systems = ttk.LabelFrame(tab, text="Systèmes comparés")
        systems.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        systems.columnconfigure(1, weight=1)
        self._entry_row(systems, 0, "Référence", "compare_reference_system")
        self._entry_row(systems, 1, "Test", "compare_test_system")
        self._path_row(
            systems,
            2,
            "Carte repères",
            "compare_landmark_map",
            [("JSON", "*.json"), ("Tous les fichiers", "*")],
        )
        self._analysis_action_row(tab, 3)

    def _build_biobuddy_c3d_model_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "BioBuddy")
        tab.rowconfigure(1, weight=1)
        creation = ttk.LabelFrame(tab, text="Création modèle depuis C3D")
        creation.grid(row=0, column=0, sticky="ew")
        creation.columnconfigure(1, weight=1)
        self._path_row(
            creation,
            0,
            "Dossier C3D (vide = Motive)",
            "biobuddy_c3d_folder",
            directory=True,
        )
        self._combo_row(
            creation,
            1,
            "Preset",
            "biobuddy_c3d_preset",
            (
                "motive_57",
                "full_body",
                "lower_limbs",
                "lower_limbs_anatomical",
                "upper_limb",
                "from_scratch",
            ),
        )
        self._save_path_row(
            creation,
            2,
            "Sortie bioMod",
            "biobuddy_c3d_output",
            [("bioMod", "*.bioMod"), ("Tous les fichiers", "*")],
        )
        self._path_row(
            creation,
            3,
            "JSON rôles Motive 57",
            "biobuddy_c3d_mapping_json",
            [("JSON", "*.json"), ("Tous les fichiers", "*")],
        )
        self._check(creation, 4, "Écrire les meshes", "biobuddy_c3d_with_mesh")
        self._check(
            creation,
            5,
            "Ne pas ajouter les points virtuels par défaut",
            "biobuddy_c3d_no_default_virtual_points",
        )

        mapping = ttk.LabelFrame(tab, text="Fichiers Motive 57")
        mapping.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        mapping.rowconfigure(1, weight=1)
        mapping.columnconfigure(0, weight=1)
        mapping.columnconfigure(1, weight=2)

        mapping_actions = ttk.Frame(mapping)
        mapping_actions.grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=6
        )
        for column in range(4):
            mapping_actions.columnconfigure(column, weight=1)
        ttk.Button(
            mapping_actions,
            text="Inventorier",
            command=self._refresh_motive57_c3d_mapping,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(
            mapping_actions,
            text="Charger JSON",
            command=self._load_motive57_mapping_json,
        ).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(
            mapping_actions,
            text="Sauver JSON",
            command=self._save_motive57_mapping_json,
        ).grid(row=0, column=2, sticky="ew", padx=4)
        ttk.Button(
            mapping_actions,
            text="Inférer",
            command=self._infer_motive57_mapping_from_files,
        ).grid(row=0, column=3, sticky="ew", padx=(4, 0))

        inventory_frame = ttk.Frame(mapping)
        inventory_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 6), pady=(0, 10))
        inventory_frame.rowconfigure(0, weight=1)
        inventory_frame.columnconfigure(0, weight=1)
        self.motive57_inventory_tree = ttk.Treeview(
            inventory_frame,
            columns=("file",),
            show="headings",
            height=8,
            selectmode="browse",
        )
        self.motive57_inventory_tree.heading("file", text="C3D Motive détectés")
        self.motive57_inventory_tree.column("file", width=280, anchor="w")
        inventory_scroll = ttk.Scrollbar(
            inventory_frame,
            orient=tk.VERTICAL,
            command=self.motive57_inventory_tree.yview,
        )
        self.motive57_inventory_tree.configure(yscrollcommand=inventory_scroll.set)
        self.motive57_inventory_tree.grid(row=0, column=0, sticky="nsew")
        inventory_scroll.grid(row=0, column=1, sticky="ns")

        roles_frame = ttk.Frame(mapping)
        roles_frame.grid(row=1, column=1, sticky="nsew", padx=(6, 10), pady=(0, 10))
        roles_frame.columnconfigure(1, weight=1)
        for row, role in enumerate(MOTIVE_57_C3D_ROLES):
            ttk.Label(roles_frame, text=role.label).grid(
                row=row, column=0, sticky="w", padx=(0, 8), pady=3
            )
            combo = ttk.Combobox(
                roles_frame,
                textvariable=self.vars[f"motive57_role_{role.key}"],
                values=("",),
                state="readonly",
            )
            combo.grid(row=row, column=1, sticky="ew", pady=3)
            self.motive57_role_combos[role.key] = combo
            ttk.Label(roles_frame, text=role.method, style="Status.TLabel").grid(
                row=row, column=2, sticky="w", padx=(8, 0), pady=3
            )

        actions = ttk.Frame(tab)
        actions.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        for column in range(5):
            actions.columnconfigure(column, weight=1)
        run_button = self._register_analysis_button(
            ttk.Button(
                actions,
                text="Créer modèle",
                style="Primary.TButton",
                command=self._run_biobuddy_c3d_model_creation,
            )
        )
        run_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(
            actions,
            text="Dossier Motive",
            command=self._use_selected_motive_folder_for_biobuddy,
        ).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(
            actions,
            text="Copier commande",
            command=self._copy_biobuddy_c3d_model_command,
        ).grid(row=0, column=2, sticky="ew", padx=6)
        ttk.Button(
            actions,
            text="Ouvrir dans BioBuddy",
            command=self._open_biobuddy_c3d_model_in_explorer,
        ).grid(row=0, column=3, sticky="ew", padx=6)
        batch_button = self._register_analysis_button(
            ttk.Button(
                actions,
                text="Batch IK non linéaire TRF",
                command=self._run_p6_ik_batch,
            )
        )
        batch_button.grid(row=0, column=4, sticky="ew", padx=(6, 0))

        notes = ttk.LabelFrame(tab, text="Notes")
        notes.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(
            notes,
            text=(
                "Ce panneau lance create_model_from_c3d_folder dans un subprocess. "
                "La commande équivalente reste disponible dans le bouton Commande. "
                "Le JSON Motive 57 mémorise la statique et les essais fonctionnels "
                "SCoRE/SARA à réutiliser quand le même dossier est rechargé."
            ),
            style="Status.TLabel",
            wraplength=760,
        ).grid(row=0, column=0, sticky="w", padx=10, pady=8)

    def _build_occlusions_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Occlusions")
        tab.rowconfigure(2, weight=1)
        panel = ttk.LabelFrame(tab, text="Marqueurs Motive")
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(1, weight=1)
        self._check(panel, 0, "Ne pas générer les PNG", "p6_no_figures")
        self._analysis_action_row(tab, 1)
        self._build_occlusion_table_panel(tab, 2)

    def _build_trial_cutting_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Découpage")
        tab.rowconfigure(2, weight=1)
        panel = ttk.LabelFrame(tab, text="Début, fin et contacts au sol")
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(1, weight=1)
        self._entry_row(panel, 0, "Essai statique", "p6_static_trial")
        self._combo_row(
            panel,
            1,
            "Synchronisation",
            "p6_temporal_sync_mode",
            ("auto", "manual", "none"),
        )
        self._entry_row(panel, 2, "Lag Captury manuel (s)", "p6_manual_lag_s")
        self._entry_row(panel, 3, "Lag auto maximal (s)", "p6_max_lag_s")
        self._entry_row(
            panel, 4, "Points phase 0-100 %", "p6_phase_normalization_points"
        )
        self._combo_row(
            panel,
            5,
            "Mode découpage",
            "p6_cut_mode",
            ("manual", "movement", "full"),
        )
        self._entry_row(panel, 6, "Début manuel (s)", "p6_time_start")
        self._entry_row(panel, 7, "Fin manuelle (s)", "p6_time_end")
        self._check(panel, 8, "Visualiser un essai enrichi", "p6_visualize")
        self._entry_row(panel, 9, "Essai visualisé", "p6_visualize_trial")
        ttk.Button(
            panel, text="Ouvrir visu 3D C3D", command=self._open_selected_trial_viewer
        ).grid(row=10, column=0, columnspan=3, sticky="ew", padx=10, pady=6)
        self._analysis_action_row(tab, 1)
        self._build_graph_panel(tab, 2, "events")

    def _build_dimensions_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Dimensions")
        tab.rowconfigure(2, weight=1)
        panel = ttk.LabelFrame(tab, text="Dimensions des modèles")
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(1, weight=1)
        self._combo_row(
            panel, 0, "Source modèle", "p6_model_source", ("bvh", "fbx", "auto")
        )
        self._check(panel, 1, "Ne pas extraire les meshes FBX", "p6_no_mesh")
        self._entry_row(panel, 2, "Max points mesh", "p6_max_mesh_points")
        self._analysis_action_row(tab, 1)
        self._build_graph_panel(tab, 2, "dimensions")

    def _build_segments_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Segments")
        tab.rowconfigure(2, weight=1)
        panel = ttk.LabelFrame(tab, text="Repères segmentaires")
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(1, weight=1)
        self._combo_row(
            panel,
            0,
            "Référence",
            "p6_segment_reference",
            ("biobuddy", "motive", "captury"),
        )
        self._combo_row(
            panel, 1, "Source modèle", "p6_model_source", ("bvh", "fbx", "auto")
        )
        self._entry_row(panel, 2, "Filtre segments", "p6_joint_filter")
        self._analysis_action_row(tab, 1)
        self._build_graph_panel(tab, 2, "segments")

    def _build_joint_centres_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Centres")
        tab.rowconfigure(2, weight=1)
        panel = ttk.LabelFrame(tab, text="Centres articulaires")
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(1, weight=1)
        self._entry_row(panel, 0, "Filtre centres", "p6_joint_filter")
        self._combo_row(
            panel,
            1,
            "Référence",
            "p6_joint_centre_reference",
            ("biobuddy", "motive", "captury"),
        )
        self._analysis_action_row(tab, 1)
        self._build_graph_panel(tab, 2, "centres")

    def _build_skin_markers_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Marqueurs")
        tab.rowconfigure(2, weight=1)
        tab.rowconfigure(3, weight=1)
        panel = ttk.LabelFrame(tab, text="Marqueurs cutanés correspondants")
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(1, weight=1)
        panel.columnconfigure(2, weight=1)
        panel.columnconfigure(3, weight=1)
        self._path_row(
            panel,
            0,
            "Carte repères",
            "compare_landmark_map",
            [("JSON", "*.json"), ("Tous les fichiers", "*")],
        )
        ttk.Button(
            panel,
            text="Charger JSON",
            command=self._load_marker_correspondence_json,
        ).grid(row=1, column=0, sticky="ew", padx=(10, 4), pady=(0, 10))
        ttk.Button(
            panel,
            text="Enregistrer JSON",
            command=self._save_marker_correspondence_json,
        ).grid(row=1, column=1, sticky="ew", padx=4, pady=(0, 10))
        ttk.Button(
            panel,
            text="Charger proposition",
            command=self._load_marker_correspondence_proposal_json,
        ).grid(row=1, column=2, sticky="ew", padx=4, pady=(0, 10))
        self.marker_metrics_button = self._register_analysis_button(
            ttk.Button(
                panel,
                text="Calculer métriques",
                command=self._save_marker_pairs_and_run_analysis,
            )
        )
        self.marker_metrics_button.grid(
            row=1, column=3, sticky="ew", padx=(4, 10), pady=(0, 10)
        )

        mapping = ttk.LabelFrame(tab, text="Mise en correspondance")
        mapping.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        for column in range(3):
            mapping.columnconfigure(column, weight=1)
        mapping.rowconfigure(1, weight=1)

        ttk.Label(mapping, text="Motive").grid(
            row=0, column=0, sticky="w", padx=10, pady=(8, 2)
        )
        ttk.Label(mapping, text="Captury").grid(
            row=0, column=1, sticky="w", padx=10, pady=(8, 2)
        )
        self.marker_motive_list = tk.Listbox(mapping, exportselection=False, height=8)
        self.marker_captury_list = tk.Listbox(mapping, exportselection=False, height=8)
        self.marker_motive_list.grid(row=1, column=0, sticky="nsew", padx=(10, 4))
        self.marker_captury_list.grid(row=1, column=1, sticky="nsew", padx=4)
        self.marker_motive_list.bind(
            "<<ListboxSelect>>", lambda _event: self._highlight_marker_pair_selection()
        )
        self.marker_captury_list.bind(
            "<<ListboxSelect>>", lambda _event: self._highlight_marker_pair_selection()
        )
        actions = ttk.Frame(mapping)
        actions.grid(row=1, column=2, sticky="nsew", padx=(4, 10))
        actions.columnconfigure(0, weight=1)
        ttk.Button(
            actions,
            text="Associer",
            command=self._add_marker_correspondence_pair,
        ).grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(
            actions,
            text="Supprimer",
            command=self._remove_selected_marker_correspondence_pair,
        ).grid(row=1, column=0, sticky="ew")

        self.marker_pair_tree = ttk.Treeview(
            mapping,
            columns=("name", "motive", "captury"),
            show="headings",
            height=5,
        )
        for column, label in (
            ("name", "Nom"),
            ("motive", "Motive"),
            ("captury", "Captury"),
        ):
            self.marker_pair_tree.heading(column, text=label)
            self.marker_pair_tree.column(column, width=160, stretch=True)
        self.marker_pair_tree.grid(
            row=2, column=0, columnspan=3, sticky="nsew", padx=10, pady=(8, 10)
        )
        self._analysis_action_row(tab, 2)
        self._build_graph_panel(tab, 3, "skin_markers")

    def _build_kinematics_compare_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Cinématiques")
        tab.rowconfigure(2, weight=1)
        panel = ttk.LabelFrame(tab, text="Angles articulaires et q")
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(1, weight=1)
        self._entry_row(panel, 0, "Regex labels angles", "angle_label_regex")
        self._entry_row(panel, 1, "Labels angles extra", "extra_angle_labels")
        self._combo_row(panel, 2, "Unité angles C3D", "c3d_angle_unit", ("deg", "rad"))
        self._check(panel, 3, "Exprimer rotations en ZXY", "p6_reexpress_rotations_zxy")
        self._check(
            panel, 4, "Lancer l'IK du système de référence en batch", "p6_run_ik_batch"
        )
        self._entry_row(panel, 5, "Max frames IK batch", "p6_ik_max_frames")
        self._analysis_action_row(tab, 1)
        self._build_graph_panel(tab, 2, "kinematics")

    def _build_gait_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Cycle normalization")
        tab.rowconfigure(3, weight=1)
        panel = ttk.LabelFrame(tab, text="Tâche et détection des cycles")
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(1, weight=1)
        ttk.Label(panel, text="Tâche").grid(row=0, column=0, sticky="w")
        task = ttk.Combobox(
            panel,
            textvariable=self.vars["p6_cycle_task"],
            values=("gait", "squat", "jump"),
            state="readonly",
        )
        task.grid(row=0, column=1, sticky="ew")
        task.bind("<<ComboboxSelected>>", self._cycle_task_changed)
        ttk.Label(panel, text="Détection").grid(row=1, column=0, sticky="w")
        ttk.Combobox(
            panel,
            textvariable=self.vars["p6_cycle_detection"],
            values=("automatic", "reviewed"),
            state="readonly",
        ).grid(row=1, column=1, sticky="ew")
        self._path_row(
            panel, 2, "Événements revus CSV", "p6_gait_events_csv", [("CSV", "*.csv")]
        )
        self.cycle_help = ttk.Label(panel, wraplength=850)
        self.cycle_help.grid(row=3, column=0, columnspan=3, sticky="w", padx=8, pady=4)
        settings = ttk.Frame(panel)
        settings.grid(row=5, column=0, columnspan=3, sticky="ew")
        settings.grid_remove()

        def toggle_settings():
            if settings.winfo_manager():
                settings.grid_remove()
            else:
                settings.grid()

        ttk.Button(
            panel,
            text="Paramètres de détection et normalisation…",
            command=toggle_settings,
        ).grid(row=4, column=0, columnspan=3, sticky="w", padx=8, pady=4)
        for column, entries in enumerate(
            (
                (
                    ("Cycle min. (s)", "p6_gait_min_cycle_s"),
                    ("Cycle max. (s)", "p6_gait_max_cycle_s"),
                    ("Points par cycle", "p6_phase_normalization_points"),
                    ("Lacune max. (s)", "p6_gait_max_gap_s"),
                ),
                (
                    ("Marche : seuil vitesse (m/s)", "p6_gait_velocity_threshold_m_s"),
                    ("Squat : seuil debout (°)", "p6_squat_standing_deg"),
                    ("Squat : flexion minimale (°)", "p6_squat_peak_deg"),
                    ("Saut : seuil hauteur pieds (mm)", "p6_jump_height_mm"),
                ),
            )
        ):
            group = ttk.Frame(settings)
            group.grid(row=0, column=column, sticky="ew")
            settings.columnconfigure(column, weight=1)
            group.columnconfigure(1, weight=1)
            for row, (label, variable) in enumerate(entries):
                self._entry_row(group, row, label, variable)
        actions = ttk.Frame(tab)
        actions.grid(row=1, column=0, sticky="ew", pady=6)
        self._register_analysis_button(
            ttk.Button(
                actions,
                text="Détecter et normaliser",
                command=self._run_selected_gait_analysis,
            )
        ).pack(side=tk.LEFT)
        ttk.Button(
            actions,
            text="Exporter les CSV sélectionnés…",
            command=self._export_gait_csvs,
        ).pack(side=tk.LEFT, padx=8)
        ttk.Button(
            actions, text="Actualiser les cycles", command=self._refresh_cycle_review
        ).pack(side=tk.LEFT)
        ttk.Button(
            actions, text="Ouvrir les 3 figures", command=self._open_cycle_mean_window
        ).pack(side=tk.LEFT, padx=8)
        review = ttk.LabelFrame(
            tab, text="Cocher les cycles à conserver dans les courbes et les CSV"
        )
        review.grid(row=2, column=0, sticky="ew")
        self.cycle_review = ttk.Frame(review)
        self.cycle_review.pack(fill="x", expand=True)
        self.cycle_side_frames = {}
        self.cycle_check_vars = {}
        for column, (side, label) in enumerate(
            (("left", "Gauche — Left"), ("right", "Droite — Right"))
        ):
            group = ttk.LabelFrame(self.cycle_review, text=label)
            group.grid(row=0, column=column, sticky="nsew", padx=4)
            self.cycle_review.columnconfigure(column, weight=1)
            canvas = tk.Canvas(group, height=90, highlightthickness=0)
            scrollbar = ttk.Scrollbar(group, orient="vertical", command=canvas.yview)
            canvas.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side="right", fill="y")
            canvas.pack(side="left", fill="both", expand=True)
            content = ttk.Frame(canvas)
            item = canvas.create_window((0, 0), window=content, anchor="nw")
            content.bind(
                "<Configure>",
                lambda event, c=canvas: c.configure(scrollregion=c.bbox("all")),
            )
            canvas.bind(
                "<Configure>",
                lambda event, c=canvas, i=item: c.itemconfigure(i, width=event.width),
            )
            self.cycle_side_frames[side] = content
        actions_review = ttk.Frame(review)
        actions_review.pack(fill="x")
        ttk.Button(
            actions_review,
            text="Tout cocher",
            command=lambda: self._set_all_cycles(True),
        ).pack(side="left")
        ttk.Button(
            actions_review,
            text="Tout décocher",
            command=lambda: self._set_all_cycles(False),
        ).pack(side="left")
        self.cycle_review_dir = None
        self._cycle_task_changed(reset_limits=False)
        self._build_graph_panel(tab, 3, "gait")

    def _open_cycle_mean_window(self):
        from gui_cycle_means import JOINT_LABELS, draw_joint_figures

        if Figure is None or FigureCanvasTkAgg is None:
            return
        directory = self._cycle_trial_dir()
        report, excluded, _ = cycle_selection(directory)
        if not report or report.get("task", "gait") != str(
            self.vars["p6_cycle_task"].get()
        ):
            messagebox.showinfo(
                "Moyennes", "Normaliser la tâche sélectionnée avant la comparaison."
            )
            return
        windows = {}
        figures = {}
        canvases = {}

        def refresh():
            current, exclusions, _ = cycle_selection(directory)
            try:
                draw_joint_figures(figures, directory, current, exclusions)
                for joint, canvas in canvases.items():
                    if windows[joint].winfo_exists():
                        canvas.draw_idle()
            except (ValueError, OSError, KeyError) as exc:
                messagebox.showerror("Moyennes indisponibles", str(exc), parent=self)

        for index, (joint, label) in enumerate(JOINT_LABELS.items()):
            window = tk.Toplevel(self)
            window.title(f"{directory.name} — {label} — jambe droite")
            window.geometry(f"1350x520+{40+index*35}+{60+index*40}")
            figure = Figure(figsize=(13.5, 5), dpi=100)
            canvas = FigureCanvasTkAgg(figure, master=window)
            ttk.Button(
                window, text="Actualiser avec les cycles cochés", command=refresh
            ).pack(anchor="w", padx=8, pady=6)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            window.figure = figure
            windows[joint] = window
            figures[joint] = figure
            canvases[joint] = canvas
        refresh()
        return windows

    def _hide_cycle_joint_figures(self, panel):
        if "mean_notebook" in panel:
            panel["mean_notebook"].grid_remove()
            panel["canvas"].get_tk_widget().grid()

    def _show_cycle_joint_figures(self, panel, directory, report, excluded, filters):
        from gui_cycle_means import JOINT_LABELS, draw_joint_figures

        if "mean_notebook" not in panel:
            notebook = ttk.Notebook(panel["canvas"].get_tk_widget().master)
            figures = {}
            canvases = {}
            tabs = {}
            for joint, label in JOINT_LABELS.items():
                tab = ttk.Frame(notebook)
                notebook.add(tab, text=label)
                figure = Figure(figsize=(10, 3.5), dpi=100)
                canvas = FigureCanvasTkAgg(figure, master=tab)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                figures[joint] = figure
                canvases[joint] = canvas
                tabs[joint] = tab
            panel.update(
                mean_notebook=notebook,
                mean_figures=figures,
                mean_canvases=canvases,
                mean_tabs=tabs,
            )
        draw_joint_figures(
            panel["mean_figures"],
            directory,
            report,
            excluded,
            side=str(filters.get("side", "right")),
        )
        panel["canvas"].get_tk_widget().grid_remove()
        panel["mean_notebook"].grid(row=0, column=0, sticky="nsew")
        if filters.get("joint") in panel["mean_tabs"]:
            panel["mean_notebook"].select(panel["mean_tabs"][filters["joint"]])
        for canvas in panel["mean_canvases"].values():
            canvas.draw_idle()

    def _cycle_task_changed(self, event=None, *, reset_limits=True):
        task = str(self.vars["p6_cycle_task"].get())
        help_text = {
            "gait": "Ghoussayni : contact au contact suivant du même pied. CSV revu : trial,side,time_s.",
            "squat": "Debout → flexion maximale → debout. Angle géométrique des deux genoux Motive. CSV revu : trial,start_s,end_s.",
            "jump": "Décollage → réception estimés par la hauteur des deux pieds. Vérifier les limites ; pas de validation plateforme de force. CSV revu : trial,start_s,end_s.",
        }
        self.cycle_help.configure(
            text=help_text[task]
            + " Les CSV sont créés uniquement avec le bouton Exporter. Signes fixes selon la référence Fukuchi et al. (2018), appliqués aux courbes et CSV ; composantes non vérifiées conservées."
        )
        if reset_limits:
            self.vars["p6_gait_min_cycle_s"].set("0.1" if task == "jump" else "0.4")
            self.vars["p6_gait_max_cycle_s"].set("10" if task == "squat" else "2.5")
        if hasattr(self, "cycle_review"):
            self._refresh_cycle_review()

    def _cycle_trial_dir(self):
        selected = str(self.vars["selected_trial"].get()).strip()
        return self._graph_output_root() / safe_trial_dir_name(selected)

    def _refresh_cycle_review(self):
        if not hasattr(self, "cycle_review"):
            return
        directory = self._cycle_trial_dir()
        report, excluded, signature = cycle_selection(directory)
        for frame in self.cycle_side_frames.values():
            for widget in frame.winfo_children():
                widget.destroy()
        self.cycle_check_vars = {}
        self.cycle_review_dir = directory
        self.cycle_review_signature = signature
        if report.get("task", "gait") != str(self.vars["p6_cycle_task"].get()):
            return
        selected = selected_cycle_keys(report, excluded)
        for cycle in cycle_options(report):
            key = cycle["selection_key"]
            variable = tk.BooleanVar(self, value=key in selected)
            self.cycle_check_vars[key] = variable
            text = f"Cycle {cycle['cycle_index']}   {cycle['start_s']:.3f}–{cycle['end_s']:.3f} s"
            if "peak_s" in cycle:
                text += f"   pic {cycle['peak_s']:.3f} s"
            ttk.Checkbutton(
                self.cycle_side_frames[cycle["side"]],
                text=text,
                variable=variable,
                command=self._store_cycle_review,
            ).pack(anchor="w", padx=6, pady=2)
        for side, frame in self.cycle_side_frames.items():
            if not any(key.startswith(side + "::") for key in self.cycle_check_vars):
                ttk.Label(frame, text="Aucun cycle détecté").pack(anchor="w", padx=6)

    def _store_cycle_review(self):
        directory = self._cycle_trial_dir()
        _, _, signature = cycle_selection(directory)
        if (
            directory != self.cycle_review_dir
            or signature != self.cycle_review_signature
        ):
            self._refresh_cycle_review()
            return
        excluded = [key for key, var in self.cycle_check_vars.items() if not var.get()]
        save_cycle_selection(directory, excluded, signature)
        self._draw_selected_graph("gait")

    def _set_all_cycles(self, include):
        for variable in self.cycle_check_vars.values():
            variable.set(include)
        if self.cycle_check_vars:
            self._store_cycle_review()

    def _run_selected_gait_analysis(self) -> None:
        selected = str(self.vars["selected_trial"].get()).strip()
        if not selected or selected == ALL_TRIALS_LABEL:
            messagebox.showinfo("Essai", "Sélectionner un essai à normaliser.")
            return
        if self.process is not None or not self._validate_p6_analysis():
            return
        values = self._var_values()
        values["p6_normalize_gait"] = True
        if values.get("p6_cycle_detection") == "automatic":
            values["p6_gait_events_csv"] = ""
        elif not str(values.get("p6_gait_events_csv", "")).strip():
            messagebox.showerror("Événements", "Choisir le CSV des événements revus.")
            return
        self._run_args(
            build_p6_auto_analysis_args(values, selected), command_mode="kinematic"
        )

    def _export_gait_csvs(self) -> None:
        from tkinter import simpledialog

        trial_dir = self._cycle_trial_dir()
        report, excluded, _ = cycle_selection(trial_dir)
        task = str(self.vars["p6_cycle_task"].get())
        if not report or report.get("task", "gait") != task:
            messagebox.showinfo(
                "Export",
                "Normaliser cet essai pour la tâche sélectionnée avant l'export.",
            )
            return
        if not selected_cycle_keys(report, excluded):
            messagebox.showinfo("Export", "Inclure au moins un cycle avant l'export.")
            return
        prefix = simpledialog.askstring(
            "Nom des CSV",
            "Préfixe des fichiers CSV (les suffixes hanche/genou/cheville sont ajoutés) :",
            initialvalue=f"{trial_dir.name}_{task}",
            parent=self,
        )
        if prefix is None:
            return
        destination = filedialog.askdirectory(
            title="Dossier d'export des cycles sélectionnés"
        )
        if destination:
            try:
                files = copy_gait_exports(trial_dir, destination, prefix=prefix)
                self.status_var.set(
                    f"Export terminé : {len(files)-1} CSV + rapport dans {destination}"
                )
            except (OSError, ValueError) as exc:
                messagebox.showerror("Export impossible", str(exc))

    def _build_visualization_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Visualisation")
        panel = ttk.LabelFrame(tab, text="Rerun")
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(1, weight=1)
        self._check(panel, 0, "Visualiser un essai enrichi", "p6_visualize")
        self._entry_row(panel, 1, "Essai visualisé", "p6_visualize_trial")
        self._check(panel, 2, "Headless", "p6_headless")
        self._entry_row(panel, 3, "Attente Rerun", "p6_rerun_wait_seconds")
        self._check(panel, 4, "Ne pas générer les PNG", "p6_no_figures")
        self._analysis_action_row(tab, 1)

    def _build_critical_methods_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Critique")
        tab.rowconfigure(1, weight=1)
        tab.rowconfigure(2, weight=1)

        report_panel = ttk.LabelFrame(tab, text="Dernier rapport d'analyse")
        report_panel.grid(row=0, column=0, sticky="ew")
        report_panel.columnconfigure(0, weight=1)
        self.run_report_summary_var = tk.StringVar(value="Aucun rapport sélectionné.")
        ttk.Label(
            report_panel,
            textvariable=self.run_report_summary_var,
            style="Status.TLabel",
            justify=tk.LEFT,
            wraplength=900,
        ).grid(row=0, column=0, sticky="ew", padx=10, pady=8)
        ttk.Button(
            report_panel,
            text="Actualiser",
            command=self._update_run_report_summary,
        ).grid(row=0, column=1, sticky="ne", padx=(0, 10), pady=8)

        isb_panel = ttk.LabelFrame(tab, text="Conformité ISB - déviations D1 à D6")
        isb_panel.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        isb_panel.rowconfigure(1, weight=1)
        isb_panel.columnconfigure(0, weight=1)

        filters = ttk.Frame(isb_panel)
        filters.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        filters.columnconfigure(1, weight=1)
        filters.columnconfigure(3, weight=1)
        filters.columnconfigure(5, weight=1)
        filters.columnconfigure(7, weight=1)
        self.isb_source_filter_var = tk.StringVar(value="Tous")
        self.isb_deviation_filter_var = tk.StringVar(value="Tous")
        self.isb_status_filter_var = tk.StringVar(value="Tous")
        self.isb_entity_filter_var = tk.StringVar(value="Tous")
        ttk.Label(filters, text="Source").grid(row=0, column=0, sticky="w")
        self.isb_source_filter_combo = ttk.Combobox(
            filters,
            textvariable=self.isb_source_filter_var,
            values=("Tous", *SOURCE_LABELS.values()),
            state="readonly",
            width=24,
        )
        self.isb_source_filter_combo.grid(row=0, column=1, sticky="ew", padx=(6, 12))
        ttk.Label(filters, text="Déviation").grid(row=0, column=2, sticky="w")
        self.isb_deviation_filter_combo = ttk.Combobox(
            filters,
            textvariable=self.isb_deviation_filter_var,
            values=("Tous", "D1", "D2", "D3", "D4", "D5", "D6"),
            state="readonly",
            width=10,
        )
        self.isb_deviation_filter_combo.grid(row=0, column=3, sticky="ew", padx=(6, 12))
        ttk.Label(filters, text="Statut").grid(row=0, column=4, sticky="w")
        self.isb_status_filter_combo = ttk.Combobox(
            filters,
            textvariable=self.isb_status_filter_var,
            values=("Tous",),
            state="readonly",
            width=18,
        )
        self.isb_status_filter_combo.grid(row=0, column=5, sticky="ew", padx=(6, 12))
        ttk.Label(filters, text="Cible").grid(row=0, column=6, sticky="w")
        self.isb_entity_filter_combo = ttk.Combobox(
            filters,
            textvariable=self.isb_entity_filter_var,
            values=("Tous",),
            state="readonly",
            width=18,
        )
        self.isb_entity_filter_combo.grid(row=0, column=7, sticky="ew", padx=(6, 0))
        for combo in (
            self.isb_source_filter_combo,
            self.isb_deviation_filter_combo,
            self.isb_status_filter_combo,
            self.isb_entity_filter_combo,
        ):
            combo.bind("<<ComboboxSelected>>", self._refresh_isb_report_table)

        table_frame = ttk.Frame(isb_panel)
        table_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 4))
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        columns = (
            "source",
            "format",
            "trial",
            "entity",
            "deviation",
            "status",
            "confidence",
            "blocking",
        )
        self.isb_report_tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", height=9
        )
        headings = (
            "Source",
            "Format",
            "Essai",
            "Segment / articulation",
            "D",
            "Statut",
            "Confiance",
            "Blocage",
        )
        widths = (150, 70, 90, 155, 40, 105, 120, 210)
        for column, heading, width in zip(columns, headings, widths, strict=True):
            self.isb_report_tree.heading(column, text=heading)
            self.isb_report_tree.column(column, width=width, minwidth=40, stretch=True)
        y_scroll = ttk.Scrollbar(
            table_frame, orient=tk.VERTICAL, command=self.isb_report_tree.yview
        )
        x_scroll = ttk.Scrollbar(
            table_frame, orient=tk.HORIZONTAL, command=self.isb_report_tree.xview
        )
        self.isb_report_tree.configure(
            yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set
        )
        self.isb_report_tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.isb_report_status_var = tk.StringVar(
            value="Rapport D1-D6 non encore généré."
        )
        ttk.Label(
            isb_panel,
            textvariable=self.isb_report_status_var,
            style="Status.TLabel",
        ).grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))

        panel = ttk.LabelFrame(tab, text="Algorithmes sensibles et contrôles")
        panel.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        panel.rowconfigure(0, weight=1)
        panel.columnconfigure(0, weight=1)

        text = tk.Text(
            panel,
            wrap=tk.WORD,
            height=10,
            font=("TkDefaultFont", 10),
            relief=tk.FLAT,
            background="#ffffff",
            foreground="#1f2933",
        )
        scrollbar = ttk.Scrollbar(panel, orient=tk.VERTICAL, command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        text.grid(row=0, column=0, sticky="nsew", padx=(10, 0), pady=10)
        scrollbar.grid(row=0, column=1, sticky="ns", pady=10)
        text.tag_configure("title", font=("TkDefaultFont", 11, "bold"))
        text.tag_configure("label", font=("TkDefaultFont", 10, "bold"))

        for index, note in enumerate(CRITICAL_METHOD_NOTES, start=1):
            if index > 1:
                text.insert(tk.END, "\n")
            text.insert(tk.END, f"{index}. {note['title']}\n", "title")
            text.insert(tk.END, "Algorithme: ", "label")
            text.insert(tk.END, f"{note['algorithm']}\n")
            text.insert(tk.END, "Risque: ", "label")
            text.insert(tk.END, f"{note['risk']}\n")
            text.insert(tk.END, "À vérifier: ", "label")
            text.insert(tk.END, f"{note['check']}\n")
        text.configure(state=tk.DISABLED)

    def _build_sources_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Sources")
        card = ttk.LabelFrame(tab, text="Fichiers")
        card.grid(row=0, column=0, sticky="ew")
        card.columnconfigure(1, weight=1)

        self._path_row(
            card, 0, "BVH", "bvh", [("BVH", "*.bvh"), ("Tous les fichiers", "*")]
        )
        self._path_row(
            card, 1, "FBX", "fbx", [("FBX", "*.fbx"), ("Tous les fichiers", "*")]
        )
        self._path_row(
            card, 2, "C3D", "c3d", [("C3D", "*.c3d"), ("Tous les fichiers", "*")]
        )
        self._path_row(card, 3, "Sortie", "out_dir", directory=True)

        tips = ttk.LabelFrame(tab, text="Flux")
        tips.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(
            tips,
            text=(
                "BVH + C3D suffit pour générer le modèle BVH et les marqueurs locaux. "
                "Ajoute un FBX pour générer le modèle surfacique et l'affichage superposé."
            ),
            wraplength=760,
        ).grid(row=0, column=0, sticky="w", padx=10, pady=8)

        kinematic = ttk.LabelFrame(tab, text="Données cinématiques multi-systèmes")
        kinematic.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        kinematic.columnconfigure(1, weight=1)
        self._path_row(kinematic, 0, "Dossier racine", "p6_data_root", directory=True)
        self._entry_row(kinematic, 1, "Essai statique", "p6_static_trial")

    def _build_model_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Modèles")
        units = ttk.LabelFrame(tab, text="Unités et angles")
        units.grid(row=0, column=0, sticky="ew")
        units.columnconfigure(1, weight=1)
        self._entry_row(units, 0, "Échelle BVH vers m", "bvh_unit_scale_to_m")
        self._entry_row(units, 1, "Échelle FBX vers m", "fbx_unit_scale_to_m")
        self._combo_row(units, 2, "Unité angles C3D", "c3d_angle_unit", ("deg", "rad"))
        self._entry_row(units, 3, "Regex labels angles", "angle_label_regex")
        self._entry_row(units, 4, "Labels angles extra", "extra_angle_labels")
        self._path_row(
            units,
            5,
            "Mapping q/C3D",
            "comparison_map",
            [("JSON", "*.json"), ("Tous les fichiers", "*")],
        )

        generation = ttk.LabelFrame(tab, text="Génération")
        generation.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        generation.columnconfigure(1, weight=1)
        self._combo_row(
            generation,
            0,
            "Offset trans. racine",
            "root_offset_mode",
            ROOT_OFFSET_MODE_CHOICES,
        )
        self._entry_row(generation, 1, "Max points mesh FBX", "max_fbx_mesh_points")
        self._check(
            generation,
            2,
            "Ne pas ajouter les marqueurs de centres articulaires",
            "no_biomod_joint_centre_markers",
        )
        self._check(
            generation,
            3,
            "Ne pas corriger le root offset (compatibilité)",
            "no_root_offset_correction",
        )
        self._check(generation, 4, "Ne pas générer les meshes FBX", "no_fbx_mesh")

        chain_compare = ttk.LabelFrame(tab, text="Chaînes cinématiques comparées")
        chain_compare.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        chain_compare.columnconfigure(1, weight=1)
        self._combo_row(
            chain_compare,
            0,
            "Source modèle",
            "p6_model_source",
            ("bvh", "fbx", "auto"),
        )
        self._combo_row(
            chain_compare,
            1,
            "Captury modèle -> C3D",
            "p6_captury_model_to_c3d_axis",
            ("auto", "y_up_to_z_up", "identity"),
        )
        self._combo_row(
            chain_compare,
            2,
            "Motive modèle -> C3D",
            "p6_motive_model_to_c3d_axis",
            ("auto", "y_up_to_z_up", "identity"),
        )
        self._combo_row(
            chain_compare,
            3,
            "Offset trans. racine",
            "root_offset_mode",
            ROOT_OFFSET_MODE_CHOICES,
        )
        self._check(
            chain_compare,
            4,
            "Désactiver recalage statique Captury -> Motive",
            "p6_disable_static_model_alignment",
        )
        self._check(
            chain_compare,
            5,
            "Désactiver recalage Motive -> marqueurs C3D",
            "p6_disable_motive_marker_alignment",
        )
        self._check(
            chain_compare,
            6,
            "Captury: axe Y cuisse = hanche -> genou",
            "p6_captury_reorient_thigh_y_from_cor",
        )
        self._check(
            chain_compare,
            7,
            "R(x,180°)R(y,180°)",
            "p6_rotate_body_segments_180_x",
        )
        self._check(chain_compare, 8, "Ne pas extraire les meshes FBX", "p6_no_mesh")
        self._entry_row(chain_compare, 9, "Max points mesh", "p6_max_mesh_points")
        ttk.Label(
            chain_compare,
            text=(
                "Utilise BVH/FBX pour construire les modèles BioBuddy/biorbd des deux systèmes. "
                "En auto, Captury reste +Y -> +Y dans son C3D et Motive passe +Y -> +Z; "
                "les menus séparés sont réservés aux diagnostics source par source. "
                "Root offset auto compare keep/subtract avec le C3D et retient la meilleure superposition."
            ),
            style="Status.TLabel",
            wraplength=760,
        ).grid(row=10, column=0, columnspan=3, sticky="w", padx=10, pady=(4, 10))

        explorer = ttk.LabelFrame(tab, text="Explorateur BioBuddy")
        explorer.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        explorer.columnconfigure(1, weight=1)
        self._path_row(
            explorer,
            0,
            "Modèle",
            "model_explorer_path",
            [
                ("Modèles BioBuddy", "*.bioMod *.osim *.urdf *.bvh"),
                ("Tous les fichiers", "*"),
            ],
        )
        ttk.Button(
            explorer,
            text="BVH généré",
            command=lambda: self._set_generated_model_path("bvh"),
        ).grid(row=1, column=0, sticky="ew", padx=(10, 4), pady=(0, 10))
        ttk.Button(
            explorer,
            text="FBX généré",
            command=lambda: self._set_generated_model_path("fbx"),
        ).grid(row=1, column=1, sticky="ew", padx=4, pady=(0, 10))
        ttk.Button(
            explorer,
            text="Ouvrir dans BioBuddy",
            command=self._launch_biobuddy_model_explorer,
        ).grid(row=1, column=2, sticky="ew", padx=(4, 10), pady=(0, 10))

    def _build_rerun_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Rerun")
        launch = ttk.LabelFrame(tab, text="Visualisation")
        launch.grid(row=0, column=0, sticky="ew")
        launch.columnconfigure(1, weight=1)
        self._check(launch, 0, "Lancer une scène Rerun par modèle", "animate")
        self._check(
            launch,
            1,
            "Lancer la scène superposée BVH + FBX + C3D",
            "animate_superposed",
        )
        self._check(
            launch, 2, "Afficher les courbes q dans Rerun", "display_q_in_rerun"
        )
        self._check(launch, 3, "Mode headless", "headless")
        self._entry_row(launch, 4, "Rayon marqueurs", "rerun_marker_radius")
        self._entry_row(launch, 5, "Attente après envoi", "rerun_wait_seconds")
        self._combo_row(
            launch, 6, "Axe vertical", "rerun_up_axis", ("y", "z", "x", "none")
        )

        filters = ttk.LabelFrame(tab, text="Lisibilité")
        filters.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        self._check(filters, 0, "Masquer mains/poignets/doigts", "hide_hands_in_rerun")
        self._check(filters, 1, "Masquer pieds/chevilles/orteils", "hide_feet_in_rerun")
        self._check(
            filters, 2, "Masquer toutes les extrémités", "hide_extremities_in_rerun"
        )

        kinematic = ttk.LabelFrame(
            tab, text="Visualisation des comparaisons cinématiques"
        )
        kinematic.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        kinematic.columnconfigure(1, weight=1)
        self._check(kinematic, 0, "Visualiser un essai enrichi", "p6_visualize")
        self._entry_row(kinematic, 1, "Essai visualisé", "p6_visualize_trial")
        self._check(kinematic, 2, "Headless", "p6_headless")
        self._entry_row(kinematic, 3, "Attente Rerun", "p6_rerun_wait_seconds")

    def _build_ik_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "IK")
        ik = ttk.LabelFrame(tab, text="Cinématique inverse")
        ik.grid(row=0, column=0, sticky="ew")
        ik.columnconfigure(1, weight=1)
        self._check(
            ik,
            0,
            "Lancer la cinématique inverse depuis les marqueurs C3D",
            "inverse_kinematics",
        )
        self._combo_row(
            ik, 1, "Solveur", "inverse_kinematics_solver", ("least_squares", "kalman")
        )
        self._combo_row(
            ik,
            2,
            "Méthode least-squares",
            "inverse_kinematics_method",
            ("trf", "lm", "only_lm"),
        )
        self._entry_row(ik, 3, "Nombre max de frames", "inverse_kinematics_max_frames")
        self._entry_row(ik, 4, "Kalman noise factor", "kalman_noise_factor")
        self._entry_row(ik, 5, "Kalman error factor", "kalman_error_factor")

        ttk.Label(
            ik,
            text="0 frame max signifie que toutes les frames du C3D sont utilisées.",
            style="Status.TLabel",
        ).grid(row=6, column=0, columnspan=3, sticky="w", padx=10, pady=(4, 10))

        batch = ttk.LabelFrame(tab, text="Batch sur essais cinématiques")
        batch.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        batch.columnconfigure(1, weight=1)
        self._check(
            batch, 0, "Lancer l'IK du système de référence en batch", "p6_run_ik_batch"
        )
        self._entry_row(batch, 1, "Max frames IK batch", "p6_ik_max_frames")

    def _build_comparison_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Comparaison")
        batch = ttk.LabelFrame(tab, text="Mode dossier ou population")
        batch.grid(row=0, column=0, sticky="ew")
        batch.columnconfigure(1, weight=1)
        self._path_row(batch, 0, "Racine", "compare_data_root", directory=True)
        self._entry_row(batch, 1, "Système référence", "compare_reference_system")
        self._entry_row(batch, 2, "Système test", "compare_test_system")
        self._entry_row(batch, 3, "Filtre participants", "compare_participant_filter")
        self._entry_row(batch, 4, "Filtre essais", "compare_trial_filter")

        single = ttk.LabelFrame(tab, text="Mode paire C3D explicite")
        single.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        single.columnconfigure(1, weight=1)
        self._path_row(
            single,
            0,
            "Référence C3D",
            "compare_reference_c3d",
            [("C3D", "*.c3d"), ("Tous les fichiers", "*")],
        )
        self._path_row(
            single,
            1,
            "Référence BVH",
            "compare_reference_bvh",
            [("BVH", "*.bvh"), ("Tous les fichiers", "*")],
        )
        self._path_row(
            single,
            2,
            "Référence FBX",
            "compare_reference_fbx",
            [("FBX", "*.fbx"), ("Tous les fichiers", "*")],
        )
        self._path_row(
            single,
            3,
            "Test C3D",
            "compare_test_c3d",
            [("C3D", "*.c3d"), ("Tous les fichiers", "*")],
        )
        self._path_row(
            single,
            4,
            "Test BVH",
            "compare_test_bvh",
            [("BVH", "*.bvh"), ("Tous les fichiers", "*")],
        )
        self._path_row(
            single,
            5,
            "Test FBX",
            "compare_test_fbx",
            [("FBX", "*.fbx"), ("Tous les fichiers", "*")],
        )
        self._entry_row(single, 6, "Nom essai", "compare_trial_name")

        options = ttk.LabelFrame(tab, text="Options")
        options.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        options.columnconfigure(1, weight=1)
        self._path_row(options, 0, "Sortie", "compare_out_dir", directory=True)
        self._path_row(
            options,
            1,
            "Carte repères",
            "compare_landmark_map",
            [("JSON", "*.json"), ("Tous les fichiers", "*")],
        )
        self._entry_row(options, 2, "Points normalisés", "compare_resample_points")
        self._combo_row(
            options,
            3,
            "Alignement",
            "compare_alignment",
            ("global_rigid", "per_frame_rigid", "none"),
        )

        centres = ttk.LabelFrame(tab, text="Centres articulaires et cinématiques")
        centres.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        centres.columnconfigure(1, weight=1)
        self._entry_row(centres, 0, "Filtre centres", "p6_joint_filter")
        self._check(centres, 1, "Ne pas générer les PNG", "p6_no_figures")
        ttk.Label(
            centres,
            text=(
                "Les centres issus des modèles sont ajoutés dans des copies du C3D de référence, "
                "puis comparés après alignement statique du système test vers le système de référence. "
                "Les rotations Euler sont reportées comme indicatives lorsque les conventions diffèrent."
            ),
            style="Status.TLabel",
            wraplength=760,
        ).grid(row=2, column=0, columnspan=3, sticky="w", padx=10, pady=(4, 10))

        actions = ttk.Frame(tab)
        actions.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        actions.columnconfigure(2, weight=1)
        actions.columnconfigure(3, weight=1)
        self.compare_button = self._register_analysis_button(
            ttk.Button(
                actions,
                text="Lancer la comparaison",
                style="Primary.TButton",
                command=self._run_comparison,
            )
        )
        self.compare_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(
            actions,
            text="Copier commande comparaison",
            command=self._copy_comparison_command,
        ).grid(row=0, column=1, sticky="ew", padx=6)
        self.p6_button = self._register_analysis_button(
            ttk.Button(
                actions,
                text="Lancer analyse cinématique",
                style="Primary.TButton",
                command=self._run_p6_analysis,
            )
        )
        self.p6_button.grid(row=0, column=2, sticky="ew", padx=6)
        ttk.Button(
            actions, text="Copier commande cinématique", command=self._copy_p6_command
        ).grid(row=0, column=3, sticky="ew", padx=(6, 0))

    def _build_advanced_tab(self, notebook: ttk.Notebook) -> None:
        tab = self._tab(notebook, "Avancé")
        legacy = ttk.LabelFrame(tab, text="Compatibilité ancienne CLI")
        legacy.grid(row=0, column=0, sticky="ew")
        legacy.columnconfigure(1, weight=1)
        self._check(
            legacy, 0, "Utiliser --inverse-dynamics (déprécié)", "inverse_dynamics"
        )
        self._combo_row(
            legacy,
            1,
            "Méthode inverse dynamics",
            "inverse_dynamics_method",
            ("", "trf", "lm", "only_lm"),
        )
        self._entry_row(
            legacy, 2, "Max frames inverse dynamics", "inverse_dynamics_max_frames"
        )

        environment = ttk.LabelFrame(tab, text="Exécution")
        environment.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(environment, text=f"Python: {sys.executable}", wraplength=760).grid(
            row=0, column=0, sticky="w", padx=10, pady=(8, 4)
        )
        ttk.Label(environment, text=f"Script: {PIPELINE_SCRIPT}", wraplength=760).grid(
            row=1, column=0, sticky="w", padx=10, pady=(0, 8)
        )
        self._check(environment, 2, "Ignorer le cache", "p6_no_cache")
        self._check(environment, 3, "Analyse auto au choix d'essai", "p6_auto_analyze")

        diagnostics = ttk.LabelFrame(tab, text="Diagnostic recalage")
        diagnostics.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self._check(
            diagnostics,
            0,
            "Désactiver recalage statique Captury -> Motive",
            "p6_disable_static_model_alignment",
        )
        self._check(
            diagnostics,
            1,
            "Désactiver recalage Motive -> marqueurs C3D",
            "p6_disable_motive_marker_alignment",
        )

    def _build_actions(self, parent: ttk.Frame) -> None:
        actions = ttk.LabelFrame(parent, text="Exécution")
        actions.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for column in range(3):
            actions.columnconfigure(column, weight=1)

        self.run_button = self._register_analysis_button(
            ttk.Button(
                actions,
                text="Lancer",
                style="Primary.TButton",
                command=self._run_selected_command,
            )
        )
        self.run_button.grid(row=0, column=0, sticky="ew", padx=(8, 4), pady=(8, 8))
        self.stop_button = ttk.Button(
            actions,
            text="Arrêter",
            style="Danger.TButton",
            command=self._stop_pipeline,
            state=tk.DISABLED,
        )
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=4, pady=(8, 8))
        ttk.Button(actions, text="Ouvrir sortie", command=self._open_output_dir).grid(
            row=0, column=2, sticky="ew", padx=(4, 8), pady=(8, 8)
        )

        self.status_var = tk.StringVar(value="Prêt")
        ttk.Label(actions, textvariable=self.status_var, style="Status.TLabel").grid(
            row=1, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 8)
        )

    def _build_embedded_trial_viewer(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Visu 3D essai")
        panel.grid(row=1, column=0, sticky="nsew")
        panel.rowconfigure(3, weight=1)
        panel.columnconfigure(0, weight=1)

        header = ttk.Frame(panel)
        header.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        header.columnconfigure(0, weight=1)
        self.viewer_path_var = tk.StringVar(value="Aucun C3D")
        ttk.Label(
            header, textvariable=self.viewer_path_var, style="Status.TLabel"
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            header, text="Grande fenêtre", command=self._open_selected_trial_viewer
        ).grid(row=0, column=1, sticky="e", padx=(8, 0))

        view_bar = ttk.Frame(panel)
        view_bar.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 4))
        view_bar.columnconfigure(1, weight=1)
        ttk.Label(view_bar, text="Vue").grid(row=0, column=0, sticky="w")
        self.viewer_view_var = tk.StringVar(value="Face")
        self.viewer_view_combo = ttk.Combobox(
            view_bar,
            textvariable=self.viewer_view_var,
            values=("Face", "Dos", "Côté", "XY", "YZ", "XZ"),
            state="readonly",
            width=16,
        )
        self.viewer_view_combo.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self.viewer_view_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self._apply_embedded_view()
        )
        layer_controls = ttk.Frame(panel)
        layer_controls.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 4))
        ttk.Label(layer_controls, text="Marqueurs").grid(row=0, column=0, sticky="w")
        self.viewer_marker_layer_vars: dict[str, tk.BooleanVar] = {}
        for column, layer in enumerate(("captury", "motive"), start=1):
            var = tk.BooleanVar(value=True)
            self.viewer_marker_layer_vars[layer] = var
            ttk.Checkbutton(
                layer_controls,
                text=COR_LAYER_LABELS[layer],
                variable=var,
                command=self._update_visible_marker_layers,
                style=f"{layer.title()}.TCheckbutton",
            ).grid(row=0, column=column, sticky="w", padx=(8, 0))

        ttk.Label(layer_controls, text="Chaîne CoR").grid(
            row=1, column=0, sticky="w", pady=(4, 0)
        )
        self.viewer_cor_layer_vars: dict[str, tk.BooleanVar] = {}
        self.viewer_cor_layer_checks: dict[str, ttk.Checkbutton] = {}
        for column, layer in enumerate(("captury", "motive", "biobuddy"), start=1):
            var = tk.BooleanVar(value=layer in {"captury", "motive"})
            self.viewer_cor_layer_vars[layer] = var
            checkbutton = ttk.Checkbutton(
                layer_controls,
                text=COR_LAYER_LABELS[layer],
                variable=var,
                command=self._update_visible_cor_layers,
                style=f"{layer.title()}.TCheckbutton",
            )
            checkbutton.grid(row=1, column=column, sticky="w", padx=(8, 0), pady=(4, 0))
            self.viewer_cor_layer_checks[layer] = checkbutton
        self.viewer_chain_axes_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            layer_controls,
            text="Repères XYZ",
            variable=self.viewer_chain_axes_var,
            command=self._update_visible_cor_layers,
        ).grid(row=1, column=4, sticky="w", padx=(8, 0), pady=(4, 0))

        self.embedded_viewer = TkC3DTrialCanvas(panel)
        self.embedded_viewer.grid(row=3, column=0, sticky="nsew", padx=8, pady=4)
        ttk.Label(view_bar, text="Maj + glisser : déplacer • Molette : zoom").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )
        camera_controls = ttk.Frame(view_bar)
        camera_controls.grid(row=2, column=0, columnspan=2, sticky="w")
        for label, action in (("−", lambda: self.embedded_viewer._zoom(1 / 1.25)),
                              ("+", lambda: self.embedded_viewer._zoom(1.25)),
                              ("Recentrer", self.embedded_viewer.reset_camera)):
            ttk.Button(camera_controls, text=label, command=action).pack(side="left", padx=2)

        controls = ttk.Frame(panel)
        controls.grid(row=4, column=0, sticky="ew", padx=8, pady=(4, 8))
        controls.columnconfigure(1, weight=1)
        self.viewer_playing = tk.BooleanVar(value=False)
        self.viewer_play_button = ttk.Button(
            controls, text="▶", width=3, command=self._toggle_embedded_viewer_play
        )
        self.viewer_play_button.grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.viewer_frame_var = tk.IntVar(value=0)
        self.viewer_frame_slider = ttk.Scale(
            controls,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            variable=self.viewer_frame_var,
            command=self._on_embedded_viewer_slider,
        )
        self.viewer_frame_slider.grid(row=0, column=1, sticky="ew")
        self.viewer_frame_label_var = tk.StringVar(value="0 / 0")
        ttk.Label(controls, textvariable=self.viewer_frame_label_var, width=12).grid(
            row=0, column=2, sticky="e", padx=(8, 0)
        )

    def _build_footer_tools(self, parent: ttk.Frame) -> None:
        footer = ttk.Frame(parent)
        footer.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        footer.columnconfigure(2, weight=1)
        ttk.Button(footer, text="Commande", command=self._open_command_window).grid(
            row=0, column=0, sticky="w", padx=(0, 6)
        )
        ttk.Button(footer, text="Log", command=self._open_log_window).grid(
            row=0, column=1, sticky="w"
        )

    def _build_log_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Log")
        panel.grid(row=2, column=0, sticky="nsew")
        panel.rowconfigure(0, weight=1)
        panel.columnconfigure(0, weight=1)

        self.log_text = tk.Text(
            panel, wrap=tk.WORD, font=("Menlo", 11), state=tk.DISABLED
        )
        scrollbar = ttk.Scrollbar(
            panel, orient=tk.VERTICAL, command=self.log_text.yview
        )
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

    def _build_occlusion_table_panel(self, parent: ttk.Widget, row: int) -> None:
        panel = ttk.LabelFrame(parent, text="Tableau des occlusions")
        panel.grid(row=row, column=0, sticky="nsew", pady=(12, 0))
        panel.rowconfigure(0, weight=1)
        panel.columnconfigure(0, weight=1)

        table_frame = ttk.Frame(panel)
        table_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        self.occlusion_table = ttk.Treeview(
            table_frame,
            columns=("marker", "missing_percent", "missing_frames", "total_frames"),
            show="headings",
            height=14,
        )
        self._set_occlusion_table_headings()
        self.occlusion_table.column("marker", width=180, stretch=True)
        self.occlusion_table.column("missing_percent", width=110, anchor=tk.E)
        self.occlusion_table.column("missing_frames", width=140, anchor=tk.E)
        self.occlusion_table.column("total_frames", width=120, anchor=tk.E)
        self.occlusion_table.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(
            table_frame, orient=tk.VERTICAL, command=self.occlusion_table.yview
        )
        self.occlusion_table.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="ns")

    def _build_graph_panel(self, parent: ttk.Widget, row: int, graph_kind: str) -> None:
        title = (
            "Découpage et contacts"
            if graph_kind == "events"
            else str(GRAPH_CONFIGS[graph_kind]["title"])
        )
        panel = ttk.LabelFrame(parent, text=f"Graphiques - {title}")
        panel.grid(row=row, column=0, sticky="nsew", pady=(12, 0))
        panel.rowconfigure(0, weight=1)
        panel.columnconfigure(1, weight=1)

        tree_frame = ttk.Frame(panel)
        tree_frame.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        tree = ttk.Treeview(tree_frame, show="tree", height=10, selectmode="extended")
        tree.grid(row=0, column=0, sticky="nsew")
        tree_scrollbar = ttk.Scrollbar(
            tree_frame, orient=tk.VERTICAL, command=tree.yview
        )
        tree.configure(yscrollcommand=tree_scrollbar.set)
        tree_scrollbar.grid(row=0, column=1, sticky="ns")
        tree.bind(
            "<<TreeviewSelect>>",
            lambda _event, kind=graph_kind: self._draw_selected_graph(kind),
        )
        tree.bind(
            "<Button-1>",
            lambda event, kind=graph_kind: self._toggle_graph_tree_selection(
                event, kind
            ),
        )

        graph_frame = ttk.Frame(panel)
        graph_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        graph_frame.rowconfigure(0, weight=1)
        graph_frame.columnconfigure(0, weight=1)

        panel_data: dict[str, object] = {"tree": tree}
        if (
            EMBEDDED_GRAPHS_AVAILABLE
            and Figure is not None
            and FigureCanvasTkAgg is not None
        ):
            figure = Figure(figsize=(6.2, 3.2), dpi=100)
            if graph_kind == "centres":
                axes_pair = figure.subplots(1, 2, squeeze=False)[0]
                axes = axes_pair[0]
            else:
                axes_pair = None
                axes = figure.add_subplot(111)
            canvas = FigureCanvasTkAgg(figure, master=graph_frame)
            canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
            panel_data.update({"figure": figure, "axes": axes, "canvas": canvas})
            if axes_pair is not None:
                panel_data["axes_pair"] = tuple(axes_pair)
            if graph_kind == "events":
                canvas.mpl_connect("button_press_event", self._on_phase_drag_press)
                canvas.mpl_connect("motion_notify_event", self._on_phase_drag_motion)
                canvas.mpl_connect("button_release_event", self._on_phase_drag_release)
        else:
            ttk.Label(
                graph_frame,
                text="Matplotlib/pandas indisponibles pour les graphiques intégrés.",
                style="Status.TLabel",
            ).grid(row=0, column=0, sticky="nsew")
        self.graph_panels[graph_kind] = panel_data
        self.graph_payloads[graph_kind] = {}

    def _refresh_graphs(self) -> None:
        for graph_kind in self.graph_panels:
            self._populate_graph_tree(graph_kind)

    def _refresh_results(self) -> None:
        self._populate_occlusion_table()
        self._refresh_graphs()
        self._update_embedded_trial_viewer()
        self._auto_enable_biobuddy_cor_layer_after_refresh()
        self._update_run_report_summary()
        self._refresh_isb_report_table()

    def _invalidate_output_caches(self) -> None:
        self.joint_chain_cache.clear()

    def _refresh_marker_correspondence_lists(self) -> None:
        if not hasattr(self, "marker_motive_list"):
            return
        c3d_paths = self._selected_trial_c3d_paths()
        self.marker_list_label_lookup = {"motive": {}, "captury": {}}
        for source, listbox in (
            ("motive", self.marker_motive_list),
            ("captury", self.marker_captury_list),
        ):
            listbox.delete(0, tk.END)
            path = c3d_paths.get(source.title())
            if path is None:
                continue
            try:
                data = self._load_cached_c3d_marker_data(path)
                if source == "captury":
                    data = self._captury_marker_display_data(data)
            except Exception as exc:
                self._append_log(f"\nListe marqueurs impossible pour {path}: {exc}\n")
                continue
            raw_display_labels = marker_display_labels(data.labels)
            visual_display_labels = (
                data.display_labels
                if data.display_labels is not None
                and len(data.display_labels) == len(data.labels)
                else raw_display_labels
            )
            pairs = sorted(
                (visual_label, raw_label)
                for label, visual_label, raw_label in zip(
                    data.labels,
                    visual_display_labels,
                    raw_display_labels,
                    strict=True,
                )
                if not is_joint_centre_marker_label(str(label))
            )
            display_labels = [display for display, _raw in pairs]
            raw_by_display = {display: raw for display, raw in pairs}
            self.marker_list_label_lookup[source] = raw_by_display
            for display_label in display_labels:
                listbox.insert(tk.END, display_label)
        if hasattr(self, "embedded_viewer"):
            self.embedded_viewer.set_selected_markers({})

    def _selected_listbox_value(self, listbox: tk.Listbox) -> str | None:
        selection = listbox.curselection()
        if not selection:
            return None
        return str(listbox.get(selection[0]))

    def _selected_marker_raw_label(self, source: str, display_label: str) -> str:
        lookup = getattr(self, "marker_list_label_lookup", {})
        return str(lookup.get(source, {}).get(display_label, display_label))

    def _highlight_marker_pair_selection(self) -> None:
        if not hasattr(self, "embedded_viewer"):
            return
        motive_label = self._selected_listbox_value(self.marker_motive_list)
        captury_label = self._selected_listbox_value(self.marker_captury_list)
        selected: dict[str, list[str]] = {}
        if motive_label:
            selected["motive"] = [
                self._selected_marker_raw_label("motive", motive_label)
            ]
        if captury_label:
            selected["captury"] = [
                self._selected_marker_raw_label("captury", captury_label)
            ]
        self.embedded_viewer.set_selected_markers(selected)

    def _marker_pair_rows_from_tree(self) -> list[dict[str, object]]:
        if not hasattr(self, "marker_pair_tree"):
            return []
        rows: list[dict[str, object]] = []
        for item_id in self.marker_pair_tree.get_children():
            rows.append(
                tree_values_to_payload(self.marker_pair_tree.item(item_id, "values"))
            )
        return rows

    def _set_marker_pair_rows(self, rows: Iterable[dict[str, object]]) -> None:
        if not hasattr(self, "marker_pair_tree"):
            return
        self.marker_pair_tree.delete(*self.marker_pair_tree.get_children())
        for row in rows:
            self.marker_pair_tree.insert(
                "",
                tk.END,
                values=payload_to_tree_values(row),
            )

    def _add_marker_correspondence_pair(self) -> None:
        motive_label = self._selected_listbox_value(self.marker_motive_list)
        captury_label = self._selected_listbox_value(self.marker_captury_list)
        if not motive_label or not captury_label:
            messagebox.showerror(
                "Marqueurs manquants",
                "Choisir un marqueur Motive et un marqueur Captury.",
            )
            return
        existing = {
            key
            for row in self._marker_pair_rows_from_tree()
            if (key := marker_pair_key(row)) is not None
        }
        pair = (motive_label, captury_label)
        if pair in existing:
            return
        self.marker_pair_tree.insert(
            "",
            tk.END,
            values=payload_to_tree_values(
                marker_pair_to_payload(motive_label, captury_label)
            ),
        )
        self.status_var.set(f"Paire ajoutée: {motive_label} / {captury_label}")

    def _remove_selected_marker_correspondence_pair(self) -> None:
        if not hasattr(self, "marker_pair_tree"):
            return
        selection = self.marker_pair_tree.selection()
        if selection:
            self.marker_pair_tree.delete(*selection)

    def _marker_correspondence_json_path(self) -> Path:
        raw_path = str(self.vars["compare_landmark_map"].get()).strip()
        if not raw_path:
            raw_path = "motive_captury_landmark_map.json"
            self.vars["compare_landmark_map"].set(raw_path)
        return self._resolve(raw_path)

    def _save_marker_correspondence_json(self) -> Path | None:
        rows = self._marker_pair_rows_from_tree()
        if not rows:
            messagebox.showerror(
                "Liste vide", "Ajouter au moins une correspondance marqueur."
            )
            return None
        path = self._marker_correspondence_json_path()
        save_marker_correspondence_payload(path, rows)
        self.status_var.set(f"Correspondances enregistrées: {path.name}")
        return path

    def _load_marker_correspondence_json(self) -> None:
        path = self._marker_correspondence_json_path()
        if not path.exists():
            messagebox.showerror("JSON introuvable", str(path))
            return
        self._load_marker_correspondence_payload_from_path(path)

    def _selected_marker_correspondence_proposal_path(self) -> Path | None:
        selected = str(self.vars["selected_trial"].get()).strip()
        if not selected or selected == ALL_TRIALS_LABEL:
            return None
        return (
            self._graph_output_root()
            / safe_trial_dir_name(selected)
            / "skin_marker_correspondence_proposal.json"
        )

    def _load_marker_correspondence_proposal_json(self) -> None:
        path = self._selected_marker_correspondence_proposal_path()
        if path is None:
            messagebox.showerror(
                "Essai non sélectionné",
                "Choisir un essai précis pour charger sa proposition.",
            )
            return
        if not path.exists():
            messagebox.showerror(
                "Proposition introuvable",
                f"Lancer l'analyse de l'essai avant de charger:\n{path}",
            )
            return
        self._load_marker_correspondence_payload_from_path(path)

    def _load_marker_correspondence_payload_from_path(self, path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            messagebox.showerror("JSON invalide", str(exc))
            return
        if not isinstance(payload, list):
            messagebox.showerror("JSON invalide", "Le fichier doit contenir une liste.")
            return
        self._set_marker_pair_rows(payload)
        self.status_var.set(f"Correspondances chargées: {path.name}")

    def _save_marker_pairs_and_run_analysis(self) -> None:
        if self._save_marker_correspondence_json() is None:
            return
        self._run_selected_trial_auto_analysis()

    def _c3d_cache_key(self, path: Path) -> tuple[str, int, int, str]:
        resolved = path.expanduser().resolve()
        stat = resolved.stat()
        return (
            str(resolved),
            stat.st_size,
            stat.st_mtime_ns,
            str(self.vars["angle_label_regex"].get()),
        )

    def _load_cached_c3d_marker_data(self, path: Path) -> C3DMarkerData:
        key = self._c3d_cache_key(path)
        if key not in self.c3d_marker_cache:
            self.c3d_marker_cache[key] = load_c3d_marker_data(
                path, angle_label_regex=str(self.vars["angle_label_regex"].get())
            )
        return self.c3d_marker_cache[key]

    def _selected_trial_report_path(self) -> Path | None:
        selected = str(self.vars["selected_trial"].get()).strip()
        if not selected or selected == ALL_TRIALS_LABEL:
            return None
        path = (
            self._graph_output_root()
            / safe_trial_dir_name(selected)
            / "run_report.json"
        )
        return path if path.exists() else None

    def _update_run_report_summary(self) -> None:
        if not hasattr(self, "run_report_summary_var"):
            return
        report_path = self._selected_trial_report_path()
        if report_path is None:
            self.run_report_summary_var.set("Aucun rapport pour l'essai sélectionné.")
            return
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.run_report_summary_var.set(f"Rapport illisible: {exc}")
            return
        self.run_report_summary_var.set(summarize_run_report(report))

    def _isb_report_path(self) -> Path:
        return self._graph_output_root() / "isb_d1_d6_report.json"

    def _load_isb_report_rows(self) -> list[dict[str, object]]:
        path = self._isb_report_path()
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows", []) if isinstance(payload, dict) else []
        return [dict(row) for row in rows if isinstance(row, dict)]

    def _refresh_isb_report_table(self, _event: object | None = None) -> None:
        if not hasattr(self, "isb_report_tree"):
            return
        try:
            rows = self._load_isb_report_rows()
        except (OSError, json.JSONDecodeError) as exc:
            self.isb_report_status_var.set(f"Rapport D1-D6 illisible: {exc}")
            return
        source_label = self.isb_source_filter_var.get()
        source_id = SOURCE_IDS_BY_LABEL.get(source_label, source_label)
        selected_trial = str(self.vars["selected_trial"].get()).strip()
        filtered = filter_isb_rows(
            rows,
            source_id=source_id,
            deviation_id=self.isb_deviation_filter_var.get(),
            entity_id=self.isb_entity_filter_var.get(),
            status_class=self.isb_status_filter_var.get(),
            selected_trial=selected_trial,
        )
        status_values = isb_filter_values(rows, "status_class")
        entity_values = isb_filter_values(rows, "entity_id")
        self.isb_status_filter_combo.configure(values=status_values)
        self.isb_entity_filter_combo.configure(values=entity_values)
        if self.isb_status_filter_var.get() not in status_values:
            self.isb_status_filter_var.set("Tous")
        if self.isb_entity_filter_var.get() not in entity_values:
            self.isb_entity_filter_var.set("Tous")
        self.isb_report_tree.delete(*self.isb_report_tree.get_children())
        for row in filtered:
            self.isb_report_tree.insert("", tk.END, values=isb_table_values(row))
        if rows:
            blocking = sum(bool(row.get("blocking")) for row in filtered)
            self.isb_report_status_var.set(
                f"{len(filtered)} ligne(s), {blocking} comparaison(s) bloquée(s)."
            )
        else:
            self.isb_report_status_var.set(
                "Rapport D1-D6 non encore généré; lancer l'analyse."
            )

    def _captury_marker_display_transform(self) -> tuple[np.ndarray, np.ndarray] | None:
        report_path = self._selected_trial_report_path()
        if report_path is None:
            return None
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._append_log(
                f"\nAlignement marqueurs Captury indisponible pour {report_path}: {exc}\n"
            )
            return None
        return captury_marker_transform_from_report(report)

    def _selected_trial_temporal_lag_s(self) -> float:
        report_path = self._selected_trial_report_path()
        if report_path is None:
            return 0.0
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            synchronization = report.get("temporal_synchronization", {})
            return float(synchronization.get("lag_s", 0.0))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return 0.0

    def _load_cached_joint_chain_data(self, path: Path) -> JointCentreChainData | None:
        key = self._c3d_cache_key(path)
        if key not in self.joint_chain_cache:
            self.joint_chain_cache[key] = load_joint_centre_chain_data(path)
        return self.joint_chain_cache[key]

    def _captury_marker_display_data(self, data: C3DMarkerData) -> C3DMarkerData:
        """Attach editable Q_* aliases without altering raw C3D identifiers."""

        configured_path = str(self.vars["p6_captury_q_landmarks_json"].get()).strip()
        try:
            config = (
                load_captury_q_landmark_config(self._resolve(configured_path))
                if configured_path
                else load_captury_q_landmark_config()
            )
            display_labels = captury_display_labels(data.labels, config)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._append_log(f"\nAlias Q_* Captury indisponibles: {exc}\n")
            display_labels = marker_display_labels(data.labels)
        return replace(data, display_labels=display_labels)

    def _selected_trial_joint_chain_path(self) -> Path | None:
        selected = str(self.vars["selected_trial"].get()).strip()
        if not selected or selected == ALL_TRIALS_LABEL:
            return None
        npz_path = (
            self._graph_output_root()
            / safe_trial_dir_name(selected)
            / "joint_centre_timeseries.npz"
        )
        if npz_path.exists():
            return npz_path
        csv_path = (
            self._graph_output_root()
            / safe_trial_dir_name(selected)
            / "joint_centre_timeseries.csv"
        )
        return csv_path if csv_path.exists() else None

    def _update_embedded_trial_viewer(self) -> None:
        if not hasattr(self, "embedded_viewer"):
            return
        c3d_paths = dict(self._selected_trial_c3d_paths())
        report_path = self._selected_trial_report_path()
        report = {}
        if report_path is not None:
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        if report.get("workflow") == "source_comparison":
            requested = set(report.get("requested_sources", []))
            if "captury" not in requested:
                c3d_paths.pop("Captury", None)
            if not requested.intersection({"motive", "biobuddy"}):
                c3d_paths.pop("Motive", None)
            if (
                "captury" in report.get("available_sources", [])
                and report.get("alignment", {}).get("status") == "native_source_frame"
            ):
                # No inter-system registration: do not fit the marker layer
                # independently of the native Captury centre chain.
                c3d_paths.pop("Motive", None)
        if not c3d_paths:
            self._stop_embedded_viewer_play()
            self.viewer_path_var.set("Aucun C3D pour l'essai sélectionné")
            self.embedded_viewer.set_marker_layers({})
            self.embedded_viewer.set_joint_centre_chains(None)
            self._update_cor_layer_check_states()
            self._refresh_marker_correspondence_lists()
            self.viewer_frame_slider.configure(from_=0, to=0)
            self.viewer_frame_var.set(0)
            self.viewer_frame_label_var.set("0 / 0")
            return
        layers: dict[str, C3DMarkerData] = {}
        failed: list[str] = []
        for source, c3d_path in c3d_paths.items():
            try:
                data = self._load_cached_c3d_marker_data(c3d_path)
                if source.lower() == "captury":
                    data = self._captury_marker_display_data(data)
                layers[source.lower()] = data
            except Exception as exc:
                failed.append(source)
                self._append_log(f"\nVisu 3D C3D impossible pour {c3d_path}: {exc}\n")
        if "captury" in layers:
            captury_coordinates = resolve_system_coordinate_frames("captury", "auto")
            layers["captury"] = transformed_marker_data(
                layers["captury"],
                captury_coordinates.own_c3d_to_common.T,
                np.zeros(3),
            )
        captury_transform = None
        if "captury" in layers and "motive" in layers:
            captury_transform = captury_marker_transform_from_c3d_layers(
                layers["captury"], layers["motive"]
            )
        if captury_transform is None:
            captury_transform = self._captury_marker_display_transform()
        if "captury" in layers and captury_transform is not None:
            layers["captury"] = transformed_marker_data(
                layers["captury"], *captury_transform
            )
        if not layers:
            self._stop_embedded_viewer_play()
            self.viewer_path_var.set("Lecture C3D impossible")
            self.embedded_viewer.set_marker_layers({})
            self.embedded_viewer.set_joint_centre_chains(None)
            self._update_cor_layer_check_states()
            self.viewer_frame_label_var.set("0 / 0")
            return
        self.embedded_viewer.set_marker_layers(layers)
        self.embedded_viewer.set_marker_time_offsets(
            {"captury": self._selected_trial_temporal_lag_s(), "motive": 0.0}
        )
        self._update_visible_marker_layers()
        self._update_embedded_joint_chain()
        max_frame = max(0, self.embedded_viewer.n_frames - 1)
        self.viewer_frame_slider.configure(from_=0, to=max_frame)
        self.viewer_frame_var.set(0)
        self.viewer_frame_label_var.set(f"1 / {self.embedded_viewer.n_frames}")
        display_parts: list[str] = []
        for source in ("Motive", "Captury"):
            path = c3d_paths.get(source)
            if path is None or source.lower() not in layers:
                continue
            try:
                display_path = path.relative_to(PROJECT_DIR).as_posix()
            except ValueError:
                display_path = str(path)
            display_parts.append(f"{source}: {display_path}")
        if failed:
            display_parts.append(f"Échec: {', '.join(failed)}")
        self.viewer_path_var.set(" | ".join(display_parts))
        self._refresh_marker_correspondence_lists()
        self._apply_embedded_view()

    def _update_embedded_joint_chain(self) -> None:
        if not hasattr(self, "embedded_viewer"):
            return
        chain_path = self._selected_trial_joint_chain_path()
        if chain_path is None:
            self.embedded_viewer.set_joint_centre_chains(None)
            self._update_cor_layer_check_states()
            return
        try:
            chain_data = self._load_cached_joint_chain_data(chain_path)
        except Exception as exc:
            self.embedded_viewer.set_joint_centre_chains(None)
            self._update_cor_layer_check_states()
            self._append_log(f"\nChaîne CoR impossible pour {chain_path}: {exc}\n")
            return
        self.embedded_viewer.set_joint_centre_chains(chain_data)
        self._update_cor_layer_check_states()
        self._update_visible_cor_layers()

    def _available_viewer_cor_layers(self) -> set[str]:
        if not hasattr(self, "embedded_viewer"):
            return set()
        chain_data = getattr(self.embedded_viewer, "chain_data", None)
        if chain_data is None:
            return set()
        return {
            str(layer).strip().lower()
            for layer, joints in chain_data.layers.items()
            if joints
        }

    def _update_cor_layer_check_states(self) -> None:
        if not hasattr(self, "viewer_cor_layer_checks") or not hasattr(
            self, "viewer_cor_layer_vars"
        ):
            return
        available_layers = self._available_viewer_cor_layers()
        for layer, checkbutton in self.viewer_cor_layer_checks.items():
            if layer in available_layers:
                checkbutton.configure(state=tk.NORMAL)
            else:
                self.viewer_cor_layer_vars[layer].set(False)
                checkbutton.configure(state=tk.DISABLED)

    def _auto_enable_biobuddy_cor_layer_after_refresh(self) -> None:
        if not self.auto_enable_biobuddy_cor_after_refresh:
            return
        self.auto_enable_biobuddy_cor_after_refresh = False
        if not hasattr(self, "viewer_cor_layer_vars"):
            return
        if "biobuddy" not in self._available_viewer_cor_layers():
            return
        if not self._biobuddy_model_exists():
            return
        self.viewer_cor_layer_vars["biobuddy"].set(True)
        self._update_visible_cor_layers()

    def _update_visible_cor_layers(self) -> None:
        if not hasattr(self, "embedded_viewer") or not hasattr(
            self, "viewer_cor_layer_vars"
        ):
            return
        self._update_cor_layer_check_states()
        layers = [
            layer
            for layer, var in self.viewer_cor_layer_vars.items()
            if bool(var.get())
        ]
        self.embedded_viewer.set_visible_cor_layers(layers)
        if hasattr(self, "viewer_chain_axes_var"):
            self.embedded_viewer.set_show_chain_axes(
                bool(self.viewer_chain_axes_var.get())
            )
        self.embedded_viewer.set_rotate_body_segments_180_x(
            bool(self.vars["p6_rotate_body_segments_180_x"].get())
        )

    def _update_visible_marker_layers(self) -> None:
        if not hasattr(self, "embedded_viewer") or not hasattr(
            self, "viewer_marker_layer_vars"
        ):
            return
        sources = [
            source
            for source, var in self.viewer_marker_layer_vars.items()
            if bool(var.get())
        ]
        self.embedded_viewer.set_visible_marker_sources(sources)

    def _apply_embedded_view(self) -> None:
        if not hasattr(self, "embedded_viewer"):
            return
        selected = str(self.viewer_view_var.get()).strip()
        if selected in {"XY", "YZ", "XZ", "ZX"}:
            self.embedded_viewer.set_camera_plane(selected)
            return
        subject_views = {"Face": "face", "Dos": "dos", "Côté": "cote"}
        self.embedded_viewer.set_subject_view(subject_views.get(selected, "face"))

    def _on_embedded_viewer_slider(self, value: str) -> None:
        if not hasattr(self, "embedded_viewer"):
            return
        frame = int(float(value))
        self.embedded_viewer.set_frame(frame)
        n_frames = self.embedded_viewer.n_frames
        if n_frames <= 0:
            self.viewer_frame_label_var.set("0 / 0")
        else:
            self.viewer_frame_label_var.set(f"{frame + 1} / {n_frames}")

    def _toggle_embedded_viewer_play(self) -> None:
        if self.viewer_playing.get():
            self._stop_embedded_viewer_play()
            return
        n_frames = (
            self.embedded_viewer.n_frames if hasattr(self, "embedded_viewer") else 0
        )
        if n_frames <= 0:
            return
        self.viewer_playing.set(True)
        self.viewer_play_button.configure(text="⏸")
        self._advance_embedded_viewer_frame()

    def _stop_embedded_viewer_play(self) -> None:
        if self.viewer_play_after_id is not None:
            self.after_cancel(self.viewer_play_after_id)
            self.viewer_play_after_id = None
        if hasattr(self, "viewer_playing"):
            self.viewer_playing.set(False)
        if hasattr(self, "viewer_play_button"):
            self.viewer_play_button.configure(text="▶")

    def _advance_embedded_viewer_frame(self) -> None:
        if not self.viewer_playing.get():
            return
        n_frames = self.embedded_viewer.n_frames
        if n_frames <= 0:
            self._stop_embedded_viewer_play()
            return
        next_frame = (int(self.viewer_frame_var.get()) + 1) % max(1, n_frames)
        self.viewer_frame_var.set(next_frame)
        self.embedded_viewer.set_frame(next_frame)
        self.viewer_frame_label_var.set(f"{next_frame + 1} / {n_frames}")
        data = self.embedded_viewer.data
        rate = data.rate if data is not None else 60.0
        interval = max(15, int(1000.0 / min(max(rate, 1.0), 60.0)))
        self.viewer_play_after_id = self.after(
            interval, self._advance_embedded_viewer_frame
        )

    def _graph_output_root(self) -> Path:
        return self._resolve(str(self.vars["p6_out_dir"].get()).strip())

    def _graph_csv_path(self, graph_kind: str) -> Path:
        return self._graph_output_root() / str(GRAPH_CONFIGS[graph_kind]["csv"])

    def _occlusion_csv_path(self) -> Path:
        return self._graph_output_root() / "all_motive_marker_occlusions.csv"

    @staticmethod
    def _read_csv_or_empty(path: Path) -> "pd.DataFrame":
        if pd is None:
            return pd.DataFrame()
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            if path.suffix == ".npz":
                return read_table_npz(path)
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def _set_occlusion_table_headings(self) -> None:
        labels = {
            "marker": "Marqueur",
            "missing_percent": "Occlusion (%)",
            "missing_frames": "Frames manquantes",
            "total_frames": "Frames totales",
        }
        for column, label in labels.items():
            indicator = ""
            if column == self._display_occlusion_sort_column():
                indicator = " ↓" if self.occlusion_sort_descending else " ↑"
            self.occlusion_table.heading(
                column,
                text=f"{label}{indicator}",
                command=lambda column=column: self._sort_occlusions_by_column(column),
            )

    def _display_occlusion_sort_column(self) -> str:
        return (
            "marker"
            if self.occlusion_sort_column in {"marker", "marker_order"}
            else self.occlusion_sort_column
        )

    def _sort_occlusions_by_column(self, column: str) -> None:
        sort_column = "marker_order" if column == "marker" else column
        if sort_column == self.occlusion_sort_column:
            self.occlusion_sort_descending = not self.occlusion_sort_descending
        else:
            self.occlusion_sort_column = sort_column
            self.occlusion_sort_descending = column != "marker"
        self._set_occlusion_table_headings()
        self._populate_occlusion_table()

    def _sorted_occlusion_dataframe(self, dataframe: "pd.DataFrame") -> "pd.DataFrame":
        values = dataframe.copy()
        if "marker" not in values.columns:
            return values
        values["display_marker"] = values["marker"].map(display_marker_name)
        if "marker_order" not in values.columns:
            values["marker_order"] = list(range(len(values)))
        sort_column = self.occlusion_sort_column
        if sort_column not in values.columns:
            sort_column = "marker_order"
        if sort_column in {"marker", "raw_marker"}:
            primary = "display_marker"
        else:
            primary = sort_column
        return values.sort_values(
            [primary, "display_marker"],
            ascending=[not self.occlusion_sort_descending, True],
        )

    def _populate_occlusion_table(self) -> None:
        if pd is None or not hasattr(self, "occlusion_table"):
            return
        table = self.occlusion_table
        table.delete(*table.get_children())
        path = self._occlusion_csv_path()
        dataframe = self._read_csv_or_empty(path)
        if dataframe.empty:
            return
        selected = str(self.vars["selected_trial"].get()).strip()
        if selected and selected != ALL_TRIALS_LABEL and "trial" in dataframe.columns:
            dataframe = dataframe[dataframe["trial"].astype(str) == selected]
        dataframe = self._sorted_occlusion_dataframe(dataframe)
        for _index, row in dataframe.iterrows():
            table.insert(
                "",
                tk.END,
                values=(
                    display_marker_name(row.get("marker", "")),
                    f"{float(row.get('missing_percent', 0.0)):.2f}",
                    int(row.get("missing_frames", 0)),
                    int(row.get("total_frames", 0)),
                ),
            )

    def _selected_graph_trials(self, available_trials: Iterable[str]) -> list[str]:
        trials = sorted(str(trial) for trial in available_trials)
        selected = str(self.vars["selected_trial"].get()).strip()
        if selected and selected != ALL_TRIALS_LABEL and selected in trials:
            return [selected]
        return trials

    def _populate_graph_tree(self, graph_kind: str) -> None:
        if pd is None:
            return
        panel = self.graph_panels.get(graph_kind)
        if not panel:
            return
        if graph_kind == "gait":
            self._refresh_cycle_review()
        tree = panel["tree"]
        assert isinstance(tree, ttk.Treeview)
        tree.delete(*tree.get_children())
        self.graph_payloads[graph_kind] = {}
        if graph_kind == "events":
            self._populate_events_tree(tree, graph_kind)
        else:
            self._populate_metric_tree(tree, graph_kind)
        first_payload_node = self._first_graph_payload_node(tree, graph_kind)
        if first_payload_node:
            tree.selection_set(first_payload_node)
            tree.focus(first_payload_node)
            self._draw_selected_graph(graph_kind)
        elif graph_kind == "gait" and "axes" in panel:
            self._hide_cycle_joint_figures(panel)
            if panel.pop("mean_layout", False):
                panel["figure"].clear()
                panel["axes"] = panel["figure"].add_subplot(111)
            panel["axes"].clear()
            panel["axes"].set_title(
                "Aucun cycle complet: vérifier contacts et fenêtre."
            )
            panel["canvas"].draw_idle()

    def _first_graph_payload_node(
        self, tree: ttk.Treeview, graph_kind: str
    ) -> str | None:
        stack = list(tree.get_children())
        payloads = self.graph_payloads.get(graph_kind, {})
        while stack:
            node_id = stack.pop(0)
            if node_id in payloads:
                return node_id
            stack[0:0] = list(tree.get_children(node_id))
        return None

    def _populate_metric_tree(self, tree: ttk.Treeview, graph_kind: str) -> None:
        config = GRAPH_CONFIGS[graph_kind]
        path = self._graph_csv_path(graph_kind)
        if not path.exists():
            tree.insert(
                "",
                tk.END,
                text=(
                    "Résultats absents : lancer la normalisation."
                    if graph_kind == "gait"
                    else f"CSV absent: {path.name}"
                ),
            )
            return
        dataframe = self._read_csv_or_empty(path)
        if dataframe.empty:
            tree.insert(
                "",
                tk.END,
                text=(
                    "Aucun cycle complet: vérifier contacts et fenêtre."
                    if graph_kind == "gait"
                    else f"CSV vide: {path.name}"
                ),
            )
            return
        metrics = graph_metric_columns(dataframe, config["metrics"])
        if not metrics:
            tree.insert("", tk.END, text="Aucune métrique numérique")
            return
        trials = self._selected_graph_trials(dataframe["trial"].dropna().unique())
        group_columns = tuple(str(column) for column in config["groups"])
        for trial in trials:
            trial_df = dataframe[dataframe["trial"].astype(str) == trial]
            if trial_df.empty:
                continue
            trial_id = tree.insert("", tk.END, text=trial, open=True)
            if graph_kind == "gait":
                mean_id = self._insert_graph_node(
                    tree,
                    graph_kind,
                    trial_id,
                    "Moyennes sélectionnées — jambe droite",
                    {"trial": trial, "side": "right"},
                    metric="selected_leg_mean",
                )
                for joint, label in (
                    ("hip", "Hip"),
                    ("knee", "Knee"),
                    ("ankle", "Ankle"),
                ):
                    self._insert_graph_node(
                        tree,
                        graph_kind,
                        mean_id,
                        label,
                        {"trial": trial, "side": "right", "joint": joint},
                        metric="selected_leg_mean",
                    )
            for metric in metrics:
                metric_id = self._insert_graph_node(
                    tree,
                    graph_kind,
                    trial_id,
                    metric,
                    {"trial": trial},
                    metric=metric,
                )
                self._insert_group_nodes(
                    tree,
                    graph_kind,
                    metric_id,
                    trial_df,
                    group_columns,
                    {"trial": trial},
                    metric,
                )

    def _insert_group_nodes(
        self,
        tree: ttk.Treeview,
        graph_kind: str,
        parent_id: str,
        dataframe: "pd.DataFrame",
        group_columns: tuple[str, ...],
        filters: dict[str, str],
        metric: str,
    ) -> None:
        if not group_columns:
            return
        column = group_columns[0]
        if column not in dataframe.columns:
            return
        for value in sorted(dataframe[column].dropna().astype(str).unique()):
            filtered = dataframe[dataframe[column].astype(str) == value]
            child_filters = {**filters, column: value}
            child_id = self._insert_graph_node(
                tree,
                graph_kind,
                parent_id,
                value,
                child_filters,
                metric=metric,
            )
            self._insert_group_nodes(
                tree,
                graph_kind,
                child_id,
                filtered,
                group_columns[1:],
                child_filters,
                metric,
            )

    def _populate_events_tree(self, tree: ttk.Treeview, graph_kind: str) -> None:
        root = self._graph_output_root()
        event_files = sorted(root.glob("*/trial_events_contacts.csv"))
        by_trial = {path.parent.name: path for path in event_files}
        synchronization_by_trial = {
            path.parent.name: path
            for path in root.glob("*/temporal_synchronization_timeseries.npz")
        }
        phase_by_trial = {
            path.parent.name: path
            for path in root.glob("*/temporal_phase_normalized.npz")
        }
        scientific_by_trial = {
            path.parent.name: path
            for path in root.glob("*/phase_normalized_scientific_timeseries.npz")
        }
        cycles_by_trial = {
            path.parent.name: path for path in root.glob("*/contact_cycles.json")
        }
        all_trials = (
            set(by_trial)
            | set(synchronization_by_trial)
            | set(phase_by_trial)
            | set(scientific_by_trial)
            | set(cycles_by_trial)
        )
        for trial in self._selected_graph_trials(all_trials):
            path = by_trial.get(trial)
            trial_id = tree.insert("", tk.END, text=trial, open=True)
            if path is not None:
                dataframe = self._read_csv_or_empty(path)
                for metric in graph_metric_columns(dataframe, EVENT_METRICS):
                    self._insert_graph_node(
                        tree,
                        graph_kind,
                        trial_id,
                        metric,
                        {"trial": trial, "path": str(path), "x": "time"},
                        metric=metric,
                    )
            scientific = scientific_by_trial.get(trial)
            if scientific is not None:
                dataframe = read_table_npz(scientific)
                group_id = tree.insert(
                    trial_id, tk.END, text="Séries scientifiques 0-100 %", open=False
                )
                families = (
                    sorted(dataframe["family"].dropna().astype(str).unique())
                    if "family" in dataframe.columns
                    else [""]
                )
                excluded = {
                    "phase_percent",
                    "time_s",
                    "family",
                    "normalization_status",
                }
                for family in families:
                    subset = (
                        dataframe[dataframe["family"].astype(str) == family]
                        if family
                        else dataframe
                    )
                    family_id = tree.insert(
                        group_id, tk.END, text=family or "données", open=False
                    )
                    metrics = [
                        column
                        for column in subset.columns
                        if column not in excluded
                        and not column.endswith(
                            (
                                "_finite_fraction",
                                "_largest_gap_s",
                                "_maximum_gap_s",
                                "_normalization_status",
                            )
                        )
                    ]
                    for metric in graph_metric_columns(subset, metrics):
                        self._insert_graph_node(
                            tree,
                            graph_kind,
                            family_id,
                            metric,
                            {
                                "trial": trial,
                                "path": str(scientific),
                                "x": "phase_percent",
                                "family": family,
                            },
                            metric=metric,
                        )
            cycles = cycles_by_trial.get(trial)
            if cycles is not None:
                payload = json.loads(cycles.read_text(encoding="utf-8"))
                cycle_rows = payload.get("cycles", [])
                if cycle_rows:
                    group_id = tree.insert(
                        trial_id, tk.END, text="Cycles de contact", open=False
                    )
                    self._insert_graph_node(
                        tree,
                        graph_kind,
                        group_id,
                        "duration_s",
                        {
                            "trial": trial,
                            "path": str(cycles),
                            "x": "cycle_index",
                        },
                        metric="duration_s",
                    )
            for label, candidate, x_column in (
                (
                    "Synchronisation",
                    synchronization_by_trial.get(trial),
                    "time_s",
                ),
                ("Phase 0-100 %", phase_by_trial.get(trial), "phase_percent"),
            ):
                if candidate is None:
                    continue
                dataframe = read_table_npz(candidate)
                group_id = tree.insert(trial_id, tk.END, text=label, open=True)
                for metric in graph_metric_columns(dataframe, EVENT_METRICS):
                    self._insert_graph_node(
                        tree,
                        graph_kind,
                        group_id,
                        metric,
                        {
                            "trial": trial,
                            "path": str(candidate),
                            "x": x_column,
                        },
                        metric=metric,
                    )
        if not all_trials:
            tree.insert("", tk.END, text="Aucun résultat de découpage")

    def _insert_graph_node(
        self,
        tree: ttk.Treeview,
        graph_kind: str,
        parent_id: str,
        text: str,
        filters: dict[str, str],
        *,
        metric: str,
    ) -> str:
        node_id = tree.insert(parent_id, tk.END, text=text, open=False)
        self.graph_payloads[graph_kind][node_id] = {
            "filters": filters,
            "metric": metric,
        }
        return node_id

    def _toggle_graph_tree_selection(
        self, event: tk.Event, graph_kind: str
    ) -> str | None:
        tree = event.widget
        if not isinstance(tree, ttk.Treeview):
            return None
        node_id = tree.identify_row(event.y)
        if not node_id or node_id not in self.graph_payloads.get(graph_kind, {}):
            return None
        if node_id in tree.selection():
            tree.selection_remove(node_id)
            self.after_idle(lambda kind=graph_kind: self._draw_selected_graph(kind))
            return "break"
        return None

    def _selected_graph_payloads(
        self, graph_kind: str, selection: tuple[str, ...]
    ) -> list[dict[str, object]]:
        payloads = self.graph_payloads.get(graph_kind, {})
        return [payloads[node_id] for node_id in selection if node_id in payloads]

    def _draw_selected_graph(self, graph_kind: str) -> None:
        panel = self.graph_panels.get(graph_kind)
        if not panel or "axes" not in panel:
            return
        tree = panel["tree"]
        assert isinstance(tree, ttk.Treeview)
        selection = tree.selection()
        if not selection:
            return
        payloads = self._selected_graph_payloads(graph_kind, selection)
        if not payloads:
            return
        if graph_kind == "events":
            self._draw_events_graph(graph_kind, payloads[0])
        else:
            self._draw_metric_graph(graph_kind, payloads)

    def _draw_metric_graph(
        self, graph_kind: str, payloads: list[dict[str, object]]
    ) -> None:
        if pd is None:
            return
        panel = self.graph_panels[graph_kind]
        if graph_kind == "gait" and payloads[0].get("metric") == "selected_leg_mean":
            filters = payloads[0]["filters"]
            directory = self._graph_output_root() / safe_trial_dir_name(
                str(filters["trial"])
            )
            report, excluded, _ = cycle_selection(directory)
            if report.get("task", "gait") == str(self.vars["p6_cycle_task"].get()):
                try:
                    self._show_cycle_joint_figures(
                        panel, directory, report, excluded, filters
                    )
                    return
                except (ValueError, OSError, KeyError) as exc:
                    self._hide_cycle_joint_figures(panel)
                    panel["figure"].clear()
                    panel["axes"] = panel["figure"].add_subplot(111)
                    panel["axes"].set_title(f"Moyenne indisponible : {exc}")
                    panel["canvas"].draw_idle()
                    return
        if graph_kind == "gait" and "mean_notebook" in panel:
            self._hide_cycle_joint_figures(panel)
        axes = panel["axes"]
        canvas = panel["canvas"]
        config = GRAPH_CONFIGS[graph_kind]
        dataframe = self._read_csv_or_empty(self._graph_csv_path(graph_kind))
        if dataframe.empty:
            axes.clear()
            axes.set_title("Aucune donnée")
            canvas.draw_idle()
            return
        if graph_kind == "kinematics":
            self._draw_kinematics_graph(payloads, dataframe)
            return
        if graph_kind == "gait":
            filters = dict(payloads[0]["filters"])
            trial = str(filters.get("trial", ""))
            path = (
                self._graph_output_root()
                / safe_trial_dir_name(trial)
                / "gait_cycle_timeseries.npz"
            )
            if path.is_file():
                report, excluded, _ = cycle_selection(path.parent)
                if report.get("task", "gait") != str(self.vars["p6_cycle_task"].get()):
                    axes.clear()
                    axes.set_title(
                        "Normaliser la tâche sélectionnée pour afficher ses cycles"
                    )
                else:
                    draw_gait_cycles(
                        axes,
                        read_table_npz(path),
                        filters,
                        excluded_cycle_ids=excluded,
                        task=report.get("task", "gait"),
                        trial_dir=path.parent,
                    )
            else:
                axes.clear()
                axes.set_title(f"Séries de cycles absentes: {trial}")
            panel["figure"].tight_layout()
            canvas.draw_idle()
            return
        if graph_kind == "centres" and self._draw_joint_centre_graph(payloads):
            return
        if graph_kind == "segments" and self._draw_segment_graph(payloads):
            return
        if graph_kind == "skin_markers" and self._draw_skin_marker_graph(payloads):
            return
        series = self._metric_series_from_payloads(dataframe, payloads, config)
        axes.clear()
        if not series:
            axes.set_title("Aucune donnée")
            canvas.draw_idle()
            return
        selected_metrics = sorted({item["metric"] for item in series})
        metric = selected_metrics[0]
        dataframe = pd.concat(
            [item["dataframe"] for item in series if item["metric"] == metric],
            ignore_index=True,
        ).drop_duplicates()
        if graph_kind == "dimensions":
            if len(selected_metrics) == 1:
                self._draw_dimension_metric_graph(axes, dataframe, metric)
            else:
                self._draw_metric_boxplot(
                    axes, series, f"{config['title']} - métriques sélectionnées"
                )
            panel["figure"].tight_layout()
            canvas.draw_idle()
            return
        if len(series) > 1 or len(dataframe) > 8:
            self._draw_metric_boxplot(
                axes, series, f"{config['title']} - métriques sélectionnées"
            )
            panel["figure"].tight_layout()
            canvas.draw_idle()
            return
        group_columns = [
            column for column in config["groups"] if column in dataframe.columns
        ]
        labels = (
            dataframe[group_columns].astype(str).agg(" / ".join, axis=1)
            if group_columns
            else dataframe.index.astype(str)
        )
        values = dataframe[metric].astype(float)
        axes.bar(range(len(values)), values)
        axes.set_title(f"{config['title']} - {metric}")
        axes.set_ylabel(metric)
        axes.set_xticks(range(len(values)))
        axes.set_xticklabels(labels, rotation=45, ha="right")
        axes.grid(axis="y", alpha=0.3)
        panel["figure"].tight_layout()
        canvas.draw_idle()

    @staticmethod
    def _is_rotation_q_name(q_name: str) -> bool:
        return is_rotation_q_name(q_name)

    @staticmethod
    def _metric_display_name(metric: str, *, q_name: str | None = None) -> str:
        return metric_display_name(metric, q_name=q_name)

    @staticmethod
    def _values_for_display(
        values: "pd.Series | np.ndarray", metric: str, *, q_name: str | None = None
    ) -> "pd.Series | np.ndarray":
        return values_for_display(values, metric, q_name=q_name)

    def _kinematics_timeseries_path(self, trial: str) -> Path:
        return (
            self._graph_output_root()
            / safe_trial_dir_name(trial)
            / "kinematics_q_timeseries.npz"
        )

    def _joint_centre_timeseries_path(self, trial: str) -> Path:
        return (
            self._graph_output_root()
            / safe_trial_dir_name(trial)
            / "joint_centre_timeseries.npz"
        )

    def _segment_rotation_timeseries_path(self, trial: str) -> Path:
        return (
            self._graph_output_root()
            / safe_trial_dir_name(trial)
            / "segment_rotation_timeseries.npz"
        )

    def _skin_marker_timeseries_path(self, trial: str) -> Path:
        return (
            self._graph_output_root()
            / safe_trial_dir_name(trial)
            / "skin_marker_correspondence_timeseries.npz"
        )

    def _legacy_kinematics_timeseries_csv_path(self, trial: str) -> Path:
        return (
            self._graph_output_root()
            / safe_trial_dir_name(trial)
            / "kinematics_q_timeseries.csv"
        )

    def _legacy_joint_centre_timeseries_csv_path(self, trial: str) -> Path:
        return (
            self._graph_output_root()
            / safe_trial_dir_name(trial)
            / "joint_centre_timeseries.csv"
        )

    def _legacy_segment_rotation_timeseries_csv_path(self, trial: str) -> Path:
        return (
            self._graph_output_root()
            / safe_trial_dir_name(trial)
            / "segment_rotation_timeseries.csv"
        )

    def _read_timeseries_table(
        self, path: Path, legacy_csv_path: Path
    ) -> "pd.DataFrame | None":
        if path.exists() and path.stat().st_size:
            return read_table_npz(path)
        if legacy_csv_path.exists() and legacy_csv_path.stat().st_size:
            dataframe = self._read_csv_or_empty(legacy_csv_path)
            return dataframe if not dataframe.empty else None
        return None

    def _draw_joint_centre_graph(self, payloads: list[dict[str, object]]) -> bool:
        if not payloads:
            return False
        metrics = {str(payload["metric"]) for payload in payloads}
        if len(metrics) != 1:
            return False
        metric = next(iter(metrics))
        filters = dict(payloads[0]["filters"])
        trial = str(filters.get("trial", ""))
        if not trial:
            return False
        panel = self.graph_panels["centres"]
        axes_pair = tuple(panel.get("axes_pair", (panel["axes"],)))
        canvas = panel["canvas"]
        for axes in axes_pair:
            axes.clear()
        path = self._joint_centre_timeseries_path(trial)
        dataframe = self._read_timeseries_table(
            path, self._legacy_joint_centre_timeseries_csv_path(trial)
        )
        if dataframe is None:
            axes_pair[0].set_title(f"Série temporelle absente: {path.name}")
        else:
            selected_joints = [
                str(dict(payload["filters"]).get("joint", ""))
                for payload in payloads
                if str(dict(payload["filters"]).get("joint", ""))
            ]
            self._draw_joint_centre_reference_comparisons(
                axes_pair, dataframe, trial, metric, selected_joints
            )
        panel["figure"].tight_layout()
        canvas.draw_idle()
        return True

    def _joint_centre_reference(self) -> str:
        reference = str(self.vars["p6_joint_centre_reference"].get()).strip().lower()
        return (
            reference if reference in {"biobuddy", "motive", "captury"} else "biobuddy"
        )

    def _joint_centre_comparison_sources(self, reference: str) -> list[str]:
        ordered_sources = ["motive", "captury", "biobuddy"]
        return [source for source in ordered_sources if source != reference]

    def _draw_joint_centre_reference_comparisons(
        self,
        axes_pair: tuple[object, ...],
        dataframe: "pd.DataFrame",
        trial: str,
        metric: str,
        selected_joints: list[str],
    ) -> None:
        reference = self._joint_centre_reference()
        sources = self._joint_centre_comparison_sources(reference)
        for index, axes in enumerate(axes_pair):
            if index >= len(sources):
                axes.set_axis_off()
                continue
            source = sources[index]
            if len(selected_joints) == 1:
                draw_joint_centre_error_timeseries_between(
                    axes, dataframe, trial, selected_joints[0], reference, source
                )
            else:
                series = joint_centre_error_boxplot_series_between(
                    dataframe,
                    metric,
                    trial=trial,
                    joints=selected_joints,
                    reference=reference,
                    source=source,
                )
                if series:
                    self._draw_metric_boxplot(
                        axes,
                        series,
                        f"{source.title()} vs {reference.title()} - {metric}",
                    )
                    axes.set_ylabel("Erreur (mm)")
                else:
                    axes.set_title(
                        f"Aucune donnée: {source.title()} vs {reference.title()}"
                    )

    def _draw_segment_graph(self, payloads: list[dict[str, object]]) -> bool:
        if not payloads:
            return False
        metrics = {str(payload["metric"]) for payload in payloads}
        if len(metrics) != 1:
            return False
        metric = next(iter(metrics))
        filters = dict(payloads[0]["filters"])
        trial = str(filters.get("trial", ""))
        if not trial:
            return False
        panel = self.graph_panels["segments"]
        axes = panel["axes"]
        canvas = panel["canvas"]
        axes.clear()
        path = self._segment_rotation_timeseries_path(trial)
        dataframe = self._read_timeseries_table(
            path, self._legacy_segment_rotation_timeseries_csv_path(trial)
        )
        if dataframe is None:
            axes.set_title(f"Série temporelle absente: {path.name}")
        else:
            selected_segments = [
                str(dict(payload["filters"]).get("segment", ""))
                for payload in payloads
                if str(dict(payload["filters"]).get("segment", ""))
            ]
            source = str(filters.get("source", ""))
            if len(payloads) == 1 and selected_segments and source:
                draw_segment_rotation_timeseries(
                    axes, dataframe, trial, source, selected_segments[0]
                )
            else:
                series = segment_rotation_boxplot_series(
                    dataframe,
                    metric,
                    trial=trial,
                    source=source or None,
                    segments=selected_segments,
                )
                if series:
                    self._draw_metric_boxplot(axes, series, f"Segments - {metric}")
                    axes.set_ylabel("Déviation absolue (deg)")
                else:
                    axes.set_title("Aucune rotation segmentaire")
        panel["figure"].tight_layout()
        canvas.draw_idle()
        return True

    def _draw_skin_marker_graph(self, payloads: list[dict[str, object]]) -> bool:
        if not payloads:
            return False
        metrics = {str(payload["metric"]) for payload in payloads}
        if len(metrics) != 1:
            return False
        metric = next(iter(metrics))
        filters = dict(payloads[0]["filters"])
        trial = str(filters.get("trial", ""))
        if not trial:
            return False
        path = self._skin_marker_timeseries_path(trial)
        if not path.exists() or not path.stat().st_size:
            return False
        dataframe = read_table_npz(path)
        if dataframe.empty or "landmark" not in dataframe.columns:
            return False
        selected_landmarks = [
            str(dict(payload["filters"]).get("landmark", ""))
            for payload in payloads
            if str(dict(payload["filters"]).get("landmark", ""))
        ]
        values = dataframe.copy()
        if selected_landmarks:
            values = values[values["landmark"].astype(str).isin(selected_landmarks)]
        if values.empty or "distance_mm" not in values.columns:
            return False
        series = []
        for landmark, landmark_df in values.groupby(values["landmark"].astype(str)):
            distances = landmark_df["distance_mm"].astype(float).dropna().to_numpy()
            if distances.size:
                series.append(
                    {
                        "metric": metric,
                        "label": str(landmark),
                        "values": distances,
                        "dataframe": landmark_df,
                    }
                )
        if not series:
            return False
        panel = self.graph_panels["skin_markers"]
        axes = panel["axes"]
        canvas = panel["canvas"]
        axes.clear()
        self._draw_metric_boxplot(axes, series, f"Marqueurs cutanés - {metric}")
        axes.set_ylabel("Distance (mm)")
        panel["figure"].tight_layout()
        canvas.draw_idle()
        return True

    def _draw_kinematics_graph(
        self, payloads: list[dict[str, object]], metrics_dataframe: "pd.DataFrame"
    ) -> None:
        """Draw q/angle comparisons from the kinematics tab selection.

        A single selected DoF is treated as a waveform exploration request and
        reads ``<trial>/kinematics_q_timeseries.npz``. Broader selections, such
        as ``trial -> bias_rad`` or several DoFs at once, are summarized with
        boxplots so every selected DoF remains visible. Rotation quantities are
        converted from radians to degrees at display time; the output files keep
        their original numerical units.
        """

        panel = self.graph_panels["kinematics"]
        axes = panel["axes"]
        canvas = panel["canvas"]
        axes.clear()
        if len(payloads) == 1:
            filters = dict(payloads[0]["filters"])
            q_name = str(filters.get("q_name", ""))
            trial = str(filters.get("trial", ""))
            source = str(filters.get("source", ""))
            if q_name and trial:
                self._draw_kinematics_timeseries(axes, trial, q_name, source)
                panel["figure"].tight_layout()
                canvas.draw_idle()
                return
        self._draw_kinematics_metric_boxplots(axes, payloads, metrics_dataframe)
        panel["figure"].tight_layout()
        canvas.draw_idle()

    def _draw_kinematics_timeseries(
        self, axes: object, trial: str, q_name: str, source: str = ""
    ) -> None:
        path = self._kinematics_timeseries_path(trial)
        dataframe = self._read_timeseries_table(
            path, self._legacy_kinematics_timeseries_csv_path(trial)
        )
        if dataframe is None:
            axes.set_title(f"Série temporelle absente: {path.name}")
            return
        if "q_name" not in dataframe.columns or "time" not in dataframe.columns:
            axes.set_title("Aucune cinématique temporelle")
            return
        values = dataframe[dataframe["q_name"].astype(str) == q_name].copy()
        if source and "source" in values.columns:
            values = values[values["source"].astype(str) == source]
        if values.empty:
            axes.set_title(f"Aucune donnée temporelle: {q_name}")
            return
        for column in KINEMATIC_TIMESERIES_COLUMNS:
            if column not in values.columns:
                continue
            y = values[column].astype(float)
            if not y.notna().any():
                continue
            y = self._values_for_display(y, column, q_name=q_name)
            color = (
                data_source_color(column)
                if column in {"captury", "motive", "biobuddy", "captury_c3d"}
                else "#64748b"
            )
            axes.plot(values["time"], y, label=column, color=color)
        has_c3d_angles = (
            "captury_c3d" in values.columns
            and values["captury_c3d"].astype(float).notna().any()
        )
        unit = "deg" if self._is_rotation_q_name(q_name) or has_c3d_angles else "native"
        source_suffix = f" - {source}" if source else ""
        axes.set_title(f"{trial} - {q_name}{source_suffix}")
        axes.set_xlabel("Temps (s)")
        axes.set_ylabel(unit)
        axes.legend()
        axes.grid(alpha=0.3)

    def _draw_kinematics_metric_boxplots(
        self,
        axes: object,
        payloads: list[dict[str, object]],
        dataframe: "pd.DataFrame",
    ) -> None:
        series: list[dict[str, object]] = []
        for payload in payloads:
            metric = str(payload["metric"])
            if metric not in dataframe.columns:
                continue
            filters = {
                str(column): str(value)
                for column, value in dict(payload["filters"]).items()
            }
            filtered = dataframe
            for column, value in filters.items():
                if column in filtered.columns:
                    filtered = filtered[filtered[column].astype(str) == value]
            if filtered.empty:
                continue
            if "q_name" in filters:
                values = filtered[metric].astype(float).dropna()
                values = self._values_for_display(
                    values, metric, q_name=filters["q_name"]
                )
                series.append(
                    {
                        "metric": metric,
                        "label": " / ".join(
                            value
                            for value in (
                                filters.get("source", ""),
                                filters["q_name"],
                            )
                            if value
                        ),
                        "values": values.to_numpy(),
                    }
                )
            elif "q_name" in filtered.columns:
                for q_name, q_rows in filtered.groupby("q_name", sort=True):
                    values = q_rows[metric].astype(float).dropna()
                    values = self._values_for_display(
                        values, metric, q_name=str(q_name)
                    )
                    if len(values):
                        series.append(
                            {
                                "metric": metric,
                                "label": str(q_name),
                                "values": values.to_numpy(),
                            }
                        )
        series = [item for item in series if len(item["values"]) > 0]
        if not series:
            axes.set_title("Aucune donnée cinématique")
            return
        metric = str(series[0]["metric"])
        title = f"Cinématiques - {self._metric_display_name(metric)}"
        self._draw_metric_boxplot(axes, series, title)
        axes.set_ylabel(self._metric_display_name(metric))

    def _metric_series_from_payloads(
        self,
        dataframe: "pd.DataFrame",
        payloads: list[dict[str, object]],
        config: dict[str, object],
    ) -> list[dict[str, object]]:
        group_columns = [
            column for column in config["groups"] if column in dataframe.columns
        ]
        series: list[dict[str, object]] = []
        seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
        for payload in payloads:
            metric = str(payload["metric"])
            if metric not in dataframe.columns:
                continue
            filters = {
                str(column): str(value)
                for column, value in dict(payload["filters"]).items()
            }
            key = (metric, tuple(sorted(filters.items())))
            if key in seen:
                continue
            seen.add(key)
            filtered = dataframe
            for column, value in filters.items():
                if column in filtered.columns:
                    filtered = filtered[filtered[column].astype(str) == value]
            if filtered.empty:
                continue
            label_parts = [metric]
            for column in group_columns:
                if column in filters:
                    label_parts.append(filters[column])
            label = " / ".join(label_parts)
            series.append(
                {
                    "metric": metric,
                    "label": label,
                    "dataframe": filtered,
                    "values": filtered[metric].astype(float).dropna().to_numpy(),
                }
            )
        return [item for item in series if len(item["values"]) > 0]

    def _draw_metric_boxplot(
        self, axes: object, series: list[dict[str, object]], title: str
    ) -> None:
        draw_metric_boxplot(axes, series, title)

    def _draw_dimension_metric_graph(
        self, axes: object, dataframe: "pd.DataFrame", metric: str
    ) -> None:
        draw_dimension_metric_graph(axes, dataframe, metric)

    def _manual_phase_bounds(self) -> tuple[float, float] | None:
        try:
            start = float(str(self.vars["p6_time_start"].get()).strip())
            end = float(str(self.vars["p6_time_end"].get()).strip())
        except (KeyError, TypeError, ValueError):
            return None
        if not np.isfinite(start) or not np.isfinite(end) or start == end:
            return None
        return (min(start, end), max(start, end))

    @staticmethod
    def _format_phase_time(value: float) -> str:
        return f"{float(value):.6f}".rstrip("0").rstrip(".")

    def _set_manual_phase_bounds(self, start: float, end: float) -> None:
        lower, upper = sorted((float(start), float(end)))
        self.vars["p6_cut_mode"].set("manual")
        self.vars["p6_time_start"].set(self._format_phase_time(lower))
        self.vars["p6_time_end"].set(self._format_phase_time(upper))
        self.status_var.set(
            f"Phase sélectionnée: {self._format_phase_time(lower)}-"
            f"{self._format_phase_time(upper)} s"
        )

    def _draw_phase_span(
        self,
        axes: object,
        start: float,
        end: float,
        *,
        alpha: float = 0.18,
    ) -> object:
        return axes.axvspan(
            min(start, end),
            max(start, end),
            color=data_source_color("captury"),
            alpha=alpha,
            zorder=0,
        )

    def _draw_manual_phase_span(self, axes: object) -> None:
        bounds = self._manual_phase_bounds()
        if bounds is None:
            return
        self._draw_phase_span(axes, bounds[0], bounds[1], alpha=0.16)

    def _on_phase_drag_press(self, event: object) -> None:
        panel = self.graph_panels.get("events")
        if not panel or event.inaxes is not panel.get("axes") or event.xdata is None:
            return
        if not bool(panel.get("phase_drag_enabled", True)):
            return
        if getattr(event, "button", 1) != 1:
            return
        self.graph_drag_selection = {
            "start": float(event.xdata),
            "patch": None,
        }

    def _on_phase_drag_motion(self, event: object) -> None:
        if "start" not in self.graph_drag_selection:
            return
        panel = self.graph_panels.get("events")
        if not panel or event.inaxes is not panel.get("axes") or event.xdata is None:
            return
        axes = panel["axes"]
        canvas = panel["canvas"]
        patch = self.graph_drag_selection.get("patch")
        if patch is not None:
            patch.remove()
        start = float(self.graph_drag_selection["start"])
        self.graph_drag_selection["patch"] = self._draw_phase_span(
            axes, start, float(event.xdata), alpha=0.24
        )
        canvas.draw_idle()

    def _on_phase_drag_release(self, event: object) -> None:
        if "start" not in self.graph_drag_selection:
            return
        start = float(self.graph_drag_selection["start"])
        end = float(event.xdata) if event.xdata is not None else start
        self.graph_drag_selection = {}
        if abs(end - start) < 1e-6:
            self._draw_selected_graph("events")
            return
        self._set_manual_phase_bounds(start, end)
        self._draw_selected_graph("events")

    def _draw_events_graph(self, graph_kind: str, payload: dict[str, object]) -> None:
        if pd is None:
            return
        panel = self.graph_panels[graph_kind]
        axes = panel["axes"]
        canvas = panel["canvas"]
        path = Path(str(payload["filters"]["path"]))
        x_column = str(payload["filters"].get("x", "time"))
        panel["phase_drag_enabled"] = x_column != "phase_percent"
        metric = str(payload["metric"])
        dataframe = (
            read_table_npz(path)
            if path.suffix.lower() == ".npz"
            else (
                pd.DataFrame(
                    json.loads(path.read_text(encoding="utf-8")).get("cycles", [])
                )
                if path.suffix.lower() == ".json"
                else self._read_csv_or_empty(path)
            )
        )
        family = str(payload["filters"].get("family", ""))
        if family and "family" in dataframe.columns:
            dataframe = dataframe[dataframe["family"].astype(str) == family]
        axes.clear()
        if dataframe.empty:
            axes.set_title("Aucune donnée")
            canvas.draw_idle()
            return
        if x_column not in dataframe.columns or metric not in dataframe.columns:
            axes.set_title("Aucune donnée")
            canvas.draw_idle()
            return
        values = dataframe[metric]
        if values.dtype == bool:
            values = values.astype(int)
        axes.plot(dataframe[x_column], values)
        axes.set_title(f"{path.parent.name} - {metric}")
        axes.set_xlabel("Phase (%)" if x_column == "phase_percent" else "Temps (s)")
        axes.set_ylabel(metric)
        axes.grid(alpha=0.3)
        if x_column != "phase_percent":
            self._draw_manual_phase_span(axes)
        panel["figure"].tight_layout()
        canvas.draw_idle()

    def _open_command_window(self) -> None:
        if self.command_window is not None and self.command_window.winfo_exists():
            self.command_window.lift()
            return

        window = tk.Toplevel(self)
        window.title("Commande")
        window.geometry("900x360")
        window.minsize(720, 280)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(1, weight=1)
        self.command_window = window

        selector = ttk.Frame(window, padding=(10, 10, 10, 0))
        selector.grid(row=0, column=0, sticky="ew")
        selector.columnconfigure(1, weight=1)
        ttk.Label(selector, text="Commande").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            selector,
            textvariable=self.vars["command_mode"],
            values=tuple(COMMAND_MODES.values()),
            state="readonly",
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        self.command_text = tk.Text(window, height=8, wrap=tk.WORD, font=("Menlo", 11))
        self.command_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        actions = ttk.Frame(window, padding=(10, 0, 10, 10))
        actions.grid(row=2, column=0, sticky="ew")
        for column in range(4):
            actions.columnconfigure(column, weight=1)
        command_run_button = self._register_analysis_button(
            ttk.Button(
                actions,
                text="Lancer",
                style="Primary.TButton",
                command=self._run_selected_command,
            )
        )
        command_run_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(actions, text="Copier", command=self._copy_command).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        ttk.Button(actions, text="Ouvrir sortie", command=self._open_output_dir).grid(
            row=0, column=2, sticky="ew", padx=4
        )

        def on_close() -> None:
            self.command_text = None
            self.command_window = None
            window.destroy()

        ttk.Button(actions, text="Fermer", command=on_close).grid(
            row=0, column=3, sticky="ew", padx=(4, 0)
        )
        window.protocol("WM_DELETE_WINDOW", on_close)
        self._update_command_preview()

    def _open_log_window(self) -> None:
        if self.log_window is not None and self.log_window.winfo_exists():
            self.log_window.lift()
            return

        window = tk.Toplevel(self)
        window.title("Log")
        window.geometry("920x520")
        window.minsize(720, 320)
        window.rowconfigure(0, weight=1)
        window.columnconfigure(0, weight=1)
        self.log_window = window

        self.log_text = tk.Text(
            window, wrap=tk.WORD, font=("Menlo", 11), state=tk.DISABLED
        )
        scrollbar = ttk.Scrollbar(
            window, orient=tk.VERTICAL, command=self.log_text.yview
        )
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=(10, 0), pady=10)
        scrollbar.grid(row=0, column=1, sticky="ns", pady=10)

        actions = ttk.Frame(window, padding=(10, 0, 10, 10))
        actions.grid(row=1, column=0, columnspan=2, sticky="ew")
        actions.columnconfigure(0, weight=1)
        ttk.Button(actions, text="Effacer", command=self._clear_log).grid(
            row=0, column=0, sticky="w"
        )

        def on_close() -> None:
            self.log_text = None
            self.log_window = None
            window.destroy()

        ttk.Button(actions, text="Fermer", command=on_close).grid(
            row=0, column=1, sticky="e"
        )
        window.protocol("WM_DELETE_WINDOW", on_close)
        self._sync_log_window()

    def _path_row(
        self,
        parent: ttk.Widget,
        row: int,
        label: str,
        var_name: str,
        filetypes: list[tuple[str, str]] | None = None,
        directory: bool = False,
    ) -> None:
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=10, pady=6
        )
        ttk.Entry(parent, textvariable=self.vars[var_name]).grid(
            row=row, column=1, sticky="ew", padx=6, pady=6
        )
        ttk.Button(
            parent,
            text="Parcourir",
            command=lambda: self._browse_path(
                var_name, filetypes=filetypes, directory=directory
            ),
        ).grid(row=row, column=2, sticky="ew", padx=10, pady=6)

    def _save_path_row(
        self,
        parent: ttk.Widget,
        row: int,
        label: str,
        var_name: str,
        filetypes: list[tuple[str, str]] | None = None,
    ) -> None:
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=10, pady=6
        )
        ttk.Entry(parent, textvariable=self.vars[var_name]).grid(
            row=row, column=1, sticky="ew", padx=6, pady=6
        )
        ttk.Button(
            parent,
            text="Parcourir",
            command=lambda: self._browse_save_path(var_name, filetypes=filetypes),
        ).grid(row=row, column=2, sticky="ew", padx=10, pady=6)

    def _entry_row(
        self, parent: ttk.Widget, row: int, label: str, var_name: str
    ) -> None:
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=10, pady=6
        )
        ttk.Entry(parent, textvariable=self.vars[var_name]).grid(
            row=row, column=1, columnspan=2, sticky="ew", padx=10, pady=6
        )

    def _combo_row(
        self,
        parent: ttk.Widget,
        row: int,
        label: str,
        var_name: str,
        values: tuple[str, ...],
    ) -> None:
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=10, pady=6
        )
        ttk.Combobox(
            parent, textvariable=self.vars[var_name], values=values, state="readonly"
        ).grid(row=row, column=1, columnspan=2, sticky="ew", padx=10, pady=6)

    def _check(self, parent: ttk.Widget, row: int, label: str, var_name: str) -> None:
        ttk.Checkbutton(parent, text=label, variable=self.vars[var_name]).grid(
            row=row, column=0, columnspan=3, sticky="w", padx=10, pady=6
        )

    def _browse_path(
        self,
        var_name: str,
        filetypes: list[tuple[str, str]] | None = None,
        directory: bool = False,
    ) -> None:
        current = Path(str(self.vars[var_name].get()))
        initial_dir = current.parent if current.parent.exists() else PROJECT_DIR
        if directory:
            selected = filedialog.askdirectory(initialdir=initial_dir)
        else:
            selected = filedialog.askopenfilename(
                initialdir=initial_dir,
                filetypes=filetypes or [("Tous les fichiers", "*")],
            )
        if selected:
            self.vars[var_name].set(
                os.path.relpath(selected, PROJECT_DIR)
                if str(selected).startswith(str(PROJECT_DIR))
                else selected
            )

    def _browse_save_path(
        self,
        var_name: str,
        filetypes: list[tuple[str, str]] | None = None,
    ) -> None:
        current = Path(str(self.vars[var_name].get()))
        initial_dir = current.parent if current.parent.exists() else PROJECT_DIR
        selected = filedialog.asksaveasfilename(
            initialdir=initial_dir,
            initialfile=current.name if current.name else "model.bioMod",
            defaultextension=".bioMod",
            filetypes=filetypes or [("Tous les fichiers", "*")],
        )
        if selected:
            self.vars[var_name].set(
                os.path.relpath(selected, PROJECT_DIR)
                if str(selected).startswith(str(PROJECT_DIR))
                else selected
            )

    def _bind_command_preview(self) -> None:
        for var in self.vars.values():
            var.trace_add("write", lambda *_: self._update_command_preview())
        self.vars["selected_trial"].trace_add(
            "write", lambda *_: self._sync_selected_trial()
        )
        self.vars["p6_data_root"].trace_add(
            "write", lambda *_: self._refresh_trial_inventory()
        )
        for name in (
            "p6_source_captury",
            "p6_source_motive",
            "p6_source_biobuddy",
            "p6_model_source",
            "p6_model_to_c3d_axis",
            "p6_captury_model_to_c3d_axis",
            "p6_motive_model_to_c3d_axis",
            "p6_captury_q_landmarks_json",
            "root_offset_mode",
            "p6_segment_reference",
            "p6_captury_reorient_thigh_y_from_cor",
            "p6_captury_rename_axes",
            "p6_captury_calibrate_segment_frames",
            "p6_rotate_body_segments_180_x",
            "p6_reexpress_rotations_zxy",
            "p6_disable_static_model_alignment",
            "p6_disable_motive_marker_alignment",
            "p6_temporal_sync_mode",
            "p6_manual_lag_s",
            "p6_max_lag_s",
            "p6_phase_normalization_points",
            "p6_normalize_gait",
            "p6_gait_events_csv",
            "p6_gait_min_cycle_s",
            "p6_gait_max_cycle_s",
            "p6_gait_max_gap_s",
            "p6_gait_velocity_threshold_m_s",
            "p6_gait_filter_hz",
            "p6_cut_mode",
            "p6_time_start",
            "p6_time_end",
        ):
            self.vars[name].trace_add(
                "write", lambda *_: self._on_p6_auto_analysis_option_changed()
            )
        self.vars["p6_rotate_body_segments_180_x"].trace_add(
            "write", lambda *_: self._update_visible_cor_layers()
        )
        self.vars["p6_captury_rename_axes"].trace_add(
            "write", lambda *_: self._enforce_captury_frame_mode("rename")
        )
        self.vars["p6_captury_calibrate_segment_frames"].trace_add(
            "write", lambda *_: self._enforce_captury_frame_mode("calibrate")
        )
        self.vars["p6_joint_centre_reference"].trace_add(
            "write", lambda *_: self._draw_selected_graph("centres")
        )
        self.vars["biobuddy_c3d_output"].trace_add(
            "write", lambda *_: self._update_biobuddy_biomod_status()
        )
        self.vars["biobuddy_c3d_folder"].trace_add(
            "write", lambda *_: self._refresh_motive57_c3d_mapping()
        )

    def _command_args(self) -> list[str]:
        return build_pipeline_args(self._var_values())

    def _var_values(self) -> dict[str, object]:
        return {name: variable.get() for name, variable in self.vars.items()}

    def _restore_gui_session_state(self) -> None:
        """Restore paths and trial selection without restoring viewer visibility."""

        for name, value in load_gui_session_state(self.gui_session_state_path).items():
            variable = self.vars.get(name)
            if variable is not None:
                variable.set(value)

    def _save_gui_session_state(self) -> None:
        """Persist the minimal context needed to rediscover cached results."""

        save_gui_session_state(self.gui_session_state_path, self._var_values())

    def _on_close(self) -> None:
        """Save the reopen context and close the Tk application."""

        try:
            self._save_gui_session_state()
        except OSError as exc:
            self._append_log(f"\nÉtat GUI non sauvegardé: {exc}\n")
        self.destroy()

    def _split_extra_labels(self) -> list[str]:
        return split_extra_labels(self._var_values())

    def _comparison_args(self) -> list[str]:
        return build_comparison_args(self._var_values())

    def _p6_args(self) -> list[str]:
        return build_p6_args(self._var_values())

    def _p6_occlusions_args(self, trial: str) -> list[str]:
        return build_p6_occlusions_args(self._var_values(), trial)

    def _p6_auto_analysis_args(self, trial: str) -> list[str]:
        return build_p6_auto_analysis_args(self._var_values(), trial)

    def _biobuddy_c3d_model_args(self) -> list[str]:
        values = self._var_values()
        values["biobuddy_c3d_folder"] = str(self._biobuddy_c3d_folder_path())
        if not str(values.get("biobuddy_c3d_mapping_json", "")).strip():
            values["biobuddy_c3d_mapping_json"] = str(
                motive57_mapping_path(values["biobuddy_c3d_folder"])
            )
        return build_biobuddy_c3d_model_args(values)

    def _biobuddy_c3d_ik_args(
        self, c3d_path: Path, source_name: str = "biobuddy_static"
    ) -> list[str]:
        values = self._var_values()
        return build_biobuddy_c3d_ik_args(
            values, str(c3d_path), source_name=source_name
        )

    def _split_var_lines(self, var_name: str) -> list[str]:
        return split_lines(self.vars[var_name].get())

    def _refresh_trial_inventory(self) -> None:
        if not hasattr(self, "trial_combobox"):
            return
        data_root = self._resolve(str(self.vars["p6_data_root"].get()).strip())
        self.trial_inventory = (
            inventory_p6_dataset(data_root) if data_root.exists() else {}
        )
        values = [ALL_TRIALS_LABEL] + sorted(self.trial_inventory)
        self.trial_combobox.configure(values=tuple(values))
        selected = str(self.vars["selected_trial"].get()).strip()
        if selected not in values:
            self.vars["selected_trial"].set(ALL_TRIALS_LABEL)
        else:
            self._update_inventory_table()
            self._update_embedded_trial_viewer()
        self._refresh_default_motive57_c3d_mapping_if_needed()

    def _sync_selected_trial(self) -> None:
        selected = str(self.vars["selected_trial"].get()).strip()
        if not selected or selected == ALL_TRIALS_LABEL:
            self.vars["p6_trials"].set("")
        else:
            self.vars["p6_trials"].set(selected)
            if not str(self.vars["p6_visualize_trial"].get()).strip():
                self.vars["p6_visualize_trial"].set(selected)
        self._update_inventory_table()
        self._populate_occlusion_table()
        self._update_embedded_trial_viewer()
        self._update_run_report_summary()
        self._refresh_isb_report_table()
        self._run_selected_trial_auto_analysis()

    def _update_inventory_table(self) -> None:
        if not hasattr(self, "inventory_trees"):
            return
        for tree in self.inventory_trees.values():
            tree.delete(*tree.get_children())
        selected = str(self.vars["selected_trial"].get()).strip()
        trials = (
            sorted(self.trial_inventory)
            if selected in {"", ALL_TRIALS_LABEL}
            else [selected]
        )
        for trial in trials:
            systems = self.trial_inventory.get(trial, {})
            for system in ("Captury", "Motive"):
                tree = self.inventory_trees.get(system)
                if tree is None:
                    continue
                for kind, path in sorted(systems.get(system, {}).items()):
                    try:
                        display_path = path.relative_to(PROJECT_DIR).as_posix()
                    except ValueError:
                        display_path = str(path)
                    tree.insert(
                        "",
                        tk.END,
                        values=(
                            kind.upper(),
                            vertical_axis_label(kind, system),
                            display_path,
                        ),
                    )

    def _command_mode(self) -> str:
        value = str(self.vars["command_mode"].get())
        for key, label in COMMAND_MODES.items():
            if value == key or value == label:
                return key
        return "kinematic"

    def _current_args(self) -> list[str]:
        mode = self._command_mode()
        if mode == "biobuddy_c3d_model":
            return self._biobuddy_c3d_model_args()
        if mode == "biobuddy_c3d_ik":
            c3d_path = self._static_motive_c3d_path()
            return self._biobuddy_c3d_ik_args(c3d_path) if c3d_path else []
        if mode == "pipeline":
            return self._command_args()
        if mode == "comparison":
            return self._comparison_args()
        return self._p6_args()

    def _current_output_dir(self) -> Path:
        mode = self._command_mode()
        if mode == "biobuddy_c3d_model":
            value = str(self.vars["biobuddy_c3d_output"].get()).strip()
            return self._resolve(value).parent
        if mode == "biobuddy_c3d_ik":
            return self._resolve(str(self.vars["p6_out_dir"].get()).strip())
        if mode == "pipeline":
            value = str(self.vars["out_dir"].get()).strip()
        elif mode == "comparison":
            value = str(self.vars["compare_out_dir"].get()).strip()
        else:
            value = str(self.vars["p6_out_dir"].get()).strip()
        return self._resolve(value)

    def _validate_current_command(self) -> bool:
        mode = self._command_mode()
        if mode == "biobuddy_c3d_model":
            return self._validate_biobuddy_c3d_model()
        if mode == "biobuddy_c3d_ik":
            return self._validate_biobuddy_c3d_ik()
        if mode == "pipeline":
            return self._validate()
        if mode == "comparison":
            return self._validate_comparison()
        return self._validate_p6_analysis()

    def _update_command_preview(self) -> None:
        command = " ".join(shlex.quote(part) for part in self._current_args())
        if self.command_text is None:
            return
        self.command_text.configure(state=tk.NORMAL)
        self.command_text.delete("1.0", tk.END)
        self.command_text.insert(tk.END, command)
        self.command_text.configure(state=tk.DISABLED)

    def _validate(self) -> bool:
        required_paths = [("BVH", "bvh"), ("C3D", "c3d")]
        for label, var_name in required_paths:
            value = str(self.vars[var_name].get()).strip()
            if not value:
                messagebox.showerror(
                    "Champ manquant", f"Le fichier {label} est requis."
                )
                return False
            if not self._resolve(value).exists():
                messagebox.showerror("Fichier introuvable", f"{label}: {value}")
                return False
        optional_paths = [("FBX", "fbx"), ("Mapping q/C3D", "comparison_map")]
        for label, var_name in optional_paths:
            value = str(self.vars[var_name].get()).strip()
            if value and not self._resolve(value).exists():
                messagebox.showerror("Fichier introuvable", f"{label}: {value}")
                return False
        if (
            bool(self.vars["animate_superposed"].get())
            and not str(self.vars["fbx"].get()).strip()
        ):
            messagebox.showerror(
                "FBX requis", "La scène superposée Rerun nécessite un fichier FBX."
            )
            return False
        return True

    def _resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else PROJECT_DIR / path

    def _update_biobuddy_biomod_status(self) -> None:
        status_var = getattr(self, "biobuddy_biomod_status_var", None)
        if status_var is None:
            return
        raw_path = str(self.vars["biobuddy_c3d_output"].get()).strip()
        if not raw_path:
            status_var.set(
                "Aucun modèle biorbd sélectionné: créer le modèle dans l'onglet BioBuddy."
            )
            return
        path = self._resolve(raw_path)
        if path.exists():
            status_var.set(f"Modèle biorbd disponible: {path.name}")
        else:
            status_var.set(
                "Modèle biorbd introuvable: créer le modèle dans l'onglet BioBuddy."
            )

    def _run_pipeline(self) -> None:
        if self.process is not None:
            messagebox.showinfo("Exécution en cours", "Le pipeline est déjà lancé.")
            return
        if not self._validate():
            return

        args = self._command_args()
        self._set_running(True)
        self._clear_log()
        self._append_log("$ " + " ".join(shlex.quote(part) for part in args) + "\n\n")

        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", "/private/tmp/captury_models_mplconfig")

        def worker() -> None:
            try:
                self.process = subprocess.Popen(
                    args,
                    cwd=PROJECT_DIR,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert self.process.stdout is not None
                for line in self.process.stdout:
                    self.output_queue.put(line)
                return_code = self.process.wait()
                self.output_queue.put(("__return_code__", return_code))
            except Exception as exc:  # pragma: no cover - surfaced to the UI
                self.output_queue.put(f"\nErreur de lancement: {exc}\n")
                self.output_queue.put(("__return_code__", 1))

        threading.Thread(target=worker, daemon=True).start()

    def _validate_comparison(self) -> bool:
        reference_c3d = str(self.vars["compare_reference_c3d"].get()).strip()
        test_c3d = str(self.vars["compare_test_c3d"].get()).strip()
        if reference_c3d or test_c3d:
            if not reference_c3d or not test_c3d:
                messagebox.showerror(
                    "Paire incomplète",
                    "Référence C3D et Test C3D doivent être fournis ensemble.",
                )
                return False
            explicit_paths = [
                ("Référence C3D", reference_c3d),
                (
                    "Référence BVH",
                    str(self.vars["compare_reference_bvh"].get()).strip(),
                ),
                (
                    "Référence FBX",
                    str(self.vars["compare_reference_fbx"].get()).strip(),
                ),
                ("Test C3D", test_c3d),
                ("Test BVH", str(self.vars["compare_test_bvh"].get()).strip()),
                ("Test FBX", str(self.vars["compare_test_fbx"].get()).strip()),
            ]
            for label, value in explicit_paths:
                if not value:
                    continue
                if not self._resolve(value).exists():
                    messagebox.showerror("Fichier introuvable", f"{label}: {value}")
                    return False
        else:
            data_root = str(self.vars["compare_data_root"].get()).strip()
            if not data_root or not self._resolve(data_root).exists():
                messagebox.showerror("Dossier introuvable", f"Racine: {data_root}")
                return False
        landmark_map = str(self.vars["compare_landmark_map"].get()).strip()
        if landmark_map and not self._resolve(landmark_map).exists():
            messagebox.showerror(
                "Fichier introuvable", f"Carte repères: {landmark_map}"
            )
            return False
        return True

    def _validate_biobuddy_c3d_model(self) -> bool:
        c3d_path = self._biobuddy_c3d_folder_path()
        if not c3d_path.exists() or not c3d_path.is_dir():
            messagebox.showerror("Dossier C3D introuvable", str(c3d_path))
            return False

        output = str(self.vars["biobuddy_c3d_output"].get()).strip()
        if not output:
            messagebox.showerror(
                "Champ manquant", "Le fichier bioMod de sortie est requis."
            )
            return False
        output_path = self._resolve(output)
        if output_path.suffix != ".bioMod":
            messagebox.showerror(
                "Sortie invalide", "La sortie doit être un fichier .bioMod."
            )
            return False
        if not output_path.parent.exists():
            messagebox.showerror("Dossier sortie introuvable", str(output_path.parent))
            return False
        return True

    def _biobuddy_model_exists(self) -> bool:
        output = str(self.vars["biobuddy_c3d_output"].get()).strip()
        return bool(output) and self._resolve(output).exists()

    def _validate_biobuddy_c3d_ik(self) -> bool:
        biomod_path = self._resolve(str(self.vars["biobuddy_c3d_output"].get()).strip())
        if not biomod_path.exists():
            messagebox.showerror("Modèle BioBuddy introuvable", str(biomod_path))
            return False
        c3d_path = self._static_motive_c3d_path()
        if c3d_path is None:
            messagebox.showerror(
                "C3D statique introuvable",
                "Aucun C3D Motive correspondant à l'essai statique n'a été trouvé.",
            )
            return False
        return True

    def _validate_p6_analysis(self) -> bool:
        if not any(
            bool(self.vars[f"p6_source_{source}"].get())
            for source in ("captury", "motive", "biobuddy")
        ):
            messagebox.showerror(
                "Aucune source sélectionnée",
                "Sélectionnez au moins une source dans Données.",
            )
            return False
        data_root = str(self.vars["p6_data_root"].get()).strip()
        if not data_root or not self._resolve(data_root).exists():
            messagebox.showerror(
                "Dossier introuvable", f"Dossier cinématique: {data_root}"
            )
            return False
        sync_mode = str(self.vars["p6_temporal_sync_mode"].get()).strip()
        manual_lag = str(self.vars["p6_manual_lag_s"].get()).strip()
        if sync_mode == "manual":
            try:
                lag_value = float(manual_lag)
            except ValueError:
                messagebox.showerror(
                    "Lag temporel invalide",
                    "Le mode manual requiert un lag Captury numérique en secondes.",
                )
                return False
            if not np.isfinite(lag_value):
                messagebox.showerror("Lag temporel invalide", manual_lag)
                return False
        try:
            max_lag = float(str(self.vars["p6_max_lag_s"].get()).strip())
            phase_points = int(
                str(self.vars["p6_phase_normalization_points"].get()).strip()
            )
        except ValueError:
            messagebox.showerror(
                "Synchronisation invalide",
                "Le lag maximal doit être numérique et les points de phase entiers.",
            )
            return False
        if not np.isfinite(max_lag) or max_lag <= 0 or phase_points < 2:
            messagebox.showerror(
                "Synchronisation invalide",
                "Le lag maximal doit être positif et la phase contenir au moins 2 points.",
            )
            return False
        return True

    def _run_selected_trial_auto_analysis(self) -> None:
        if not any(
            bool(self.vars[f"p6_source_{source}"].get())
            for source in ("captury", "motive", "biobuddy")
        ):
            self.pending_auto_analysis = False
            self.status_var.set("Sélectionnez au moins une source dans Données")
            return
        if self.process is not None:
            self.pending_auto_analysis = True
            self.status_var.set("Analyse P6 différée: exécution en cours")
            return
        if not bool(self.vars["p6_auto_analyze"].get()):
            return
        selected = str(self.vars["selected_trial"].get()).strip()
        if not selected or selected == ALL_TRIALS_LABEL:
            return
        data_root = str(self.vars["p6_data_root"].get()).strip()
        if not data_root or not self._resolve(data_root).exists():
            return
        if selected not in self.trial_inventory:
            return
        self.pending_auto_analysis = False
        self.status_var.set(f"Analyse P6 en cours: {selected}")
        self._run_args(self._p6_auto_analysis_args(selected), command_mode="kinematic")

    def _on_p6_auto_analysis_option_changed(self) -> None:
        self._schedule_p6_auto_analysis("option modifiée")

    def _enforce_captury_frame_mode(self, selected_mode: str) -> None:
        """Keep the pedagogical and landmark-based Captury modes exclusive."""

        if selected_mode == "rename":
            selected = bool(self.vars["p6_captury_rename_axes"].get())
            other_name = "p6_captury_calibrate_segment_frames"
        else:
            selected = bool(self.vars["p6_captury_calibrate_segment_frames"].get())
            other_name = "p6_captury_rename_axes"
        if selected and bool(self.vars[other_name].get()):
            self.vars[other_name].set(False)

    def _schedule_p6_auto_analysis(self, reason: str) -> None:
        if self.process is not None:
            self.pending_auto_analysis = True
            self.status_var.set(f"Analyse P6 différée: {reason}")
            return
        if self.auto_analysis_after_id is not None:
            self.after_cancel(self.auto_analysis_after_id)
        self.status_var.set(f"Analyse P6 planifiée: {reason}")
        self.auto_analysis_after_id = self.after(
            P6_AUTO_ANALYSIS_DEBOUNCE_MS,
            lambda: self._run_scheduled_p6_auto_analysis(reason),
        )

    def _run_scheduled_p6_auto_analysis(self, reason: str) -> None:
        self.auto_analysis_after_id = None
        self.status_var.set(f"Analyse P6 relancée: {reason}")
        self._run_selected_trial_auto_analysis()

    def _run_pending_auto_analysis_if_needed(self) -> None:
        if not self.pending_auto_analysis:
            return
        self.pending_auto_analysis = False
        self._run_selected_trial_auto_analysis()

    def _load_p6_debug_preset(self) -> None:
        self.vars["command_mode"].set(COMMAND_MODES["kinematic"])
        self.vars["p6_data_root"].set("local_trials/2026-06-30_P6_flat")
        self.vars["p6_out_dir"].set("out_p6_motive_captury_debug")
        self.vars["p6_trials"].set("Static")
        self.vars["selected_trial"].set("Static")
        self.vars["p6_static_trial"].set("Static")
        self.vars["p6_cut_mode"].set("manual")
        self.vars["p6_time_start"].set("")
        self.vars["p6_time_end"].set("")
        self.vars["p6_joint_filter"].set("Hip|Knee|Ankle|Leg|Foot")
        self.vars["p6_auto_analyze"].set(True)
        self.vars["p6_model_source"].set("bvh")
        self.vars["p6_model_to_c3d_axis"].set("auto")
        self.vars["p6_captury_model_to_c3d_axis"].set("auto")
        self.vars["p6_motive_model_to_c3d_axis"].set("auto")
        self.vars["root_offset_mode"].set(ROOT_OFFSET_MODE_LABELS["auto"])
        self.vars["p6_segment_reference"].set("biobuddy")
        self.vars["p6_no_mesh"].set(True)
        self.vars["p6_no_figures"].set(True)
        self.vars["p6_no_cache"].set(False)
        self.vars["p6_run_ik_batch"].set(False)
        self.vars["p6_ik_max_frames"].set("0")
        self.vars["p6_visualize"].set(False)
        self.vars["p6_visualize_trial"].set("Static")
        self.vars["p6_headless"].set(True)
        self.vars["p6_rerun_wait_seconds"].set("0")
        self.status_var.set("Preset P6 debug chargé")

    def _selected_trial_c3d_path(self) -> Path | None:
        path, _source = self._selected_trial_c3d_path_and_source()
        return path

    def _selected_trial_c3d_path_and_source(self) -> tuple[Path | None, str | None]:
        c3d_paths = self._selected_trial_c3d_paths()
        for system in ("Motive", "Captury"):
            c3d_path = c3d_paths.get(system)
            if c3d_path is not None:
                return c3d_path, system
        return None, None

    def _selected_trial_c3d_paths(self) -> dict[str, Path]:
        selected = str(self.vars["selected_trial"].get()).strip()
        if not selected or selected == ALL_TRIALS_LABEL:
            selected = str(self.vars["p6_static_trial"].get()).strip()
        files = self.trial_inventory.get(selected, {})
        paths: dict[str, Path] = {}
        for system in ("Motive", "Captury"):
            c3d_path = files.get(system, {}).get("c3d")
            if c3d_path is not None and c3d_path.exists():
                paths[system] = c3d_path
        return paths

    def _static_motive_c3d_path(self) -> Path | None:
        static_trial = str(self.vars["p6_static_trial"].get()).strip() or "Static"
        if not self.trial_inventory:
            self._refresh_trial_inventory()
        c3d_path = (
            self.trial_inventory.get(static_trial, {}).get("Motive", {}).get("c3d")
        )
        if c3d_path is not None and c3d_path.exists():
            return c3d_path
        matching_static = [
            files.get("Motive", {}).get("c3d")
            for name, files in self.trial_inventory.items()
            if name.endswith("_" + static_trial)
        ]
        matching_static = [
            path for path in matching_static if path is not None and path.exists()
        ]
        if len(matching_static) == 1:
            return matching_static[0]
        folder = self._biobuddy_c3d_folder_path()
        for candidate in (
            folder / f"P6_{static_trial}.c3d",
            folder / f"{static_trial}.c3d",
            folder / "P6_Static.c3d",
            folder / "Static.c3d",
        ):
            if candidate.exists():
                return candidate
        return None

    def _open_selected_trial_viewer(self) -> None:
        from gui_trial_viewer import open_trial_view_window

        source = self.embedded_viewer
        if not source.marker_layers and source.chain_data is None:
            messagebox.showinfo("Visu 3D essai", "Charge un essai avant d'ouvrir la grande fenêtre.")
            return
        open_trial_view_window(source.winfo_toplevel(), source,
                               "Visu 3D essai — " + self.viewer_path_var.get())

    def _run_selected_command(self) -> None:
        if self.process is not None:
            messagebox.showinfo("Exécution en cours", "Une commande est déjà lancée.")
            return
        if not self._validate_current_command():
            return
        self._run_args(self._current_args())

    def _run_args(
        self,
        args: list[str],
        command_mode: str | None = None,
        log_intro: str | None = None,
        clear_log: bool = True,
    ) -> None:
        self.running_command_mode = command_mode or self._command_mode()
        self._set_running(True)
        if clear_log:
            self._clear_log()
        self._append_log("$ " + " ".join(shlex.quote(part) for part in args) + "\n\n")
        if log_intro:
            self._append_log(log_intro.rstrip() + "\n\n")
        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", "/private/tmp/captury_models_mplconfig")
        env.setdefault("PYTHONUNBUFFERED", "1")

        def worker() -> None:
            try:
                self.process = subprocess.Popen(
                    args,
                    cwd=PROJECT_DIR,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert self.process.stdout is not None
                for line in self.process.stdout:
                    self.output_queue.put(line)
                return_code = self.process.wait()
                self.output_queue.put(("__return_code__", return_code))
            except Exception as exc:  # pragma: no cover - surfaced to the UI
                self.output_queue.put(f"\nErreur de lancement: {exc}\n")
                self.output_queue.put(("__return_code__", 1))

        threading.Thread(target=worker, daemon=True).start()

    def _run_comparison(self) -> None:
        if self.process is not None:
            messagebox.showinfo("Exécution en cours", "Une commande est déjà lancée.")
            return
        if not self._validate_comparison():
            return
        self._run_args(self._comparison_args(), command_mode="comparison")

    def _run_p6_analysis(self) -> None:
        if self.process is not None:
            messagebox.showinfo("Exécution en cours", "Une commande est déjà lancée.")
            return
        if not self._validate_p6_analysis():
            return
        self._run_args(self._p6_args(), command_mode="kinematic")

    def _run_p6_ik_batch(self) -> None:
        if self.process is not None:
            messagebox.showinfo("Exécution en cours", "Une commande est déjà lancée.")
            return
        if not self._validate_p6_analysis():
            return
        biomod_path = self._resolve(str(self.vars["biobuddy_c3d_output"].get()).strip())
        if not biomod_path.exists():
            messagebox.showerror("Modèle BioBuddy introuvable", str(biomod_path))
            return
        values = self._var_values()
        values["p6_run_ik_batch"] = True
        values["p6_trials"] = ""
        values["biobuddy_c3d_output"] = str(biomod_path)
        args = build_p6_args(values)
        self.vars["command_mode"].set(COMMAND_MODES["kinematic"])
        self._run_args(args, command_mode="kinematic")

    def _run_biobuddy_c3d_model_creation(self) -> None:
        if self.process is not None:
            messagebox.showinfo("Exécution en cours", "Une commande est déjà lancée.")
            return
        if not self._validate_biobuddy_c3d_model():
            return
        self._save_motive57_mapping_json()
        self.vars["command_mode"].set(COMMAND_MODES["biobuddy_c3d_model"])
        self._run_args(
            self._biobuddy_c3d_model_args(),
            command_mode="biobuddy_c3d_model",
            log_intro=(
                "Création BioBuddy: préparation du dossier C3D, lecture du mapping "
                "Motive 57, construction du modèle, puis écriture du bioMod."
            ),
        )

    def _stop_pipeline(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        self.status_var.set("Arrêt demandé")

    def _set_running(self, running: bool) -> None:
        state = tk.DISABLED if running else tk.NORMAL
        active_buttons: list[ttk.Button] = []
        for button in self.analysis_buttons:
            try:
                button.configure(state=state)
            except tk.TclError:
                continue
            active_buttons.append(button)
        self.analysis_buttons = active_buttons
        self.stop_button.configure(state=tk.NORMAL if running else tk.DISABLED)
        self.status_var.set(
            self._running_status_message(self.running_command_mode)
            if running
            else "Prêt"
        )

    def _running_status_message(self, command_mode: str | None) -> str:
        messages = {
            "biobuddy_c3d_model": "Création du modèle BioBuddy en cours...",
            "biobuddy_c3d_ik": "IK non linéaire TRF statique en cours...",
            "kinematic": "Analyse comparative en cours...",
            "comparison": "Comparaison en cours...",
            "pipeline": "Pipeline en cours...",
        }
        return messages.get(str(command_mode), "Exécution en cours")

    def _drain_output_queue(self) -> None:
        try:
            while True:
                item = self.output_queue.get_nowait()
                if isinstance(item, tuple) and item[0] == "__return_code__":
                    return_code = item[1]
                    finished_mode = self.running_command_mode
                    self.process = None
                    self.running_command_mode = None
                    self._set_running(False)
                    self.status_var.set(
                        "Terminé" if return_code == 0 else f"Échec ({return_code})"
                    )
                    self._append_log(
                        f"\nProcessus terminé avec le code {return_code}.\n"
                    )
                    if return_code == 0:
                        self._invalidate_output_caches()
                        self._refresh_results()
                    if finished_mode == "biobuddy_c3d_model":
                        self._notify_biobuddy_c3d_model_creation_finished(return_code)
                    elif finished_mode == "biobuddy_c3d_ik" and return_code == 0:
                        self._run_static_p6_analysis_after_biobuddy_ik()
                    self._run_pending_auto_analysis_if_needed()
                else:
                    output_text = str(item)
                    self._update_status_from_process_output(output_text)
                    self._append_log(output_text)
        except queue.Empty:
            pass
        self.after(100, self._drain_output_queue)

    def _update_status_from_process_output(self, line: str) -> None:
        """Promote concise BioBuddy IK progress from stdout to the status line."""

        if not line.startswith("[BioBuddy IK] frame "):
            return
        self.status_var.set(line.removeprefix("[BioBuddy IK] ").strip())

    def _append_log(self, text: str) -> None:
        self.log_buffer += text
        self._append_log_to_window(text)

    def _clear_log(self) -> None:
        self.log_buffer = ""
        if self.log_text is None:
            return
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _append_log_to_window(self, text: str) -> None:
        if self.log_text is None:
            return
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _sync_log_window(self) -> None:
        if self.log_text is None:
            return
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, self.log_buffer)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _copy_command(self) -> None:
        command = " ".join(shlex.quote(part) for part in self._current_args())
        self.clipboard_clear()
        self.clipboard_append(command)
        self.status_var.set("Commande copiée")

    def _copy_comparison_command(self) -> None:
        command = " ".join(shlex.quote(part) for part in self._comparison_args())
        self.clipboard_clear()
        self.clipboard_append(command)
        self.status_var.set("Commande comparaison copiée")

    def _copy_p6_command(self) -> None:
        command = " ".join(shlex.quote(part) for part in self._p6_args())
        self.clipboard_clear()
        self.clipboard_append(command)
        self.status_var.set("Commande cinématique copiée")

    def _copy_biobuddy_c3d_model_command(self) -> None:
        if self._validate_biobuddy_c3d_model():
            self._save_motive57_mapping_json()
        command = " ".join(
            shlex.quote(part) for part in self._biobuddy_c3d_model_args()
        )
        self.clipboard_clear()
        self.clipboard_append(command)
        self.status_var.set("Commande BioBuddy copiée")

    def _biobuddy_c3d_model_creation_message(
        self, return_code: int
    ) -> tuple[str, str, str]:
        output_path = self._resolve(str(self.vars["biobuddy_c3d_output"].get()).strip())
        if return_code != 0:
            return (
                "error",
                "Création BioBuddy échouée",
                (
                    f"La commande s'est terminée avec le code {return_code}.\n\n"
                    "Consulte le log pour voir l'erreur BioBuddy/Python complète."
                ),
            )
        if not output_path.exists():
            return (
                "error",
                "Modèle BioBuddy introuvable",
                (
                    "La commande s'est terminée sans erreur, mais le fichier attendu "
                    f"n'existe pas:\n{output_path}\n\n"
                    "Vérifie le dossier de sortie et le log de création."
                ),
            )
        size = output_path.stat().st_size
        return (
            "info",
            "Modèle BioBuddy créé",
            (
                f"Le modèle a été écrit ici:\n{output_path}\n\n"
                f"Taille: {size} octets.\n\n"
                "L'IK non linéaire TRF de l'essai statique Motive va être lancée "
                "automatiquement pour rendre ce modèle disponible dans les comparaisons."
            ),
        )

    def _notify_biobuddy_c3d_model_creation_finished(self, return_code: int) -> None:
        level, title, message = self._biobuddy_c3d_model_creation_message(return_code)
        output_path = self._resolve(str(self.vars["biobuddy_c3d_output"].get()).strip())
        if level == "info" and output_path.exists():
            self.vars["model_explorer_path"].set(str(output_path))
            self.status_var.set(f"Modèle BioBuddy créé: {output_path.name}")
            self._append_log(f"\n{title}\n{message}\n")
            self._run_biobuddy_static_ik_after_model_creation()
        else:
            self.status_var.set(title)
            messagebox.showerror(title, message)

    def _run_biobuddy_static_ik_after_model_creation(self) -> None:
        c3d_path = self._static_motive_c3d_path()
        if c3d_path is None:
            self.status_var.set("Modèle créé; C3D statique Motive introuvable")
            return
        self.vars["command_mode"].set(COMMAND_MODES["biobuddy_c3d_ik"])
        self._run_args(
            self._biobuddy_c3d_ik_args(c3d_path),
            command_mode="biobuddy_c3d_ik",
            log_intro=(
                "Étape suivante: IK non linéaire TRF de l'essai statique "
                f"avec {c3d_path.name}."
            ),
            clear_log=False,
        )

    def _run_static_p6_analysis_after_biobuddy_ik(self) -> None:
        values = self._var_values()
        values["p6_trials"] = (
            str(self.vars["p6_static_trial"].get()).strip() or "Static"
        )
        values["p6_run_ik_batch"] = True
        values["p6_no_figures"] = True
        values["p6_no_mesh"] = True
        values["p6_max_mesh_points"] = "0"
        self.auto_enable_biobuddy_cor_after_refresh = True
        self.status_var.set("IK statique terminée; analyse P6 Static en cours")
        self._run_args(
            build_p6_args(values),
            command_mode="kinematic",
            log_intro=(
                "Étape suivante: analyse Static avec le modèle BioBuddy reconstruit "
                "pour exposer la couche BioBuddy dans le viewer."
            ),
            clear_log=False,
        )

    def _biobuddy_c3d_folder_path(self) -> Path:
        raw_path = str(self.vars["biobuddy_c3d_folder"].get()).strip()
        if raw_path:
            return self._resolve(raw_path)
        data_root = self._resolve(str(self.vars["p6_data_root"].get()).strip())
        if data_root.exists() and data_root.is_dir():
            for candidate in data_root.iterdir():
                if candidate.is_dir() and candidate.name.lower() == "motive":
                    return candidate
        return data_root / "Motive"

    def _refresh_default_motive57_c3d_mapping_if_needed(self) -> None:
        if "biobuddy_c3d_folder" not in self.vars:
            return
        if str(self.vars["biobuddy_c3d_folder"].get()).strip():
            return
        if "motive57_role_combos" not in self.__dict__:
            return
        folder = self._biobuddy_c3d_folder_path()
        if folder.exists() and folder.is_dir():
            self._refresh_motive57_c3d_mapping()

    def _motive57_mapping_json_path(self) -> Path:
        raw_path = str(self.vars["biobuddy_c3d_mapping_json"].get()).strip()
        if raw_path:
            return self._resolve(raw_path)
        return motive57_mapping_path(self._biobuddy_c3d_folder_path())

    def _motive57_role_assignments_from_vars(self) -> dict[str, str]:
        assignments: dict[str, str] = {}
        for role in MOTIVE_57_C3D_ROLES:
            filename = str(self.vars[f"motive57_role_{role.key}"].get()).strip()
            if filename:
                assignments[role.key] = filename
        return assignments

    def _set_motive57_role_assignments(self, assignments: dict[str, str]) -> None:
        available = set(self.motive57_c3d_files)
        for role in MOTIVE_57_C3D_ROLES:
            filename = assignments.get(role.key, "")
            self.vars[f"motive57_role_{role.key}"].set(
                filename if filename in available else ""
            )

    def _update_motive57_file_widgets(self) -> None:
        values = ("", *self.motive57_c3d_files)
        for combo in self.motive57_role_combos.values():
            combo.configure(values=values)
        if self.motive57_inventory_tree is not None:
            self.motive57_inventory_tree.delete(
                *self.motive57_inventory_tree.get_children()
            )
            for filename in self.motive57_c3d_files:
                self.motive57_inventory_tree.insert("", tk.END, values=(filename,))

    def _refresh_motive57_c3d_mapping(self) -> None:
        folder = self._biobuddy_c3d_folder_path()
        default_mapping_path = motive57_mapping_path(folder)
        raw_mapping_path = str(self.vars["biobuddy_c3d_mapping_json"].get()).strip()
        if (
            not raw_mapping_path
            or Path(raw_mapping_path).name == default_mapping_path.name
        ):
            self.vars["biobuddy_c3d_mapping_json"].set(str(default_mapping_path))
        mapping_path = self._motive57_mapping_json_path()
        self.motive57_c3d_files = discover_c3d_files(folder)
        self._update_motive57_file_widgets()
        if not self.motive57_c3d_files:
            self.status_var.set("Aucun C3D Motive trouvé")
            return
        payload = load_motive57_mapping(mapping_path)
        if payload is None:
            assignments = infer_motive57_role_assignments(self.motive57_c3d_files)
            save_motive57_mapping(folder, assignments, mapping_path)
            self.status_var.set(f"Mapping Motive 57 créé: {mapping_path.name}")
        else:
            assignments = assignments_from_payload(payload)
            self.status_var.set(f"Mapping Motive 57 chargé: {mapping_path.name}")
        self._set_motive57_role_assignments(assignments)

    def _infer_motive57_mapping_from_files(self) -> None:
        if not self.motive57_c3d_files:
            self._refresh_motive57_c3d_mapping()
        assignments = infer_motive57_role_assignments(self.motive57_c3d_files)
        self._set_motive57_role_assignments(assignments)
        self.status_var.set("Rôles Motive 57 inférés")

    def _save_motive57_mapping_json(self) -> None:
        folder = self._biobuddy_c3d_folder_path()
        mapping_path = self._motive57_mapping_json_path()
        output_path = save_motive57_mapping(
            folder, self._motive57_role_assignments_from_vars(), mapping_path
        )
        self.vars["biobuddy_c3d_mapping_json"].set(str(output_path))
        self.status_var.set(f"Mapping Motive 57 sauvé: {output_path.name}")

    def _load_motive57_mapping_json(self) -> None:
        mapping_path = self._motive57_mapping_json_path()
        payload = load_motive57_mapping(mapping_path)
        if payload is None:
            messagebox.showerror("JSON introuvable", str(mapping_path))
            return
        self.motive57_c3d_files = discover_c3d_files(self._biobuddy_c3d_folder_path())
        self._update_motive57_file_widgets()
        self._set_motive57_role_assignments(assignments_from_payload(payload))
        self.status_var.set(f"Mapping Motive 57 chargé: {mapping_path.name}")

    def _use_selected_motive_folder_for_biobuddy(self) -> None:
        data_root = self._resolve(str(self.vars["p6_data_root"].get()).strip())
        motive_dir = data_root / "Motive"
        if not motive_dir.exists():
            messagebox.showerror("Dossier Motive introuvable", str(motive_dir))
            return
        self.vars["biobuddy_c3d_folder"].set(str(motive_dir))
        self.vars["biobuddy_c3d_preset"].set("motive_57")
        self._refresh_motive57_c3d_mapping()
        self.status_var.set("Dossier Motive sélectionné pour BioBuddy")

    def _open_biobuddy_c3d_model_in_explorer(self) -> None:
        self.vars["model_explorer_path"].set(
            str(self.vars["biobuddy_c3d_output"].get()).strip()
        )
        self._launch_biobuddy_model_explorer()

    def _generated_biomod_path(self, source: str) -> Path:
        filename = {
            "bvh": "model_from_bvh_biobuddy.bioMod",
            "fbx": "model_from_fbx_biobuddy.bioMod",
        }[source]
        return self._resolve(str(self.vars["out_dir"].get()).strip()) / filename

    def _set_generated_model_path(self, source: str) -> None:
        path = self._generated_biomod_path(source)
        self.vars["model_explorer_path"].set(
            os.path.relpath(path, PROJECT_DIR)
            if str(path).startswith(str(PROJECT_DIR))
            else str(path)
        )
        if not path.exists():
            self.status_var.set(
                f"Le modèle {source.upper()} généré n'existe pas encore"
            )

    def _launch_biobuddy_model_explorer(self) -> None:
        raw_path = str(self.vars["model_explorer_path"].get()).strip()
        if not raw_path:
            messagebox.showerror("Modèle manquant", "Sélectionne un modèle à explorer.")
            return
        model_path = self._resolve(raw_path)
        if not model_path.exists():
            messagebox.showerror("Modèle introuvable", str(model_path))
            return

        command = [sys.executable, str(MODEL_EDITOR_SCRIPT), str(model_path)]
        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", "/private/tmp/captury_models_mplconfig")
        try:
            subprocess.Popen(command, cwd=PROJECT_DIR, env=env)
        except Exception as exc:
            messagebox.showerror(
                "BioBuddy", f"Impossible de lancer l'explorateur BioBuddy:\n{exc}"
            )
            return

        self._append_log("$ " + " ".join(shlex.quote(part) for part in command) + "\n")
        self.status_var.set("Explorateur BioBuddy lancé")

    def _open_output_dir(self) -> None:
        output_dir = self._current_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(output_dir)])
        elif os.name == "nt":
            os.startfile(output_dir)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(output_dir)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Captury/BioBuddy GUI.")
    parser.add_argument(
        "--p6-debug",
        action="store_true",
        help="Load the local P6 debug preset on startup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = CapturyBioBuddyGui()
    if args.p6_debug:
        app._load_p6_debug_preset()
    app.mainloop()


if __name__ == "__main__":
    main()
