from __future__ import annotations

import inspect
import unittest

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from captury_biobuddy_gui import (
    COMMAND_MODES,
    CapturyBioBuddyGui,
    C3DMarkerData,
    TkC3DTrialCanvas,
    available_cor_layers,
    captury_marker_transform_from_c3d_layers,
    captury_marker_transform_from_report,
    data_source_color,
    data_source_marker_color,
    display_marker_name,
    graph_metric_columns,
    inventory_p6_dataset,
    joint_chain_edges,
    load_joint_centre_chain_data,
    transformed_marker_data,
    vertical_axis_label,
)
from gui_trial_viewer import is_joint_centre_marker_label, marker_display_labels
from gui_trial_viewer import JointCentreChainData


class FakeVar:
    def __init__(self, value: object = "") -> None:
        self.value = value

    def get(self) -> object:
        return self.value

    def set(self, value: object) -> None:
        self.value = value


class FakeButton:
    def __init__(self) -> None:
        self.state = "normal"

    def configure(self, **kwargs: object) -> None:
        if "state" in kwargs:
            self.state = str(kwargs["state"])


class FakeViewer:
    def __init__(self) -> None:
        self.visible_cor_layers: list[str] = []
        self.show_chain_axes = False
        self.rotate_body_segments_180_x = False

    def set_visible_cor_layers(self, layers: list[str]) -> None:
        self.visible_cor_layers = layers

    def set_show_chain_axes(self, show: bool) -> None:
        self.show_chain_axes = show

    def set_rotate_body_segments_180_x(self, enabled: bool) -> None:
        self.rotate_body_segments_180_x = enabled


class P6DebugGuiTests(unittest.TestCase):
    def make_gui_stub(self) -> CapturyBioBuddyGui:
        gui = object.__new__(CapturyBioBuddyGui)
        keys = [
            "command_mode",
            "p6_data_root",
            "p6_out_dir",
            "p6_trials",
            "selected_trial",
            "p6_static_trial",
            "p6_cut_mode",
            "p6_temporal_sync_mode",
            "p6_manual_lag_s",
            "p6_max_lag_s",
            "p6_phase_normalization_points",
            "p6_time_start",
            "p6_time_end",
            "p6_joint_filter",
            "p6_joint_centre_reference",
            "p6_auto_analyze",
            "p6_model_source",
            "p6_model_to_c3d_axis",
            "p6_captury_model_to_c3d_axis",
            "p6_motive_model_to_c3d_axis",
            "p6_captury_q_landmarks_json",
            "p6_segment_reference",
            "p6_captury_reorient_thigh_y_from_cor",
            "p6_captury_rename_axes",
            "p6_captury_calibrate_segment_frames",
            "p6_rotate_body_segments_180_x",
            "p6_reexpress_rotations_zxy",
            "p6_disable_static_model_alignment",
            "p6_disable_motive_marker_alignment",
            "root_offset_mode",
            "c3d_angle_unit",
            "p6_no_figures",
            "p6_no_cache",
            "p6_no_mesh",
            "p6_max_mesh_points",
            "p6_run_ik_batch",
            "p6_ik_max_frames",
            "p6_visualize",
            "p6_visualize_trial",
            "p6_headless",
            "p6_rerun_wait_seconds",
            "biobuddy_c3d_folder",
            "biobuddy_c3d_mapping_json",
            "biobuddy_c3d_output",
            "model_explorer_path",
        ]
        gui.vars = {key: FakeVar("") for key in keys}
        for source in ("captury", "motive", "biobuddy"):
            gui.vars[f"p6_source_{source}"] = FakeVar(True)
        gui.vars["p6_no_figures"].set(False)
        gui.vars["root_offset_mode"].set("auto")
        gui.vars["c3d_angle_unit"].set("deg")
        gui.vars["p6_auto_analyze"].set(True)
        gui.vars["p6_no_cache"].set(False)
        gui.vars["p6_segment_reference"].set("biobuddy")
        gui.vars["p6_joint_centre_reference"].set("biobuddy")
        gui.vars["p6_captury_reorient_thigh_y_from_cor"].set(False)
        gui.vars["p6_captury_model_to_c3d_axis"].set("auto")
        gui.vars["p6_motive_model_to_c3d_axis"].set("auto")
        gui.vars["p6_captury_rename_axes"].set(False)
        gui.vars["p6_captury_calibrate_segment_frames"].set(False)
        gui.vars["p6_rotate_body_segments_180_x"].set(False)
        gui.vars["p6_reexpress_rotations_zxy"].set(False)
        gui.vars["p6_disable_static_model_alignment"].set(False)
        gui.vars["p6_disable_motive_marker_alignment"].set(False)
        gui.vars["p6_no_mesh"].set(False)
        gui.vars["p6_run_ik_batch"].set(False)
        gui.vars["p6_visualize"].set(False)
        gui.vars["p6_headless"].set(False)
        gui.vars["biobuddy_c3d_folder"].set("")
        gui.vars["biobuddy_c3d_mapping_json"].set("")
        gui.vars["biobuddy_c3d_output"].set("/tmp/motive_57.bioMod")
        gui.vars["model_explorer_path"].set("")
        gui.occlusion_sort_column = "marker_order"
        gui.occlusion_sort_descending = False
        gui.process = None
        gui.running_command_mode = None
        gui.pending_auto_analysis = False
        gui.auto_analysis_after_id = None
        gui.auto_enable_biobuddy_cor_after_refresh = False
        gui.trial_inventory = {}
        gui.joint_chain_cache = {}
        gui.status_var = FakeVar()
        return gui

    def test_temporal_gui_validation_accepts_finite_manual_settings(self) -> None:
        gui = self.make_gui_stub()
        with tempfile.TemporaryDirectory() as tmp:
            gui.vars["p6_data_root"].set(tmp)
            gui.vars["p6_temporal_sync_mode"].set("manual")
            gui.vars["p6_manual_lag_s"].set("0.125")
            gui.vars["p6_max_lag_s"].set("0.5")
            gui.vars["p6_phase_normalization_points"].set("101")

            self.assertTrue(CapturyBioBuddyGui._validate_p6_analysis(gui))

    def test_captury_frame_modes_are_mutually_exclusive(self) -> None:
        gui = self.make_gui_stub()
        gui.vars["p6_captury_rename_axes"].set(True)
        gui.vars["p6_captury_calibrate_segment_frames"].set(True)

        CapturyBioBuddyGui._enforce_captury_frame_mode(gui, "calibrate")

        self.assertTrue(gui.vars["p6_captury_calibrate_segment_frames"].get())
        self.assertFalse(gui.vars["p6_captury_rename_axes"].get())

    def test_temporal_gui_validation_rejects_invalid_settings(self) -> None:
        gui = self.make_gui_stub()
        with tempfile.TemporaryDirectory() as tmp, patch(
            "captury_biobuddy_gui.messagebox.showerror"
        ) as showerror:
            gui.vars["p6_data_root"].set(tmp)
            gui.vars["p6_temporal_sync_mode"].set("manual")
            gui.vars["p6_manual_lag_s"].set("")
            gui.vars["p6_max_lag_s"].set("0.5")
            gui.vars["p6_phase_normalization_points"].set("101")

            self.assertFalse(CapturyBioBuddyGui._validate_p6_analysis(gui))
            self.assertTrue(showerror.called)

            showerror.reset_mock()
            gui.vars["p6_manual_lag_s"].set("0.0")
            gui.vars["p6_max_lag_s"].set("-1")
            self.assertFalse(CapturyBioBuddyGui._validate_p6_analysis(gui))
            self.assertTrue(showerror.called)

    def test_running_state_disables_all_analysis_buttons(self) -> None:
        gui = self.make_gui_stub()
        first = FakeButton()
        second = FakeButton()
        stop = FakeButton()
        gui.analysis_buttons = [first, second]  # type: ignore[list-item]
        gui.stop_button = stop  # type: ignore[assignment]

        CapturyBioBuddyGui._set_running(gui, True)

        self.assertEqual(first.state, "disabled")
        self.assertEqual(second.state, "disabled")
        self.assertEqual(stop.state, "normal")
        self.assertEqual(gui.status_var.get(), "Exécution en cours")

        CapturyBioBuddyGui._set_running(gui, False)

        self.assertEqual(first.state, "normal")
        self.assertEqual(second.state, "normal")
        self.assertEqual(stop.state, "disabled")
        self.assertEqual(gui.status_var.get(), "Prêt")

    def test_new_analysis_button_is_disabled_when_process_is_running(self) -> None:
        gui = self.make_gui_stub()
        gui.process = object()
        gui.analysis_buttons = []
        button = FakeButton()

        tracked = CapturyBioBuddyGui._register_analysis_button(gui, button)  # type: ignore[arg-type]

        self.assertIs(tracked, button)
        self.assertEqual(button.state, "disabled")
        self.assertEqual(gui.analysis_buttons, [button])

    def test_biobuddy_cor_checkbox_is_disabled_until_layer_exists(self) -> None:
        gui = self.make_gui_stub()
        gui.viewer_cor_layer_vars = {
            "captury": FakeVar(True),
            "motive": FakeVar(True),
            "biobuddy": FakeVar(True),
        }
        captury_button = FakeButton()
        motive_button = FakeButton()
        biobuddy_button = FakeButton()
        gui.viewer_cor_layer_checks = {
            "captury": captury_button,
            "motive": motive_button,
            "biobuddy": biobuddy_button,
        }
        gui.embedded_viewer = SimpleNamespace(
            chain_data=JointCentreChainData(
                layers={
                    "captury": {"Hips": np.zeros((1, 3))},
                    "motive": {"Hips": np.zeros((1, 3))},
                },
                edges=[],
            )
        )

        CapturyBioBuddyGui._update_cor_layer_check_states(gui)

        self.assertEqual(captury_button.state, "normal")
        self.assertEqual(motive_button.state, "normal")
        self.assertEqual(biobuddy_button.state, "disabled")
        self.assertFalse(gui.viewer_cor_layer_vars["biobuddy"].get())

        gui.embedded_viewer.chain_data = JointCentreChainData(
            layers={
                "captury": {"Hips": np.zeros((1, 3))},
                "motive": {"Hips": np.zeros((1, 3))},
                "biobuddy": {"Hips": np.zeros((1, 3))},
            },
            edges=[],
        )

        CapturyBioBuddyGui._update_cor_layer_check_states(gui)

        self.assertEqual(biobuddy_button.state, "normal")

    def test_biobuddy_cor_checkbox_auto_enables_once_after_static_refresh(self) -> None:
        gui = self.make_gui_stub()
        with tempfile.TemporaryDirectory() as tmp:
            biomod = Path(tmp) / "motive_57.bioMod"
            biomod.write_text("model\n", encoding="utf-8")
            gui.vars["biobuddy_c3d_output"].set(str(biomod))
            gui.auto_enable_biobuddy_cor_after_refresh = True
            gui.viewer_cor_layer_vars = {
                "captury": FakeVar(False),
                "motive": FakeVar(False),
                "biobuddy": FakeVar(False),
            }
            biobuddy_button = FakeButton()
            gui.viewer_cor_layer_checks = {"biobuddy": biobuddy_button}
            gui.embedded_viewer = FakeViewer()
            gui.embedded_viewer.chain_data = JointCentreChainData(
                layers={"biobuddy": {"Hips": np.zeros((1, 3))}},
                edges=[],
            )
            gui.viewer_chain_axes_var = FakeVar(False)

            CapturyBioBuddyGui._update_cor_layer_check_states(gui)
            CapturyBioBuddyGui._auto_enable_biobuddy_cor_layer_after_refresh(gui)

        self.assertEqual(biobuddy_button.state, "normal")
        self.assertTrue(gui.viewer_cor_layer_vars["biobuddy"].get())
        self.assertFalse(gui.auto_enable_biobuddy_cor_after_refresh)

    def test_biobuddy_cor_checkbox_does_not_recheck_after_manual_uncheck(self) -> None:
        gui = self.make_gui_stub()
        with tempfile.TemporaryDirectory() as tmp:
            biomod = Path(tmp) / "motive_57.bioMod"
            biomod.write_text("model\n", encoding="utf-8")
            gui.vars["biobuddy_c3d_output"].set(str(biomod))
            gui.auto_enable_biobuddy_cor_after_refresh = False
            gui.viewer_cor_layer_vars = {"biobuddy": FakeVar(False)}
            biobuddy_button = FakeButton()
            gui.viewer_cor_layer_checks = {"biobuddy": biobuddy_button}
            gui.embedded_viewer = SimpleNamespace(
                chain_data=JointCentreChainData(
                    layers={"biobuddy": {"Hips": np.zeros((1, 3))}},
                    edges=[],
                )
            )

            CapturyBioBuddyGui._update_cor_layer_check_states(gui)

        self.assertEqual(biobuddy_button.state, "normal")
        self.assertFalse(gui.viewer_cor_layer_vars["biobuddy"].get())

    def test_restored_biobuddy_layer_is_available_but_not_checked(self) -> None:
        gui = self.make_gui_stub()
        gui.viewer_cor_layer_vars = {
            "captury": FakeVar(True),
            "motive": FakeVar(True),
            "biobuddy": FakeVar(False),
        }
        biobuddy_button = FakeButton()
        gui.viewer_cor_layer_checks = {"biobuddy": biobuddy_button}
        gui.embedded_viewer = SimpleNamespace(
            chain_data=JointCentreChainData(
                layers={"biobuddy": {"Hips": np.zeros((1, 3))}},
                edges=[],
            )
        )

        CapturyBioBuddyGui._update_cor_layer_check_states(gui)

        self.assertEqual(biobuddy_button.state, "normal")
        self.assertFalse(gui.viewer_cor_layer_vars["biobuddy"].get())

    def test_visible_cor_update_applies_rotate_body_segments_to_viewer(self) -> None:
        gui = self.make_gui_stub()
        gui.viewer_cor_layer_vars = {
            "captury": FakeVar(True),
            "motive": FakeVar(False),
            "biobuddy": FakeVar(False),
        }
        gui.viewer_cor_layer_checks = {}
        gui.viewer_chain_axes_var = FakeVar(True)
        gui.embedded_viewer = FakeViewer()
        gui.vars["p6_rotate_body_segments_180_x"].set(True)

        CapturyBioBuddyGui._update_visible_cor_layers(gui)

        self.assertEqual(gui.embedded_viewer.visible_cor_layers, ["captury"])
        self.assertTrue(gui.embedded_viewer.show_chain_axes)
        self.assertTrue(gui.embedded_viewer.rotate_body_segments_180_x)

    def test_biobuddy_c3d_model_message_reports_failed_process(self) -> None:
        gui = self.make_gui_stub()

        level, title, message = CapturyBioBuddyGui._biobuddy_c3d_model_creation_message(
            gui, 2
        )

        self.assertEqual(level, "error")
        self.assertEqual(title, "Création BioBuddy échouée")
        self.assertIn("code 2", message)

    def test_biobuddy_c3d_model_message_reports_missing_output(self) -> None:
        gui = self.make_gui_stub()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "missing.bioMod"
            gui.vars["biobuddy_c3d_output"].set(str(output))

            level, title, message = (
                CapturyBioBuddyGui._biobuddy_c3d_model_creation_message(gui, 0)
            )

        self.assertEqual(level, "error")
        self.assertEqual(title, "Modèle BioBuddy introuvable")
        self.assertIn("n'existe pas", message)

    def test_biobuddy_c3d_model_message_reports_created_model(self) -> None:
        gui = self.make_gui_stub()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "motive_57.bioMod"
            output.write_text("BioBuddy model\n")
            gui.vars["biobuddy_c3d_output"].set(str(output))

            level, title, message = (
                CapturyBioBuddyGui._biobuddy_c3d_model_creation_message(gui, 0)
            )

        self.assertEqual(level, "info")
        self.assertEqual(title, "Modèle BioBuddy créé")
        self.assertIn("motive_57.bioMod", message)
        self.assertIn("IK non linéaire TRF", message)

    def test_successful_biobuddy_model_creation_starts_static_ik_without_popup(
        self,
    ) -> None:
        gui = self.make_gui_stub()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "motive_57.bioMod"
            output.write_text("BioBuddy model\n", encoding="utf-8")
            gui.vars["biobuddy_c3d_output"].set(str(output))
            calls = []
            gui._run_biobuddy_static_ik_after_model_creation = (  # type: ignore[method-assign]
                lambda: calls.append("ik")
            )
            gui._append_log = lambda text: calls.append(text)  # type: ignore[method-assign]

            with patch("captury_biobuddy_gui.messagebox.showinfo") as showinfo:
                CapturyBioBuddyGui._notify_biobuddy_c3d_model_creation_finished(gui, 0)

        showinfo.assert_not_called()
        self.assertIn("ik", calls)
        self.assertEqual(gui.vars["model_explorer_path"].get(), str(output))

    def test_successful_process_invalidates_joint_chain_cache_before_refresh(
        self,
    ) -> None:
        gui = self.make_gui_stub()
        gui.joint_chain_cache = {("old", 1, 1): None}  # type: ignore[dict-item]

        CapturyBioBuddyGui._invalidate_output_caches(gui)

        self.assertEqual(gui.joint_chain_cache, {})

    def test_p6_debug_preset_populates_fast_static_analysis(self) -> None:
        gui = self.make_gui_stub()

        CapturyBioBuddyGui._load_p6_debug_preset(gui)

        self.assertEqual(gui.vars["command_mode"].get(), COMMAND_MODES["kinematic"])
        self.assertEqual(
            gui.vars["p6_data_root"].get(), "local_trials/2026-06-30_P6_flat"
        )
        self.assertEqual(gui.vars["p6_out_dir"].get(), "out_p6_motive_captury_debug")
        self.assertEqual(gui.vars["p6_trials"].get(), "Static")
        self.assertEqual(gui.vars["selected_trial"].get(), "Static")
        self.assertEqual(gui.vars["p6_cut_mode"].get(), "manual")
        self.assertEqual(gui.vars["p6_time_start"].get(), "")
        self.assertEqual(gui.vars["p6_time_end"].get(), "")
        self.assertEqual(gui.vars["p6_joint_filter"].get(), "Hip|Knee|Ankle|Leg|Foot")
        self.assertIs(gui.vars["p6_auto_analyze"].get(), True)
        self.assertEqual(gui.vars["p6_model_source"].get(), "bvh")
        self.assertEqual(gui.vars["p6_model_to_c3d_axis"].get(), "auto")
        self.assertEqual(gui.vars["p6_segment_reference"].get(), "biobuddy")
        self.assertIs(gui.vars["p6_no_mesh"].get(), True)
        self.assertIs(gui.vars["p6_headless"].get(), True)
        self.assertEqual(gui.vars["p6_rerun_wait_seconds"].get(), "0")
        self.assertEqual(gui.status_var.get(), "Preset P6 debug chargé")

    def test_trial_inventory_refresh_loads_default_motive_folder_when_empty(
        self,
    ) -> None:
        gui = self.make_gui_stub()
        gui.trial_combobox = FakeButton()  # type: ignore[assignment]
        gui._update_inventory_table = lambda: None  # type: ignore[method-assign]
        gui._update_embedded_trial_viewer = lambda: None  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            motive_dir = data_root / "Motive"
            motive_dir.mkdir()
            gui.vars["p6_data_root"].set(str(data_root))
            gui.vars["biobuddy_c3d_folder"].set("")
            gui.motive57_role_combos = {}
            calls = []
            gui._resolve = lambda value: Path(value)  # type: ignore[method-assign]
            gui._refresh_motive57_c3d_mapping = lambda: calls.append(  # type: ignore[method-assign]
                CapturyBioBuddyGui._biobuddy_c3d_folder_path(gui)
            )

            CapturyBioBuddyGui._refresh_trial_inventory(gui)

        self.assertEqual(calls, [motive_dir])

    def test_p6_debug_command_targets_kinematic_cli(self) -> None:
        gui = self.make_gui_stub()
        CapturyBioBuddyGui._load_p6_debug_preset(gui)

        args = CapturyBioBuddyGui._p6_args(gui)

        self.assertTrue(args[1].endswith("compare_p6_motive_captury.py"))
        self.assertIn("--trial", args)
        self.assertIn("Static", args)
        self.assertIn("--cut-mode", args)
        self.assertIn("manual", args)
        self.assertIn("--no-mesh", args)
        self.assertIn("--headless", args)
        self.assertNotIn("--no-cache", args)
        self.assertNotIn("--run-ik-batch", args)
        self.assertNotIn("--visualize", args)

    def test_p6_debug_command_can_disable_alignment_diagnostics(self) -> None:
        gui = self.make_gui_stub()
        CapturyBioBuddyGui._load_p6_debug_preset(gui)
        gui.vars["p6_disable_static_model_alignment"].set(True)
        gui.vars["p6_disable_motive_marker_alignment"].set(True)

        args = CapturyBioBuddyGui._p6_args(gui)

        self.assertIn("--disable-static-model-alignment", args)
        self.assertIn("--disable-motive-marker-alignment", args)

    def test_p6_debug_command_sets_segment_reference(self) -> None:
        gui = self.make_gui_stub()
        CapturyBioBuddyGui._load_p6_debug_preset(gui)
        gui.vars["p6_segment_reference"].set("motive")

        args = CapturyBioBuddyGui._p6_args(gui)

        self.assertIn("--segment-reference", args)
        option_index = args.index("--segment-reference")
        self.assertEqual(args[option_index + 1], "motive")

    def test_p6_debug_command_sets_segment_orientation_corrections(self) -> None:
        gui = self.make_gui_stub()
        CapturyBioBuddyGui._load_p6_debug_preset(gui)
        gui.vars["p6_captury_reorient_thigh_y_from_cor"].set(True)
        gui.vars["p6_rotate_body_segments_180_x"].set(True)
        gui.vars["p6_reexpress_rotations_zxy"].set(True)

        args = CapturyBioBuddyGui._p6_args(gui)

        self.assertIn("--captury-reorient-thigh-y-from-cor", args)
        self.assertIn("--rotate-body-segments-180-x", args)
        self.assertIn("--reexpress-rotations-zxy", args)

    def test_loading_tab_exposes_rotate_body_segments_option(self) -> None:
        source = inspect.getsource(CapturyBioBuddyGui._build_loading_matching_tab)

        self.assertIn('"R(x,180°)R(y,180°)"', source)
        self.assertIn('"p6_rotate_body_segments_180_x"', source)

    def test_loading_tab_exposes_biorbd_model_without_manual_trial_field(self) -> None:
        source = inspect.getsource(CapturyBioBuddyGui._build_loading_matching_tab)

        self.assertIn('"Modèle biorbd"', source)
        self.assertIn('"biobuddy_c3d_output"', source)
        self.assertIn('"Offset trans. racine"', source)
        self.assertNotIn('"Offset racine"', source)
        self.assertNotIn('"Essais"', source)

    def test_biorbd_model_status_reports_missing_and_existing_model(self) -> None:
        gui = self.make_gui_stub()
        gui.biobuddy_biomod_status_var = FakeVar("")

        gui.vars["biobuddy_c3d_output"].set("/tmp/does_not_exist_for_captury.bioMod")
        CapturyBioBuddyGui._update_biobuddy_biomod_status(gui)
        self.assertIn("créer le modèle", str(gui.biobuddy_biomod_status_var.get()))

        with tempfile.TemporaryDirectory() as directory:
            biomod = Path(directory) / "motive_57.bioMod"
            biomod.write_text("model\n", encoding="utf-8")
            gui.vars["biobuddy_c3d_output"].set(str(biomod))
            CapturyBioBuddyGui._update_biobuddy_biomod_status(gui)

        self.assertIn(
            "Modèle biorbd disponible", str(gui.biobuddy_biomod_status_var.get())
        )

    def test_running_status_messages_are_specific_for_biobuddy_steps(self) -> None:
        gui = self.make_gui_stub()

        self.assertIn(
            "Création du modèle BioBuddy",
            CapturyBioBuddyGui._running_status_message(gui, "biobuddy_c3d_model"),
        )
        self.assertIn(
            "IK non linéaire TRF statique",
            CapturyBioBuddyGui._running_status_message(gui, "biobuddy_c3d_ik"),
        )

    def test_biobuddy_ik_progress_line_updates_gui_status(self) -> None:
        gui = self.make_gui_stub()
        line = (
            "[BioBuddy IK] frame 20/100 | temps 2.5 s | nfev 4 | "
            "erreur 0.42 mm | reste ~10.0 s\n"
        )

        CapturyBioBuddyGui._update_status_from_process_output(gui, line)

        self.assertIn("20/100", str(gui.status_var.get()))
        self.assertIn("nfev 4", str(gui.status_var.get()))
        self.assertIn("0.42 mm", str(gui.status_var.get()))
        self.assertIn("10.0 s", str(gui.status_var.get()))

    def test_manual_phase_bounds_are_sorted_and_set_manual_cut_mode(self) -> None:
        gui = self.make_gui_stub()

        CapturyBioBuddyGui._set_manual_phase_bounds(gui, 4.25, 1.5)

        self.assertEqual(gui.vars["p6_cut_mode"].get(), "manual")
        self.assertEqual(gui.vars["p6_time_start"].get(), "1.5")
        self.assertEqual(gui.vars["p6_time_end"].get(), "4.25")
        self.assertEqual(gui.status_var.get(), "Phase sélectionnée: 1.5-4.25 s")

    def test_manual_phase_bounds_parse_existing_entries(self) -> None:
        gui = self.make_gui_stub()
        gui.vars["p6_time_start"].set("0.500000")
        gui.vars["p6_time_end"].set("2")

        self.assertEqual(CapturyBioBuddyGui._manual_phase_bounds(gui), (0.5, 2.0))

    def test_p6_command_can_force_recompute_without_cache(self) -> None:
        gui = self.make_gui_stub()
        CapturyBioBuddyGui._load_p6_debug_preset(gui)
        gui.vars["p6_no_cache"].set(True)

        args = CapturyBioBuddyGui._p6_args(gui)

        self.assertIn("--no-cache", args)

    def test_p6_command_includes_manual_time_window_when_set(self) -> None:
        gui = self.make_gui_stub()
        CapturyBioBuddyGui._load_p6_debug_preset(gui)
        gui.vars["p6_time_start"].set("0.5")
        gui.vars["p6_time_end"].set("2.0")

        args = CapturyBioBuddyGui._p6_args(gui)

        self.assertIn("--time-start", args)
        self.assertIn("0.5", args)
        self.assertIn("--time-end", args)
        self.assertIn("2.0", args)

    def test_p6_occlusions_command_targets_selected_trial_only(self) -> None:
        gui = self.make_gui_stub()
        gui.vars["p6_data_root"].set("local_trials/2026-06-30_P6_flat")
        gui.vars["p6_out_dir"].set("out_p6_motive_captury_debug")

        args = CapturyBioBuddyGui._p6_occlusions_args(gui, "Marche_001")

        self.assertIn("--occlusions-only", args)
        self.assertIn("--no-figures", args)
        self.assertIn("--trial", args)
        self.assertIn("Marche_001", args)

    def test_p6_auto_analysis_command_is_lightweight_for_selected_trial(self) -> None:
        gui = self.make_gui_stub()
        gui.vars["p6_data_root"].set("local_trials/2026-06-30_P6_flat")
        gui.vars["p6_out_dir"].set("out_p6_motive_captury_debug")
        gui.vars["p6_static_trial"].set("Static")
        gui.vars["p6_cut_mode"].set("manual")
        gui.vars["p6_model_source"].set("bvh")
        gui.vars["p6_model_to_c3d_axis"].set("auto")

        args = CapturyBioBuddyGui._p6_auto_analysis_args(gui, "Marche_001")

        self.assertIn("--trial", args)
        self.assertIn("Marche_001", args)
        self.assertIn("--no-figures", args)
        self.assertIn("--no-mesh", args)
        self.assertIn("--max-mesh-points", args)
        self.assertNotIn("--run-ik-batch", args)
        self.assertNotIn("--visualize", args)
        self.assertNotIn("--occlusions-only", args)

    def test_selected_trial_runs_auto_analysis_when_idle(self) -> None:
        gui = self.make_gui_stub()
        gui.vars["p6_data_root"].set("local_trials/2026-06-30_P6_flat")
        gui.vars["p6_out_dir"].set("out_p6_motive_captury_debug")
        gui.vars["selected_trial"].set("Static")
        gui.trial_inventory = {"Static": {"Motive": {}}}
        calls = []
        gui._resolve = lambda value: Path(".")  # type: ignore[method-assign]
        gui._run_args = lambda args, **_kwargs: calls.append(args)  # type: ignore[method-assign]

        CapturyBioBuddyGui._run_selected_trial_auto_analysis(gui)

        self.assertEqual(len(calls), 1)
        self.assertIn("--no-figures", calls[0])
        self.assertNotIn("--occlusions-only", calls[0])

    def test_selected_trial_does_not_run_auto_analysis_while_busy(self) -> None:
        gui = self.make_gui_stub()
        gui.process = object()
        gui.vars["selected_trial"].set("Static")
        gui.trial_inventory = {"Static": {"Motive": {}}}
        calls = []
        gui._run_args = lambda args, **_kwargs: calls.append(args)  # type: ignore[method-assign]

        CapturyBioBuddyGui._run_selected_trial_auto_analysis(gui)

        self.assertEqual(calls, [])
        self.assertTrue(gui.pending_auto_analysis)

    def test_pending_auto_analysis_runs_after_current_process_finishes(self) -> None:
        gui = self.make_gui_stub()
        gui.pending_auto_analysis = True
        gui.vars["p6_data_root"].set("local_trials/2026-06-30_P6_flat")
        gui.vars["p6_out_dir"].set("out_p6_motive_captury_debug")
        gui.vars["selected_trial"].set("Static")
        gui.vars["p6_static_trial"].set("Static")
        gui.vars["p6_segment_reference"].set("captury")
        gui.trial_inventory = {"Static": {"Motive": {}, "Captury": {}}}
        calls = []
        gui._resolve = lambda value: Path(".")  # type: ignore[method-assign]
        gui._run_args = lambda args, **_kwargs: calls.append(args)  # type: ignore[method-assign]

        CapturyBioBuddyGui._run_pending_auto_analysis_if_needed(gui)

        self.assertEqual(len(calls), 1)
        option_index = calls[0].index("--segment-reference")
        self.assertEqual(calls[0][option_index + 1], "captury")
        self.assertFalse(gui.pending_auto_analysis)

    def test_selected_trial_can_disable_auto_analysis(self) -> None:
        gui = self.make_gui_stub()
        gui.vars["p6_auto_analyze"].set(False)
        gui.vars["p6_data_root"].set("local_trials/2026-06-30_P6_flat")
        gui.vars["selected_trial"].set("Static")
        gui.trial_inventory = {"Static": {"Motive": {}}}
        calls = []
        gui._resolve = lambda value: Path(".")  # type: ignore[method-assign]
        gui._run_args = lambda args, **_kwargs: calls.append(args)  # type: ignore[method-assign]

        CapturyBioBuddyGui._run_selected_trial_auto_analysis(gui)

        self.assertEqual(calls, [])

    def test_model_source_change_runs_auto_analysis_for_selected_trial(self) -> None:
        gui = self.make_gui_stub()
        gui.vars["p6_data_root"].set("local_trials/2026-06-30_P6_flat")
        gui.vars["p6_out_dir"].set("out_p6_motive_captury_debug")
        gui.vars["selected_trial"].set("Static")
        gui.vars["p6_static_trial"].set("Static")
        gui.vars["p6_model_source"].set("fbx")
        gui.vars["p6_model_to_c3d_axis"].set("auto")
        gui.trial_inventory = {"Static": {"Motive": {}, "Captury": {}}}
        calls = []
        scheduled = []
        gui._resolve = lambda value: Path(".")  # type: ignore[method-assign]
        gui._run_args = lambda args, **_kwargs: calls.append(args)  # type: ignore[method-assign]
        gui.after = lambda _ms, callback: scheduled.append(callback) or "after-1"  # type: ignore[method-assign]

        CapturyBioBuddyGui._on_p6_auto_analysis_option_changed(gui)

        self.assertEqual(calls, [])
        self.assertEqual(gui.auto_analysis_after_id, "after-1")
        scheduled[0]()

        self.assertEqual(len(calls), 1)
        self.assertIn("--model-source", calls[0])
        self.assertIn("fbx", calls[0])
        self.assertIn("--trial", calls[0])
        self.assertIn("Static", calls[0])

    def test_root_offset_change_runs_auto_analysis_for_selected_trial(self) -> None:
        gui = self.make_gui_stub()
        gui.vars["p6_data_root"].set("local_trials/2026-06-30_P6_flat")
        gui.vars["p6_out_dir"].set("out_p6_motive_captury_debug")
        gui.vars["selected_trial"].set("Static")
        gui.vars["p6_static_trial"].set("Static")
        gui.vars["root_offset_mode"].set("conserver les translations root du fichier")
        gui.trial_inventory = {"Static": {"Motive": {}, "Captury": {}}}
        calls = []
        scheduled = []
        gui._resolve = lambda value: Path(".")  # type: ignore[method-assign]
        gui._run_args = lambda args, **_kwargs: calls.append(args)  # type: ignore[method-assign]
        gui.after = lambda _ms, callback: scheduled.append(callback) or "after-1"  # type: ignore[method-assign]

        CapturyBioBuddyGui._on_p6_auto_analysis_option_changed(gui)

        scheduled[0]()

        self.assertEqual(len(calls), 1)
        option_index = calls[0].index("--root-offset-mode")
        self.assertEqual(calls[0][option_index + 1], "keep")

    def test_auto_analysis_option_changes_are_debounced(self) -> None:
        gui = self.make_gui_stub()
        gui.vars["p6_data_root"].set("local_trials/2026-06-30_P6_flat")
        gui.vars["selected_trial"].set("Static")
        gui.trial_inventory = {"Static": {"Motive": {}, "Captury": {}}}
        scheduled = []
        cancelled = []
        gui._resolve = lambda value: Path(".")  # type: ignore[method-assign]
        gui.after = lambda _ms, callback: scheduled.append(callback) or f"after-{len(scheduled)}"  # type: ignore[method-assign]
        gui.after_cancel = lambda after_id: cancelled.append(after_id)  # type: ignore[method-assign]
        gui._run_args = lambda _args: None  # type: ignore[method-assign]

        CapturyBioBuddyGui._on_p6_auto_analysis_option_changed(gui)
        CapturyBioBuddyGui._on_p6_auto_analysis_option_changed(gui)

        self.assertEqual(cancelled, ["after-1"])
        self.assertEqual(gui.auto_analysis_after_id, "after-2")
        self.assertEqual(gui.status_var.get(), "Analyse P6 planifiée: option modifiée")

    def test_inventory_p6_dataset_collects_flat_files_for_trial_dropdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Captury").mkdir()
            (root / "Motive").mkdir()
            for suffix in ("c3d", "bvh", "fbx"):
                (root / "Captury" / f"Static_P6.{suffix}").write_text(
                    "captury", encoding="utf-8"
                )
            (root / "Motive" / "P6_Static.c3d").write_text("motive", encoding="utf-8")
            (root / "Motive" / "P6_Static_Skeleton 001.bvh").write_text(
                "motive", encoding="utf-8"
            )

            inventory = inventory_p6_dataset(root)

            self.assertEqual(sorted(inventory), ["Static"])
            self.assertEqual(
                inventory["Static"]["Captury"]["c3d"].name, "Static_P6.c3d"
            )
            self.assertEqual(inventory["Static"]["Motive"]["c3d"].name, "P6_Static.c3d")

    def test_selected_trial_c3d_prefers_motive_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            motive_c3d = root / "P6_Static.c3d"
            captury_c3d = root / "Static_P6.c3d"
            motive_c3d.write_text("motive", encoding="utf-8")
            captury_c3d.write_text("captury", encoding="utf-8")
            gui = self.make_gui_stub()
            gui.trial_inventory = {
                "Static": {
                    "Captury": {"c3d": captury_c3d},
                    "Motive": {"c3d": motive_c3d},
                }
            }
            gui.vars["selected_trial"].set("Static")

            self.assertEqual(
                CapturyBioBuddyGui._selected_trial_c3d_path(gui), motive_c3d
            )
            self.assertEqual(
                CapturyBioBuddyGui._selected_trial_c3d_path_and_source(gui),
                (motive_c3d, "Motive"),
            )
            self.assertEqual(
                CapturyBioBuddyGui._selected_trial_c3d_paths(gui),
                {"Motive": motive_c3d, "Captury": captury_c3d},
            )

    def test_selected_trial_c3d_paths_accepts_empty_inventory(self) -> None:
        gui = self.make_gui_stub()
        gui.trial_inventory = {}
        gui.vars["selected_trial"].set("Static")

        self.assertEqual(CapturyBioBuddyGui._selected_trial_c3d_paths(gui), {})

    def test_graph_metric_columns_keeps_only_numeric_requested_columns(self) -> None:
        dataframe = pd.DataFrame(
            {
                "trial": ["Static"],
                "joint": ["Hip"],
                "median_error_mm": [12.0],
                "comment": ["ignore"],
            }
        )

        self.assertEqual(
            graph_metric_columns(dataframe, ("median_error_mm", "comment", "missing")),
            ["median_error_mm"],
        )

    def test_metric_series_keeps_multiple_selected_metrics(self) -> None:
        gui = self.make_gui_stub()
        dataframe = pd.DataFrame(
            {
                "trial": ["Static", "Static"],
                "joint": ["Hip", "Knee"],
                "median_error_mm": [10.0, 20.0],
                "p95_error_mm": [15.0, 25.0],
            }
        )
        payloads = [
            {"filters": {"trial": "Static"}, "metric": "median_error_mm"},
            {"filters": {"trial": "Static"}, "metric": "p95_error_mm"},
        ]

        series = CapturyBioBuddyGui._metric_series_from_payloads(
            gui, dataframe, payloads, {"groups": ("joint",)}
        )

        self.assertEqual(
            [item["metric"] for item in series],
            ["median_error_mm", "p95_error_mm"],
        )
        self.assertEqual([len(item["values"]) for item in series], [2, 2])

    def test_kinematic_rad_values_are_displayed_in_degrees(self) -> None:
        values = pd.Series([np.pi, np.pi / 2.0])

        converted = CapturyBioBuddyGui._values_for_display(values, "bias_rad")

        np.testing.assert_allclose(converted.to_numpy(), [180.0, 90.0])
        self.assertEqual(
            CapturyBioBuddyGui._metric_display_name("bias_rad"), "bias_deg"
        )

    def test_kinematic_timeseries_rotations_are_displayed_in_degrees(self) -> None:
        values = pd.Series([np.pi, np.pi / 2.0])

        converted = CapturyBioBuddyGui._values_for_display(
            values, "captury", q_name="Head_rotX"
        )

        np.testing.assert_allclose(converted.to_numpy(), [180.0, 90.0])
        self.assertEqual(
            CapturyBioBuddyGui._metric_display_name("captury", q_name="Head_rotX"),
            "captury (deg)",
        )

    def test_captury_marker_transform_composes_static_and_model_marker_alignment(
        self,
    ) -> None:
        static_rotation = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        static_translation = np.asarray([10.0, 20.0, 30.0])
        yaw_rotation = np.asarray([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        yaw_translation = np.asarray([-5.0, 2.0, 1.0])
        report = {
            "alignment": {
                "rotation": static_rotation.tolist(),
                "translation_mm": static_translation.tolist(),
                "motive_model_to_c3d_markers": {
                    "rotation": yaw_rotation.tolist(),
                    "translation_mm": yaw_translation.tolist(),
                },
            }
        }

        rotation, translation = captury_marker_transform_from_report(report)

        np.testing.assert_allclose(rotation, static_rotation @ yaw_rotation)
        np.testing.assert_allclose(
            translation, static_translation @ yaw_rotation + yaw_translation
        )

    def test_transformed_marker_data_keeps_metadata_and_transforms_all_frames(
        self,
    ) -> None:
        points = np.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[10.0, 20.0], [30.0, 40.0]],
                [[100.0, 200.0], [300.0, 400.0]],
            ]
        )
        data = C3DMarkerData(labels=["A", "B"], points=points, rate=120.0, unit="mm")

        transformed = transformed_marker_data(
            data, np.eye(3), np.asarray([1.0, 2.0, 3.0])
        )

        self.assertEqual(transformed.labels, ["A", "B"])
        self.assertEqual(transformed.rate, 120.0)
        self.assertEqual(transformed.unit, "mm")
        np.testing.assert_allclose(
            transformed.points, points + np.asarray([1.0, 2.0, 3.0])[:, None, None]
        )

    def test_captury_marker_transform_can_use_c3d_landmark_map(self) -> None:
        translation = np.asarray([10.0, -20.0, 30.0])
        motive_labels = [
            "Skeleton_001_LIAS",
            "Skeleton_001_RIAS",
            "Skeleton_001_LIPS",
            "Skeleton_001_RIPS",
            "Skeleton_001_LFTC",
            "Skeleton_001_RFTC",
            "Skeleton_001_LFLE",
            "Skeleton_001_LFME",
        ]
        motive_points = np.asarray(
            [
                [[0.0], [0.0], [0.0], [0.0], [100.0], [0.0], [0.0], [0.0]],
                [[0.0], [0.0], [0.0], [0.0], [0.0], [100.0], [0.0], [0.0]],
                [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [100.0], [100.0]],
            ]
        )
        captury_labels = ["Q_Wa", "Q_LT", "Q_RT", "Q_LK"]
        captury_reference_points = np.asarray(
            [
                [[0.0], [100.0], [0.0], [0.0]],
                [[0.0], [0.0], [100.0], [0.0]],
                [[0.0], [0.0], [0.0], [100.0]],
            ]
        )
        captury_points = captury_reference_points - translation[:, None, None]
        motive = C3DMarkerData(
            labels=motive_labels, points=motive_points, rate=100.0, unit="mm"
        )
        captury = C3DMarkerData(
            labels=captury_labels, points=captury_points, rate=100.0, unit="mm"
        )

        rotation, offset = captury_marker_transform_from_c3d_layers(captury, motive)

        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(offset, translation, atol=1e-12)

    def test_display_marker_name_removes_motive_skeleton_prefix(self) -> None:
        self.assertEqual(display_marker_name("Skeleton_001_LIAS"), "LIAS")

    def test_marker_display_labels_number_duplicate_names(self) -> None:
        self.assertEqual(
            marker_display_labels(["Q_Hip", "Q_Knee", "Q_Hip"]),
            ["Q_Hip#1", "Q_Knee", "Q_Hip#2"],
        )
        self.assertEqual(
            marker_display_labels(["Skeleton_001_LIAS", "Skeleton_001_LIAS"]),
            ["LIAS#1", "LIAS#2"],
        )

    def test_joint_centre_marker_labels_are_detected_for_marker_mapping(self) -> None:
        self.assertTrue(is_joint_centre_marker_label("CAPJC_Hips"))
        self.assertTrue(is_joint_centre_marker_label("MOTJC_LeftKnee"))
        self.assertTrue(is_joint_centre_marker_label("FBXJC_RightAnkle"))
        self.assertTrue(is_joint_centre_marker_label("BVHJC_RightFoot"))
        self.assertFalse(is_joint_centre_marker_label("Skeleton_001_LIAS"))
        self.assertFalse(is_joint_centre_marker_label("Q_LH#1"))

    def test_occlusion_sort_can_rank_by_missing_percent_descending(self) -> None:
        gui = self.make_gui_stub()
        gui.occlusion_sort_column = "missing_percent"
        gui.occlusion_sort_descending = True
        dataframe = pd.DataFrame(
            {
                "marker_order": [0, 1, 2],
                "marker": ["Skeleton_001_A", "Skeleton_001_B", "Skeleton_001_C"],
                "missing_percent": [0.0, 25.0, 10.0],
            }
        )

        sorted_df = CapturyBioBuddyGui._sorted_occlusion_dataframe(gui, dataframe)

        self.assertEqual(sorted_df["display_marker"].tolist(), ["B", "C", "A"])

    def test_occlusion_sort_can_keep_model_order_for_marker_column(self) -> None:
        gui = self.make_gui_stub()
        gui.occlusion_sort_column = "marker_order"
        gui.occlusion_sort_descending = False
        dataframe = pd.DataFrame(
            {
                "marker_order": [2, 0, 1],
                "marker": ["Skeleton_001_C", "Skeleton_001_A", "Skeleton_001_B"],
                "missing_percent": [0.0, 25.0, 10.0],
            }
        )

        sorted_df = CapturyBioBuddyGui._sorted_occlusion_dataframe(gui, dataframe)

        self.assertEqual(sorted_df["display_marker"].tolist(), ["A", "B", "C"])

    def test_cor_layers_are_detected_from_joint_centre_columns(self) -> None:
        layers = available_cor_layers(
            (
                "trial",
                "time",
                "joint",
                "captury_x_mm",
                "captury_y_mm",
                "captury_z_mm",
                "motive_x_mm",
                "motive_y_mm",
                "motive_z_mm",
                "biobuddy_x_mm",
                "biobuddy_y_mm",
                "biobuddy_z_mm",
            )
        )

        self.assertEqual(layers, ["captury", "motive", "biobuddy"])

    def test_data_source_colors_are_stable_for_three_sources(self) -> None:
        self.assertEqual(data_source_color("Captury"), "#f97316")
        self.assertEqual(data_source_color("Motive"), "#0ea5e9")
        self.assertEqual(data_source_color("BioBuddy"), "#22c55e")

    def test_data_source_marker_colors_use_lighter_source_nuances(self) -> None:
        self.assertEqual(data_source_marker_color("Captury"), "#fb923c")
        self.assertEqual(data_source_marker_color("Motive"), "#38bdf8")
        self.assertEqual(data_source_marker_color("BioBuddy"), "#86efac")

    def test_vertical_axis_labels_match_p6_file_conventions(self) -> None:
        self.assertEqual(vertical_axis_label("bvh"), "+Y modèle")
        self.assertEqual(vertical_axis_label("fbx"), "+Y modèle")
        self.assertEqual(vertical_axis_label("c3d"), "+Z labo")
        self.assertEqual(vertical_axis_label("c3d", "Captury"), "+Y labo")
        self.assertEqual(vertical_axis_label("c3d", "Motive"), "+Z labo")

    def test_viewer_anatomical_axis_uses_motive_left_right_markers(self) -> None:
        viewer = object.__new__(TkC3DTrialCanvas)
        viewer.frame = 0
        viewer.visible_marker_sources = {"motive"}
        data = C3DMarkerData(
            labels=["Skeleton_001_RIAS", "Skeleton_001_LIAS"],
            points=np.asarray(
                [[[0.0], [100.0]], [[0.0], [0.0]], [[900.0], [900.0]]],
                dtype=float,
            ),
            rate=120.0,
        )
        viewer.marker_layers = {"motive": data}

        axis = TkC3DTrialCanvas._anatomical_left_axis(viewer)

        np.testing.assert_allclose(axis, [1.0, 0.0, 0.0])

    def test_joint_chain_edges_keep_known_available_segments(self) -> None:
        edges = joint_chain_edges(["Hips", "Spine", "LeftUpLeg", "LeftLeg"])

        self.assertIn(("Hips", "Spine"), edges)
        self.assertIn(("Hips", "LeftUpLeg"), edges)
        self.assertIn(("LeftUpLeg", "LeftLeg"), edges)

    def test_load_joint_centre_chain_data_reads_layers_and_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "joint_centre_timeseries.npz"
            columns = np.asarray(
                [
                    "trial",
                    "time",
                    "joint",
                    "captury_x_mm",
                    "captury_y_mm",
                    "captury_z_mm",
                    "motive_x_mm",
                    "motive_y_mm",
                    "motive_z_mm",
                ]
            )
            values = {
                "trial": np.asarray(["Static", "Static", "Static", "Static"]),
                "time": np.asarray([0.0, 0.0, 1.0, 1.0]),
                "joint": np.asarray(["Hips", "Spine", "Hips", "Spine"]),
                "captury_x_mm": np.asarray([0.0, 0.0, 0.0, 0.0]),
                "captury_y_mm": np.asarray([0.0, 0.0, 1.0, 1.0]),
                "captury_z_mm": np.asarray([0.0, 10.0, 0.0, 10.0]),
                "motive_x_mm": np.asarray([1.0, 1.0, 1.0, 1.0]),
                "motive_y_mm": np.asarray([0.0, 0.0, 1.0, 1.0]),
                "motive_z_mm": np.asarray([0.0, 10.0, 0.0, 10.0]),
            }
            np.savez_compressed(
                path,
                columns=columns,
                **{
                    f"col_{index}": values[column]
                    for index, column in enumerate(columns)
                },
            )

            chain = load_joint_centre_chain_data(path)

        self.assertIsNotNone(chain)
        assert chain is not None
        self.assertEqual(sorted(chain.layers), ["captury", "motive"])
        self.assertEqual(chain.layers["captury"]["Hips"].shape, (2, 3))
        self.assertIn(("Hips", "Spine"), chain.edges)


if __name__ == "__main__":
    unittest.main()
