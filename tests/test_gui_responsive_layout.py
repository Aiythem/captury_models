"""Desktop checks: run with CAPTURY_GUI_SMOKE=1 in a graphical session."""

import os
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get("CAPTURY_GUI_SMOKE") == "1", "requires Tk desktop")
class ResponsiveLayoutTests(unittest.TestCase):
    def test_sections_and_cycle_graph_remain_accessible(self):
        from captury_biobuddy_gui import CapturyBioBuddyGui

        with patch.object(
            CapturyBioBuddyGui,
            "_restore_gui_session_state",
            lambda app: app.vars["p6_auto_analyze"].set(False),
        ), patch.object(CapturyBioBuddyGui, "_refresh_results"), patch.object(
            CapturyBioBuddyGui, "_refresh_trial_inventory"
        ):
            app = CapturyBioBuddyGui()
        try:
            for geometry in ("980x680", "1180x780", "1500x850"):
                app.geometry(geometry)
                app.update()
                self.assertGreater(app.viewer_panel.winfo_width(), 200)
                self.assertEqual(
                    len(app.section_selector["values"]), len(app.main_notebook.tabs())
                )
                for index in range(len(app.main_notebook.tabs())):
                    app.section_selector.current(index)
                    app.section_selector.event_generate("<<ComboboxSelected>>")
                    app.update()
                    self.assertEqual(app.main_notebook.index("current"), index)
                    viewport = app.nametowidget(app.main_notebook.select())
                    self.assertGreaterEqual(
                        viewport.content.winfo_height(),
                        viewport.content.winfo_reqheight(),
                    )
                    self.assertGreaterEqual(
                        viewport.content.winfo_width(),
                        viewport.content.winfo_reqwidth(),
                    )
                app.main_notebook.select(9)
                app.update()
                canvas = app.graph_panels["gait"]["canvas"].get_tk_widget()
                self.assertGreaterEqual(canvas.winfo_height(), 300)
                viewport = app.nametowidget(app.main_notebook.select())
                viewport.canvas.yview_moveto(1)
                app.update()
                self.assertLessEqual(
                    canvas.winfo_rooty() + canvas.winfo_height(),
                    viewport.winfo_rooty() + viewport.winfo_height(),
                )
            app.viewer_visible.set(False)
            app._toggle_viewer_panel()
            app.update()
            self.assertEqual(len(app.main_body.panes()), 1)
            self.assertTrue(app.stop_button.winfo_ismapped())
            self.assertTrue(app.run_button.winfo_ismapped())
            app.viewer_visible.set(True)
            app._toggle_viewer_panel()
            app.update()
            self.assertEqual(len(app.main_body.panes()), 2)
        finally:
            app.destroy()


class CycleFigureLayoutTests(unittest.TestCase):
    def test_missing_plane_titles_fit_at_minimum_plot_size(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from gui_cycle_means import draw_selected_leg_means, selected_leg_means
        from tests.test_gui_cycle_means import fixture

        raw, report = fixture()
        curves = selected_leg_means(raw, report)
        for joint in ("hip", "knee", "ankle"):
            figure = Figure(figsize=(9, 4), dpi=100)
            canvas = FigureCanvasAgg(figure)
            axes = draw_selected_leg_means(
                figure, ".", report, joint=joint, curves=curves
            )
            canvas.draw()
            renderer = canvas.get_renderer()
            for axis in axes:
                bounds = axis.get_tightbbox(renderer)
                self.assertGreaterEqual(bounds.x0, -1)
                self.assertGreaterEqual(bounds.y0, -1)
                self.assertLessEqual(bounds.x1, 901)
                self.assertLessEqual(bounds.y1, 401)
