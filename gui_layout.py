"""Tk layout helpers: preserve usable content sizes inside resizable viewports."""

import tkinter as tk
from tkinter import ttk


class ScrollableSection(ttk.Frame):
    """Expand with the window; scroll overflow rather than crushing controls/plots.

    Scroll bindings belong to this viewport's toplevel and are removed on destroy.
    Nested canvases (3D/Matplotlib) and lists keep their own wheel interactions.
    """

    def __init__(self, parent, *, padding=12):
        super().__init__(parent)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            self, width=1, height=1, highlightthickness=0, background="#f6f7f8"
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(
            yscrollcommand=vertical.set, xscrollcommand=horizontal.set
        )
        self.content = ttk.Frame(self.canvas, padding=padding)
        self.content.columnconfigure(0, weight=1)
        self._item = self.canvas.create_window(0, 0, window=self.content, anchor="nw")
        self.canvas.bind("<Configure>", self._resize)
        self.content.bind("<Configure>", self._resize)
        self._top = self.winfo_toplevel()
        self._wheel_bindings = {
            event: self._top.bind(event, self._wheel, add="+")
            for event in ("<MouseWheel>", "<Button-4>", "<Button-5>")
        }
        self.bind("<Destroy>", self._destroy_bindings, add="+")

    def _resize(self, event=None):
        width = max(self.canvas.winfo_width(), self.content.winfo_reqwidth())
        height = max(self.canvas.winfo_height(), self.content.winfo_reqheight())
        self.canvas.itemconfigure(self._item, width=width, height=height)
        self.canvas.configure(scrollregion=(0, 0, width, height))

    def _wheel(self, event):
        widget = event.widget
        while widget is not self and widget is not None:
            if widget is not self.canvas and isinstance(
                widget, (tk.Canvas, tk.Text, tk.Listbox, ttk.Treeview, ttk.Combobox)
            ):
                return
            widget = getattr(widget, "master", None)
        if widget is not self:
            return
        horizontal = bool(event.state & 1)
        view = self.canvas.xview if horizontal else self.canvas.yview
        if view() == (0.0, 1.0):
            return
        delta = getattr(event, "delta", 0)
        steps = -1 if getattr(event, "num", None) == 4 or delta > 0 else 1
        scroll = self.canvas.xview_scroll if horizontal else self.canvas.yview_scroll
        scroll(steps * 3, "units")
        return "break"

    def _destroy_bindings(self, event):
        if event.widget is self:
            for sequence, binding in self._wheel_bindings.items():
                self._top.unbind(sequence, binding)


def fit_window(window, width, height, *, offset=0):
    """Keep an initial floating figure inside the screen (including its title bar)."""
    width = min(width, max(400, window.winfo_screenwidth() - 80))
    height = min(height, max(300, window.winfo_screenheight() - 120))
    x = min(40 + offset, max(0, window.winfo_screenwidth() - width - 20))
    y = min(40 + offset, max(0, window.winfo_screenheight() - height - 80))
    window.geometry(f"{width}x{height}+{x}+{y}")
