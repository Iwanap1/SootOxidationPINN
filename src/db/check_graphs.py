import json
import math
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class GraphReviewGUI:
    """
    Review temperature-dependent experimental curves five at a time.

    Gas concentration fields reported in ppm are multiplied by gas_flow_ml_min:
        y_displayed = y_recorded * gas_flow_ml_min

    Gas concentrations already reported as percentages are plotted unchanged.

    The ppm × flow quantity is a flow-normalised signal for visual comparison.
    It is NOT converted to a true molar flow unless the concentration units and
    gas conditions are explicitly converted as well.
    """

    PAGE_SIZE = 5

    # These are fractions/selectivities/conversions and should NOT be multiplied
    # by gas flow.
    NEVER_FLOW_NORMALISE = {
        "soot_oxidation_conversion",
        "soot_oxidation_weight",
        "soot_oxidation_co2_selectivity",
        "no2_%_of_nox",
        "total_nox_conversion_%",
    }

    def __init__(self, root, json_path):
        self.root = root
        self.json_path = Path(json_path)

        with self.json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        self.entries = payload.get("create", payload) if isinstance(payload, dict) else payload
        if not isinstance(self.entries, list):
            raise ValueError("Expected a list of entries, or a dict containing a 'create' list.")

        self.curves_by_field = self._collect_curves()
        self.fields = sorted(
            self.curves_by_field,
            key=lambda field: (-len(self.curves_by_field[field]), field.lower())
        )

        if not self.fields:
            raise ValueError("No non-empty list-valued graph fields were found.")

        self.current_field = tk.StringVar(value=self.fields[0])
        self.page_start = 0

        # Keys are (field, doi, material, experiment_index)
        self.flagged = set()

        self.root.title(f"Migration graph review — {self.json_path.name}")
        self.root.geometry("1400x1000")

        self._build_controls()
        self._build_plot_area()
        self._bind_shortcuts()
        self._render_page()

    # ------------------------------------------------------------------
    # Data collection / field classification
    # ------------------------------------------------------------------

    def _collect_curves(self):
        curves_by_field = {}

        for entry_index, entry in enumerate(self.entries):
            doi = entry.get("doi", "")
            material = entry.get("material", {})
            material_name = material.get("name", "")

            for experiment_index, exp in enumerate(entry.get("experiments", [])):
                temps = exp.get("temps", [])

                if not isinstance(temps, list) or not temps:
                    continue

                for field, values in exp.items():
                    if field == "temps":
                        continue

                    if not isinstance(values, list) or not values:
                        continue

                    # Only graph arrays that align to temperature.
                    # Your validator should already enforce this, but keeping the
                    # check here stops the GUI crashing if a bad file is loaded.
                    if len(values) != len(temps):
                        continue

                    if not all(self._is_finite_number(v) for v in values):
                        continue

                    curves_by_field.setdefault(field, []).append({
                        "entry_index": entry_index,
                        "experiment_index": experiment_index,
                        "doi": doi,
                        "material": material_name,
                        "note": exp.get("note"),
                        "temps": temps,
                        "values": values,
                        "flow": exp.get("gas_flow_ml_min"),
                        "NO_ppm": exp.get("NO_ppm"),
                        "NO2_ppm": exp.get("NO2_ppm"),
                        "O2_vol%": exp.get("O2_vol%"),
                        "mass_soot": exp.get("mass_soot"),
                        "tight_contact": exp.get("tight_contact"),
                        "soot_name": exp.get("soot_name"),
                    })

        return curves_by_field

    @staticmethod
    def _is_finite_number(value):
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )

    def _is_flow_normalised_field(self, field):
        """
        Flow-normalise gas concentration signals.

        This catches:
          * soot_oxidation_co2_concentration_ppm
          * soot_oxidation_co2_concentration_%
          * soot_oxidation_co_concentration_ppm
          * soot_oxidation_cox_concentration_ppm / %
          * no_ppm
          * no2_ppm
          * nox_ppm
          * n2_ppm
          * future gas species fields ending in _ppm

        It intentionally excludes conversion/selectivity fields.
        """
        if field in self.NEVER_FLOW_NORMALISE:
            return False

        lower = field.lower()

        # Percentage concentration is already normalised as a gas fraction,
        # so leave fields such as soot_oxidation_co2_concentration_% unchanged.
        if "concentration_%" in lower or lower.endswith("_%"):
            return False

        # ppm concentration signals are multiplied by volumetric flow for
        # visual comparison between experiments run at different flow rates.
        if lower.endswith("_ppm"):
            return True

        return False

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_controls(self):
        controls = ttk.Frame(self.root, padding=8)
        controls.pack(fill="x")

        ttk.Label(controls, text="Graph type:").pack(side="left")

        self.field_combo = ttk.Combobox(
            controls,
            textvariable=self.current_field,
            values=self.fields,
            state="readonly",
            width=44,
        )
        self.field_combo.pack(side="left", padx=(6, 12))
        self.field_combo.bind("<<ComboboxSelected>>", self._on_field_change)

        ttk.Button(controls, text="◀ Previous 5", command=self.previous_page).pack(
            side="left", padx=3
        )
        ttk.Button(controls, text="Next 5 ▶", command=self.next_page).pack(
            side="left", padx=3
        )

        ttk.Button(
            controls,
            text="Save flagged",
            command=self.save_flagged,
        ).pack(side="right", padx=3)

        ttk.Button(
            controls,
            text="Clear flags on page",
            command=self.clear_page_flags,
        ).pack(side="right", padx=3)

        self.status_label = ttk.Label(self.root, padding=(10, 0, 10, 4))
        self.status_label.pack(fill="x")

        flag_bar = ttk.Frame(self.root, padding=(8, 0, 8, 4))
        flag_bar.pack(fill="x")

        ttk.Label(
            flag_bar,
            text="Flag suspicious graph:",
        ).pack(side="left")

        for i in range(self.PAGE_SIZE):
            ttk.Button(
                flag_bar,
                text=str(i + 1),
                width=4,
                command=lambda i=i: self.toggle_flag(i),
            ).pack(side="left", padx=2)

        ttk.Label(
            flag_bar,
            text="    Keyboard: ←/→ page, 1–5 flag current graph",
        ).pack(side="left", padx=8)

    def _build_plot_area(self):
        self.figure, self.ax = plt.subplots(
            figsize=(13.5, 8.5),
            constrained_layout=True,
        )

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _bind_shortcuts(self):
        self.root.bind("<Left>", lambda event: self.previous_page())
        self.root.bind("<Right>", lambda event: self.next_page())

        for number in range(1, 6):
            self.root.bind(
                str(number),
                lambda event, idx=number - 1: self.toggle_flag(idx)
            )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _render_page(self):
        field = self.current_field.get()
        curves = self.curves_by_field[field]
        total = len(curves)

        if total == 0:
            self.page_start = 0
        elif self.page_start >= total:
            self.page_start = max(
                0,
                ((total - 1) // self.PAGE_SIZE) * self.PAGE_SIZE,
            )

        normalised = self._is_flow_normalised_field(field)

        # ONE AXIS ONLY. Every curve on the current page is drawn here.
        ax = self.ax
        ax.clear()

        page_curves = curves[
            self.page_start:self.page_start + self.PAGE_SIZE
        ]

        all_y = []
        any_missing_flow = False

        for page_position, curve in enumerate(page_curves):
            x = curve["temps"]
            y_raw = curve["values"]
            flow = curve["flow"]

            missing_flow = False

            if normalised:
                if self._is_finite_number(flow) and flow > 0:
                    y = [value * flow for value in y_raw]
                else:
                    y = y_raw
                    missing_flow = True
                    any_missing_flow = True
            else:
                y = y_raw

            all_y.extend(
                value
                for value in y
                if self._is_finite_number(value)
            )

            key = self._curve_key(field, curve)
            is_flagged = key in self.flagged

            label = (
                f"{page_position + 1}. {curve['material']} "
                f"| exp {curve['experiment_index'] + 1}"
            )

            if curve.get("note") not in (None, ""):
                label += f" | {curve['note']}"

            if self._is_finite_number(flow):
                label += f" | {flow:g} mL/min"

            if is_flagged:
                label += " | FLAGGED"

            if missing_flow:
                label += " | RAW: no flow"

            ax.plot(
                x,
                y,
                marker="o",
                markersize=4,
                linewidth=1.6,
                label=label,
            )

        ax.grid(alpha=0.25)
        ax.set_xlabel("Temperature (°C)")

        if normalised:
            ax.set_ylabel(f"{field} × gas_flow_ml_min")
        else:
            ax.set_ylabel(field)

        # Keep percent-type data on a common useful scale.
        if "%" in field and not normalised and all_y:
            if min(all_y) >= -5 and max(all_y) <= 105:
                ax.set_ylim(-5, 105)

        shown_start = self.page_start + 1 if total else 0
        shown_end = min(
            self.page_start + self.PAGE_SIZE,
            total,
        )

        ax.set_title(
            f"{field} — curves {shown_start}–{shown_end} of {total}"
        )

        if page_curves:
            ax.legend(
                loc="best",
                fontsize=8,
                frameon=True,
            )

        normalisation_text = (
            "FLOW-NORMALISED: value × gas_flow_ml_min"
            if normalised
            else "raw values"
        )

        missing_text = (
            " | WARNING: displayed curve missing gas flow"
            if any_missing_flow
            else ""
        )

        self.status_label.config(
            text=(
                f"{field}: showing {shown_start}–{shown_end} of {total} curves"
                f" | {normalisation_text}"
                f" | {len(self.flagged)} total flagged"
                f"{missing_text}"
            )
        )

        self.canvas.draw_idle()

    @staticmethod
    def _metadata_text(curve):
        bits = []

        if curve.get("note") not in (None, ""):
            bits.append(f"note={curve['note']}")

        if curve.get("flow") not in (None, ""):
            bits.append(f"flow={curve['flow']} mL/min")

        if curve.get("mass_soot") not in (None, ""):
            bits.append(f"soot={curve['mass_soot']} g")

        if curve.get("NO_ppm") not in (None, ""):
            bits.append(f"NO={curve['NO_ppm']} ppm")

        if curve.get("NO2_ppm") not in (None, ""):
            bits.append(f"NO₂={curve['NO2_ppm']} ppm")

        if curve.get("O2_vol%") not in (None, ""):
            bits.append(f"O₂={curve['O2_vol%']}%")

        if curve.get("tight_contact") is not None:
            bits.append(f"tight_contact={curve['tight_contact']}")

        return "  |  ".join(bits)

    # ------------------------------------------------------------------
    # Paging / field switching
    # ------------------------------------------------------------------

    def _on_field_change(self, event=None):
        self.page_start = 0
        self._render_page()

    def previous_page(self):
        self.page_start = max(0, self.page_start - self.PAGE_SIZE)
        self._render_page()

    def next_page(self):
        field = self.current_field.get()
        total = len(self.curves_by_field[field])

        if self.page_start + self.PAGE_SIZE < total:
            self.page_start += self.PAGE_SIZE
            self._render_page()

    # ------------------------------------------------------------------
    # Flagging
    # ------------------------------------------------------------------

    @staticmethod
    def _curve_key(field, curve):
        return (
            field,
            curve["doi"],
            curve["material"],
            curve["experiment_index"],
        )

    def toggle_flag(self, page_position):
        field = self.current_field.get()
        curves = self.curves_by_field[field]
        curve_idx = self.page_start + page_position

        if curve_idx >= len(curves):
            return

        key = self._curve_key(field, curves[curve_idx])

        if key in self.flagged:
            self.flagged.remove(key)
        else:
            self.flagged.add(key)

        self._render_page()

    def clear_page_flags(self):
        field = self.current_field.get()
        curves = self.curves_by_field[field]

        for page_position in range(self.PAGE_SIZE):
            curve_idx = self.page_start + page_position
            if curve_idx < len(curves):
                self.flagged.discard(
                    self._curve_key(field, curves[curve_idx])
                )

        self._render_page()

    def save_flagged(self):
        if not self.flagged:
            messagebox.showinfo("Save flagged", "No graphs are currently flagged.")
            return

        output_path = filedialog.asksaveasfilename(
            title="Save flagged graphs",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile="flagged_graphs.json",
        )

        if not output_path:
            return

        records = []

        for field, curves in self.curves_by_field.items():
            for curve in curves:
                key = self._curve_key(field, curve)

                if key not in self.flagged:
                    continue

                records.append({
                    "field": field,
                    "doi": curve["doi"],
                    "material": curve["material"],
                    "experiment_index": curve["experiment_index"],
                    "experiment_number": curve["experiment_index"] + 1,
                    "note": curve["note"],
                    "gas_flow_ml_min": curve["flow"],
                    "mass_soot": curve["mass_soot"],
                    "NO_ppm": curve["NO_ppm"],
                    "NO2_ppm": curve["NO2_ppm"],
                    "O2_vol%": curve["O2_vol%"],
                })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        messagebox.showinfo(
            "Saved",
            f"Saved {len(records)} flagged graph(s) to:\n{output_path}"
        )


def choose_json_file():
    root = tk.Tk()
    root.withdraw()

    path = filedialog.askopenfilename(
        title="Choose migration JSON",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )

    root.destroy()
    return path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visually review migration graph data five curves at a time."
    )
    parser.add_argument(
        "json_file",
        nargs="?",
        help="Migration JSON file. If omitted, a file picker opens.",
    )
    args = parser.parse_args()

    json_path = args.json_file or choose_json_file()

    if not json_path:
        return

    root = tk.Tk()

    try:
        GraphReviewGUI(root, json_path)
    except Exception as exc:
        root.destroy()
        raise exc

    root.mainloop()


if __name__ == "__main__":
    main()