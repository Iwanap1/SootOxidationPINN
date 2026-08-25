from typing import Union
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

from .utils import load_model


class ModelInterface:
    GAS_INPUTS = [
        "NO_initial_ppm",
        "NO2_initial_ppm",
        "O2_vol%",
        "gas_flow_ml_min",
    ]

    MATERIAL_INPUTS = [
        "mass_catalyst",
        "tight_contact",
        "mass_silica_beads",
        "sup_calcination_temp",
        "sup_calcination_time",
        "Sbet",
    ]

    DEFAULT_OVERRIDES = {
        "NO_initial_ppm": 500.0,
        "NO2_initial_ppm": 0.0,
        "O2_vol%": 20.0,
        "gas_flow_ml_min": 100.0,
        "mass_silica_beads": 0.0,
    }

    def __init__(self, model_dir: Union[str, Path]):
        loaded = load_model(Path(model_dir), just_model=False)

        self.model = loaded["model"]
        self.scaler = loaded["scaler"]
        self.cfg = loaded["config"]

        self.model.eval()

        self.input_cols = self.cfg["nn"]["input_cols"]
        self.state_cols = self.cfg["nn"]["state"]
        self.static_cols = [col for col in self.input_cols if col not in self.state_cols]

        if self.input_cols != self.scaler.input_cols:
            raise ValueError("Config input columns do not match scaler input columns.")

        configured_static = set(self.cfg["nn"]["static_inputs"])

        # Automatically-added sup_X columns are elemental composition inputs.
        # Explicit static inputs such as sup_calcination_temp are not.
        self.metal_cols = [
            col for col in self.static_cols
            if col.startswith("sup_") and col not in configured_static
        ]

        self.non_metal_static_cols = [
            col for col in self.static_cols
            if col not in self.metal_cols
        ]

        self.defaults = {
            col: float(self.scaler.scaler.mean_[i])
            for i, col in enumerate(self.scaler.input_cols)
        }

        self.defaults.update({
            name: value
            for name, value in self.DEFAULT_OVERRIDES.items()
            if name in self.defaults
        })

        self.controls = {}
        self.metal_slots = []

        self.last_batch = None
        self.last_predictions = None

        self._build_controls()

    def _build_controls(self):
        style = {"description_width": "180px"}

        for col in self.non_metal_static_cols:
            value = float(self.defaults[col])

            if col == "tight_contact":
                self.controls[col] = widgets.Dropdown(
                    options=[("Loose", 0.0), ("Tight", 1.0)],
                    value=float(round(value)),
                    description=col,
                    style=style,
                )
            else:
                self.controls[col] = widgets.FloatText(
                    value=value,
                    description=col,
                    style=style,
                )

        metal_options = [("None", None)] + [
            (col.replace("sup_", ""), col)
            for col in self.metal_cols
        ]

        for i in range(4):
            metal = widgets.Dropdown(
                options=metal_options,
                value=None,
                description=f"Metal {i + 1}",
                style={"description_width": "80px"},
                layout=widgets.Layout(width="220px"),
            )

            fraction = widgets.BoundedFloatText(
                value=0.0,
                min=0.0,
                max=1.0,
                step=0.01,
                description="Fraction",
                style={"description_width": "70px"},
                layout=widgets.Layout(width="200px"),
            )

            self.metal_slots.append((metal, fraction))

        # Start with a physically valid single-metal catalyst.
        if "sup_Ce" in self.metal_cols:
            self.metal_slots[0][0].value = "sup_Ce"
            self.metal_slots[0][1].value = 1.0
        elif self.metal_cols:
            self.metal_slots[0][0].value = self.metal_cols[0]
            self.metal_slots[0][1].value = 1.0

        default_mass = max(
            float(self.defaults.get("mass_soot_remaining_mg", 5.0)),
            0.0,
        )

        self.controls["m_C_initial"] = widgets.FloatText(
            value=default_mass,
            description="Initial soot (mg)",
            style=style,
        )

        self.controls["start_temp_C"] = widgets.FloatText(
            value=300.0,
            description="Start T (°C)",
            style=style,
        )

        self.controls["end_temp_C"] = widgets.FloatText(
            value=700.0,
            description="End T (°C)",
            style=style,
        )

        self.controls["ramp_rate"] = widgets.FloatText(
            value=10.0,
            description="Ramp (°C/min)",
            style=style,
        )

        self.controls["duration_min"] = widgets.FloatText(
            value=60.0,
            description="Duration (min)",
            style=style,
        )

        self.controls["n_points"] = widgets.IntText(
            value=150,
            description="Plot points",
            style=style,
        )

        self.controls["x_axis"] = widgets.ToggleButtons(
            options=[
                ("Temperature", "temperature"),
                ("Time", "time"),
            ],
            value="temperature",
            description="X axis:",
            style=style,
        )

        self.run_button = widgets.Button(
            description="Run model",
            button_style="primary",
        )

        self.run_button.on_click(self._run_clicked)
        self.controls["x_axis"].observe(self._x_axis_changed, names="value")
        self.controls["ramp_rate"].observe(self._ramp_changed, names="value")

        self.output = widgets.Output()

        self._ramp_changed(None)

    def _metal_composition(self):
        # Every trained metal input starts at zero.
        composition = {col: 0.0 for col in self.metal_cols}

        selected = []
        total = 0.0

        for metal_control, fraction_control in self.metal_slots:
            metal = metal_control.value
            fraction = float(fraction_control.value)

            if metal is None:
                if fraction != 0:
                    raise ValueError(
                        "A metal fraction was provided without selecting a metal."
                    )
                continue

            if metal in selected:
                raise ValueError(
                    f"{metal.replace('sup_', '')} has been selected more than once."
                )

            if fraction <= 0:
                raise ValueError(
                    f"{metal.replace('sup_', '')} is selected but its fraction is zero."
                )

            selected.append(metal)
            composition[metal] = fraction
            total += fraction

        if not selected:
            raise ValueError("Select at least one catalyst metal.")

        if not np.isclose(total, 1.0, atol=1e-4):
            raise ValueError(
                f"Metal fractions must sum to 1. Current sum = {total:.4f}"
            )

        return composition

    def _static_dataframe(self):
        values = {
            col: float(self.controls[col].value)
            for col in self.non_metal_static_cols
        }

        values.update(self._metal_composition())

        # These are required by the scaler, although the actual dynamic state
        # is rescaled by the NODE internally during integration.
        values["mass_soot_remaining_mg"] = float(
            self.controls["m_C_initial"].value
        )

        values["temperature_K"] = (
            float(self.controls["start_temp_C"].value) + 273.15
        )

        return pd.DataFrame([
            {col: values[col] for col in self.input_cols}
        ])

    def _scaled_static_inputs(self):
        row = self.scaler.scale(self._static_dataframe())

        scaled_cols = [
            f"{col}_scaled"
            for col in self.static_cols
        ]

        values = row[scaled_cols].to_numpy(dtype=np.float32)

        return torch.tensor(
            values,
            dtype=torch.float32,
        )

    def _make_batch(self):
        start_C = float(self.controls["start_temp_C"].value)
        end_C = float(self.controls["end_temp_C"].value)
        ramp_rate = float(self.controls["ramp_rate"].value)
        duration = float(self.controls["duration_min"].value)
        m_C_initial = max(float(self.controls["m_C_initial"].value), 0.0)
        n_points = int(self.controls["n_points"].value)

        if ramp_rate < 0:
            raise ValueError("Ramp rate cannot be negative.")

        if n_points < 2:
            raise ValueError("At least two plot points are required.")

        # Isothermal
        if ramp_rate == 0:
            if duration <= 0:
                raise ValueError(
                    "Duration must be greater than zero for an isothermal experiment."
                )

            time = torch.linspace(
                0.0,
                duration,
                n_points,
                dtype=torch.float32,
            )

            temperature_K = torch.full(
                (n_points,),
                start_C + 273.15,
                dtype=torch.float32,
            )

        # Temperature ramp
        else:
            if end_C <= start_C:
                raise ValueError(
                    "End temperature must be greater than start temperature."
                )

            duration = (end_C - start_C) / ramp_rate

            time = torch.linspace(
                0.0,
                duration,
                n_points,
                dtype=torch.float32,
            )

            temperature_K = (
                start_C
                + 273.15
                + ramp_rate * time
            )

        flow_ml_min = float(
            self.controls["gas_flow_ml_min"].value
        )

        NO_ppm = float(
            self.controls["NO_initial_ppm"].value
        )

        NO2_ppm = float(
            self.controls["NO2_initial_ppm"].value
        )

        O2_fraction = (
            float(self.controls["O2_vol%"].value)
            / 100.0
        )

        if flow_ml_min <= 0:
            raise ValueError("Gas flow must be greater than zero.")

        if NO_ppm < 0 or NO2_ppm < 0:
            raise ValueError("NO and NO2 concentrations cannot be negative.")

        if not 0 <= O2_fraction <= 1:
            raise ValueError("O2 vol% must be between 0 and 100.")

        pc = self.model.physics_calculator

        F_total = (
            flow_ml_min
            / pc.MOLAR_VOLUME_STP_ML
        )

        F_NO_in = (
            NO_ppm
            * 1e-6
            * F_total
        )

        F_NO2_in = (
            NO2_ppm
            * 1e-6
            * F_total
        )

        return {
            "static_inputs_scaled": self._scaled_static_inputs(),

            "shared_time": time,

            "observation_indices": torch.arange(
                n_points,
                dtype=torch.long,
            ).unsqueeze(0),

            "observation_mask": torch.ones(
                (1, n_points),
                dtype=torch.bool,
            ),

            "temperature_K": temperature_K.unsqueeze(0),

            "start_temp_K": torch.tensor(
                [start_C + 273.15],
                dtype=torch.float32,
            ),

            "ramp_rate_K_min": torch.tensor(
                [ramp_rate],
                dtype=torch.float32,
            ),

            "m_C_initial": torch.tensor(
                [m_C_initial],
                dtype=torch.float32,
            ),

            "F_total": torch.tensor(
                [F_total],
                dtype=torch.float32,
            ),

            "F_NO_in": torch.tensor(
                [F_NO_in],
                dtype=torch.float32,
            ),

            "F_NO2_in": torch.tensor(
                [F_NO2_in],
                dtype=torch.float32,
            ),

            "o2_fraction": torch.tensor(
                [O2_fraction],
                dtype=torch.float32,
            ),
        }

    def predict(self):
        batch = self._make_batch()

        with torch.no_grad():
            predictions = self.model(batch)

        return batch, predictions

    def _array(self, predictions, name):
        if name not in predictions:
            return None

        value = predictions[name]

        if not torch.is_tensor(value):
            return None

        if value.ndim >= 2:
            value = value[0]

        return value.detach().cpu().numpy()

    def _get_x_axis(self, batch, predictions):
        if self.controls["x_axis"].value == "temperature":
            x = self._array(
                predictions,
                "temperature_K",
            ) - 273.15

            return x, "Temperature (°C)"

        x = batch["shared_time"].detach().cpu().numpy()

        return x, "Time (min)"

    def plot(self, batch, predictions):
        x, x_label = self._get_x_axis(
            batch,
            predictions,
        )

        plots = []

        if "mass_soot_remaining_mg" in predictions:
            plots.append((
                "Soot mass",
                ["mass_soot_remaining_mg"],
            ))

        if "soot_oxidation_conversion" in predictions:
            plots.append((
                "Soot conversion",
                ["soot_oxidation_conversion"],
            ))

        if "no2_fraction_of_nox" in predictions:
            names = ["no2_fraction_of_nox"]

            if "S_NO2_eq" in predictions:
                names.append("S_NO2_eq")

            plots.append((
                "NO$_2$ / NOx",
                names,
            ))

        if "soot_oxidation_co2_selectivity" in predictions:
            plots.append((
                "CO$_2$ selectivity",
                ["soot_oxidation_co2_selectivity"],
            ))

        carbon_gases = [
            name
            for name in [
                "soot_oxidation_co2_concentration_ppm",
                "soot_oxidation_co_concentration_ppm",
            ]
            if name in predictions
        ]

        if carbon_gases:
            plots.append((
                "Carbon gases",
                carbon_gases,
            ))

        nox_gases = [
            name
            for name in [
                "no2_ppm",
                "no_ppm",
                "nox_ppm",
            ]
            if name in predictions
        ]

        if nox_gases:
            plots.append((
                "NOx gases",
                nox_gases,
            ))

        # Automatically works with r1/r2/... and R1/R2/...
        rate_names = sorted(
            [
                name
                for name in predictions
                if re.fullmatch(r"[rR]\d+", name)
            ],
            key=lambda name: (
                int(name[1:]),
                name[0].isupper(),
            ),
        )

        # Each reaction rate gets its own plot.
        for name in rate_names:
            plots.append((
                name,
                [name],
            ))

        kinetic_names = [
            name
            for name in [
                "y1",
                "y2",
                "y3",
                "Da1",
                "Da2",
                "Da3",
            ]
            if name in predictions
        ]

        # These can have different units/scales, so use separate axes.
        for name in kinetic_names:
            plots.append((
                name,
                [name],
            ))

        n_cols = 3
        n_rows = int(
            np.ceil(len(plots) / n_cols)
        )

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(16, 4.2 * n_rows),
        )

        axes = np.asarray(
            axes,
        ).reshape(-1)

        labels = {
            "mass_soot_remaining_mg": "Soot mass (mg)",
            "soot_oxidation_conversion": "Soot conversion",
            "no2_fraction_of_nox": "NO$_2$ / NOx",
            "S_NO2_eq": "Thermodynamic equilibrium",
            "soot_oxidation_co2_selectivity": "CO$_2$ selectivity",
            "soot_oxidation_co2_concentration_ppm": "CO$_2$",
            "soot_oxidation_co_concentration_ppm": "CO",
            "no2_ppm": "NO$_2$",
            "no_ppm": "NO",
            "nox_ppm": "NOx",
        }

        for ax, (title, names) in zip(
            axes,
            plots,
        ):
            plotted = False

            for name in names:
                y = self._array(
                    predictions,
                    name,
                )

                if y is None:
                    continue

                if np.ndim(y) == 0:
                    continue

                finite = (
                    np.isfinite(x)
                    & np.isfinite(y)
                )

                if not finite.any():
                    continue

                if name == "S_NO2_eq":
                    ax.plot(
                        x[finite],
                        y[finite],
                        "--",
                        label=labels.get(name, name),
                    )
                else:
                    ax.plot(
                        x[finite],
                        y[finite],
                        label=labels.get(name, name),
                    )

                plotted = True

            if not plotted:
                ax.remove()
                continue

            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.grid(alpha=0.2)

            if len(names) > 1:
                ax.legend()

        for ax in axes[len(plots):]:
            ax.remove()

        fig.tight_layout()
        plt.show()

        return fig

    def _run_clicked(self, _):
        with self.output:
            clear_output(wait=True)

            try:
                self.last_batch, self.last_predictions = self.predict()

                self.plot(
                    self.last_batch,
                    self.last_predictions,
                )

            except Exception as e:
                print(
                    f"{type(e).__name__}: {e}"
                )

    def _x_axis_changed(self, change):
        if self.last_predictions is None:
            return

        with self.output:
            clear_output(wait=True)

            self.plot(
                self.last_batch,
                self.last_predictions,
            )

    def _ramp_changed(self, change):
        isothermal = (
            float(
                self.controls["ramp_rate"].value
            )
            == 0.0
        )

        self.controls[
            "end_temp_C"
        ].disabled = isothermal

        self.controls[
            "duration_min"
        ].disabled = not isothermal

    def set_inputs(self, **kwargs):
        for name, value in kwargs.items():
            if name not in self.controls:
                raise KeyError(
                    f"Unknown input: {name}"
                )

            self.controls[name].value = value

    def set_composition(self, composition):
        """
        Example:
            interface.set_composition({
                "Ce": 0.8,
                "Zr": 0.15,
                "Pr": 0.05,
            })

        Names may be supplied either as "Ce" or "sup_Ce".
        """
        if len(composition) > 4:
            raise ValueError(
                "A maximum of 4 metals can be selected."
            )

        cleaned = {}

        for name, fraction in composition.items():
            col = (
                name
                if name.startswith("sup_")
                else f"sup_{name}"
            )

            if col not in self.metal_cols:
                raise ValueError(
                    f"{name} is not an available metal input."
                )

            cleaned[col] = float(fraction)

        total = sum(
            cleaned.values()
        )

        if not np.isclose(
            total,
            1.0,
            atol=1e-4,
        ):
            raise ValueError(
                f"Metal fractions must sum to 1. Current sum = {total:.4f}"
            )

        # Reset all four slots.
        for metal, fraction in self.metal_slots:
            metal.value = None
            fraction.value = 0.0

        for i, (metal_name, fraction_value) in enumerate(
            cleaned.items()
        ):
            self.metal_slots[i][0].value = metal_name
            self.metal_slots[i][1].value = fraction_value

    def get_inputs(self):
        inputs = {
            name: control.value
            for name, control in self.controls.items()
        }

        inputs["composition"] = {
            name.replace("sup_", ""): fraction
            for name, fraction in self._metal_composition().items()
            if fraction > 0
        }

        return inputs

    def display(self):
        gas = [
            self.controls[name]
            for name in self.GAS_INPUTS
            if name in self.controls
        ]

        material = [
            self.controls[name]
            for name in self.MATERIAL_INPUTS
            if name in self.controls
        ]

        used = set(
            self.GAS_INPUTS
            + self.MATERIAL_INPUTS
        )

        other_names = [
            name
            for name in self.non_metal_static_cols
            if name not in used
        ]

        other = [
            self.controls[name]
            for name in other_names
        ]

        metal_rows = [
            widgets.HBox([
                metal,
                fraction,
            ])
            for metal, fraction in self.metal_slots
        ]

        simulation = [
            self.controls["m_C_initial"],
            self.controls["start_temp_C"],
            self.controls["end_temp_C"],
            self.controls["ramp_rate"],
            self.controls["duration_min"],
            self.controls["n_points"],
            self.controls["x_axis"],
        ]

        accordion = widgets.Accordion(
            children=[
                widgets.VBox(gas),
                widgets.VBox(material),
                widgets.VBox(metal_rows),
                widgets.VBox(other),
                widgets.VBox(simulation),
            ]
        )

        titles = [
            "Gas conditions",
            "Catalyst / reactor",
            "Catalyst composition",
            "Other inputs",
            "Simulation",
        ]

        for i, title in enumerate(titles):
            accordion.set_title(
                i,
                title,
            )

        display(
            accordion,
            self.run_button,
            self.output,
        )