from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ExperimentDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        config: Dict,
        experiment_id_col: str = "_id_experiment",
        element_prefix: str = "sup_",
        non_element_cols: Optional[List[str]] = None,
        use_initial_nox: bool = True,
    ):
        self.data = data.copy()
        self.config = config
        self.experiment_id_col = experiment_id_col
        self.use_initial_nox = use_initial_nox

        if non_element_cols is None:
            non_element_cols = [
                "sup_calcination_temp",
                "sup_calcination_time",
                "sup_crystallite_size_nm",
                "sup_M_to_O_ratio",
            ]

        self.element_cols = [
            col for col in data.columns
            if col.startswith(element_prefix)
            and not col.endswith("_scaled")
            and col not in non_element_cols
        ]

        self.static_cols = (
            list(config["nn"]["static_inputs"])
            + self.element_cols
        )

        self.static_scaled_cols = [
            f"{col}_scaled"
            for col in self.static_cols
        ]

        self.target_cols = list(
            config["physics"]["targets"]
        )

        self.experiment_ids = (
            self.data[experiment_id_col]
            .drop_duplicates()
            .tolist()
        )

        self._validate_columns()

    def _validate_columns(self):
        required = (
            [self.experiment_id_col]
            + self.static_scaled_cols
            + self.target_cols
            + [
                "temperature_K",
                "start_temp_K",
                "ramp_rate_K_min",
                "mass_soot_mg",
                "F_total_mol_min",
                "O2_fraction",
            ]
        )

        if self.use_initial_nox:
            required += [
                "F_NO_initial",
                "F_NO2_initial",
            ]
        else:
            required += [
                "F_NO_in",
                "F_NO2_in",
            ]

        missing = [
            col for col in required
            if col not in self.data.columns
        ]

        if missing:
            raise ValueError(
                f"Dataset is missing required columns: {missing}"
            )

    def __len__(self):
        return len(self.experiment_ids)

    def __getitem__(self, index):
        experiment_id = self.experiment_ids[index]

        exp = self.data[self.data[self.experiment_id_col] == experiment_id].sort_values("temperature_K").copy()
        first = exp.iloc[0]

        static_inputs = torch.tensor(first[self.static_scaled_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
        temperature = torch.tensor(exp["temperature_K"].to_numpy(dtype=np.float32), dtype=torch.float32)

        start_temp = float(first["start_temp_K"])
        ramp_rate = float(first["ramp_rate_K_min"])
        ode_start = self.config["data"].get("ode_start", "experiment_start")

        if ode_start == "experiment_start":
            start_temp = float(first["start_temp_K"])
            m_C_initial = float(first["mass_soot_mg"])
            soot_initial_source = ""

        elif ode_start == "first_data_point":
            start_temp = float(first["temperature_K"])

            if bool(first["has_soot"]):
                if pd.notna(first["mass_soot_remaining_mg"]):
                    m_C_initial = float(first["mass_soot_remaining_mg"])
                    soot_initial_source = "first_data_point"
                elif pd.notna(first["mass_soot_mg"]):
                    m_C_initial = float(first["mass_soot_mg"])
                    soot_initial_source = "reported_initial_mass/conversion"
                else:
                    raise ValueError(
                        f"Experiment {experiment_id} has soot but no usable "
                        "initial soot mass."
                    )
            else:
                m_C_initial = 0.0
                soot_initial_source = "no_soot"

        else:
            raise ValueError(
                f"Unknown data.ode_start option: {ode_start}"
            )

        time = (temperature - start_temp) / ramp_rate
        m_C_initial = torch.tensor(m_C_initial, dtype=torch.float32)

        F_total = torch.tensor(
            float(first["F_total_mol_min"]),
            dtype=torch.float32,
        )

        if self.use_initial_nox:
            F_NO_in = torch.tensor(
                float(first["F_NO_initial"]),
                dtype=torch.float32,
            )

            F_NO2_in = torch.tensor(
                float(first["F_NO2_initial"]),
                dtype=torch.float32,
            )

        else:
            F_NO_in = torch.tensor(
                float(first["F_NO_in"]),
                dtype=torch.float32,
            )

            F_NO2_in = torch.tensor(
                float(first["F_NO2_in"]),
                dtype=torch.float32,
            )

        o2_fraction = torch.tensor(
            float(first["O2_fraction"]),
            dtype=torch.float32,
        )

        targets = {}
        masks = {}

        for col in self.target_cols:
            values = pd.to_numeric(
                exp[col],
                errors="coerce",
            ).to_numpy(dtype=np.float32)

            mask = np.isfinite(values)

            targets[col] = torch.tensor(
                np.nan_to_num(
                    values,
                    nan=0.0,
                ),
                dtype=torch.float32,
            )

            masks[col] = torch.tensor(
                mask,
                dtype=torch.bool,
            )
        if ode_start == "first_data_point" and soot_initial_source == "first_data_point":
            masks["mass_soot_remaining_mg"][0] = False

        return {
            "experiment_id": experiment_id,

            "static_inputs_scaled": static_inputs,

            "time": time,
            "temperature_K": temperature,
            "start_temp_K": torch.tensor(
                start_temp,
                dtype=torch.float32,
            ),
            "ramp_rate_K_min": torch.tensor(
                ramp_rate,
                dtype=torch.float32,
            ),

            "m_C_initial": m_C_initial,
            "m_C_intial_obtained_from": soot_initial_source,

            "F_total": F_total,
            "F_NO_in": F_NO_in,
            "F_NO2_in": F_NO2_in,
            "o2_fraction": o2_fraction,

            "targets": targets,
            "masks": masks,
        }


def collate_experiments(experiments):
    """
    Collate variable-length experiments into one batch.

    Each experiment retains its own observation times, but torchdiffeq
    receives one shared 1-D time vector containing the union of all
    observation times in the batch.

    Padded observation positions are masked out of the loss.
    """

    batch_size = len(experiments)

    lengths = torch.tensor(
        [len(exp["time"]) for exp in experiments],
        dtype=torch.long,
    )

    max_length = int(lengths.max())

    # ------------------------------------------------------------------
    # Shared ODE evaluation times
    # ------------------------------------------------------------------

    experiment_times = [
        torch.clamp(exp["time"], min=0.0)
        for exp in experiments
    ]

    shared_time = torch.unique(
        torch.cat([
            torch.zeros(1, dtype=experiment_times[0].dtype),
            *experiment_times,
        ]),
        sorted=True,
    )

    # For each original observation, store its location in shared_time.
    observation_indices = torch.zeros(
        batch_size,
        max_length,
        dtype=torch.long,
    )

    observation_mask = torch.zeros(
        batch_size,
        max_length,
        dtype=torch.bool,
    )

    time_padded = torch.zeros(
        batch_size,
        max_length,
        dtype=torch.float32,
    )

    temperature_padded = torch.zeros(
        batch_size,
        max_length,
        dtype=torch.float32,
    )

    for i, exp in enumerate(experiments):
        n = lengths[i].item()
        times = torch.clamp(exp["time"], min=0.0)

        indices = torch.searchsorted(
            shared_time,
            times,
        )

        observation_indices[i, :n] = indices
        observation_mask[i, :n] = True

        time_padded[i, :n] = times
        temperature_padded[i, :n] = exp["temperature_K"]

    # ------------------------------------------------------------------
    # Targets and masks
    # ------------------------------------------------------------------

    target_names = experiments[0]["targets"].keys()

    targets = {}
    masks = {}

    for name in target_names:
        target_padded = torch.zeros(
            batch_size,
            max_length,
            dtype=torch.float32,
        )

        mask_padded = torch.zeros(
            batch_size,
            max_length,
            dtype=torch.bool,
        )

        for i, exp in enumerate(experiments):
            n = lengths[i].item()

            target_padded[i, :n] = exp["targets"][name]
            mask_padded[i, :n] = exp["masks"][name]

        targets[name] = target_padded
        masks[name] = mask_padded

    # ------------------------------------------------------------------
    # Stack experiment-level inputs
    # ------------------------------------------------------------------

    return {
        "experiment_id": [
            exp["experiment_id"]
            for exp in experiments
        ],

        "static_inputs_scaled": torch.stack([
            exp["static_inputs_scaled"]
            for exp in experiments
        ]),

        # Shared integration information
        "shared_time": shared_time,
        "observation_indices": observation_indices,
        "observation_mask": observation_mask,
        "lengths": lengths,

        # Padded original observation information
        "time": time_padded,
        "temperature_K": temperature_padded,

        # Experiment-level physical quantities
        "start_temp_K": torch.stack([
            exp["start_temp_K"]
            for exp in experiments
        ]),

        "ramp_rate_K_min": torch.stack([
            exp["ramp_rate_K_min"]
            for exp in experiments
        ]),

        "m_C_initial": torch.stack([
            exp["m_C_initial"]
            for exp in experiments
        ]),

        "F_total": torch.stack([
            exp["F_total"]
            for exp in experiments
        ]),

        "F_NO_in": torch.stack([
            exp["F_NO_in"]
            for exp in experiments
        ]),

        "F_NO2_in": torch.stack([
            exp["F_NO2_in"]
            for exp in experiments
        ]),

        "o2_fraction": torch.stack([
            exp["o2_fraction"]
            for exp in experiments
        ]),

        "m_C_intial_obtained_from": [
            exp["m_C_intial_obtained_from"]
            for exp in experiments
        ],

        "targets": targets,
        "masks": masks,
    }