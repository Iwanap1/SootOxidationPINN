from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
import re

from .data_processing import collate_experiments


class ModelAnalyser:
    TARGET_LABELS = {
        "mass_soot_remaining_mg": "Soot mass (mg)",
        "soot_oxidation_co2_concentration_ppm": "CO$_2$ (ppm)",
        "soot_oxidation_co_concentration_ppm": "CO (ppm)",
        "soot_oxidation_co2_selectivity": "CO$_2$ selectivity",
        "nox_ppm": "NOx (ppm)",
        "no2_ppm": "NO$_2$ (ppm)",
        "no_ppm": "NO (ppm)",
        "n2_ppm": "N$_2$ (ppm)",
        "no2_fraction_of_nox": "NO$_2$ / NOx",
    }

    def __init__(self, model, dataset, device=None, batch_size=32, seed=1, outdir=None):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.outdir = Path(outdir) if outdir is not None else None

        if self.outdir is not None:
            self.outdir.mkdir(parents=True, exist_ok=True)

        if device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)
        self.target_names = [name for name in self.TARGET_LABELS if name in dataset.target_cols]

    def move_to_device(self, value):
        if torch.is_tensor(value):
            return value.to(self.device)
        if isinstance(value, dict):
            return {k: self.move_to_device(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.move_to_device(v) for v in value]
        return value

    def predict_batch(self, batch):
        batch = self.move_to_device(batch)
        was_training = self.model.training
        self.model.eval()

        with torch.no_grad():
            predictions = self.model(batch)

        if was_training:
            self.model.train()

        return batch, predictions

    def _temperature_C(self, batch, i, mask):
        return batch["temperature_K"][i, mask].detach().cpu().numpy() - 273.15

    def _plot_mask(self, batch, target_name, i):
        mask = batch["masks"][target_name][i].clone()

        # First soot mass may have been masked because it was used as the ODE IC.
        # Still show it as experimental data on the curve plot.
        if target_name == "mass_soot_remaining_mg" and batch["m_C_intial_obtained_from"][i] == "first_data_point":
            mask[0] = True

        return mask

    def plot_random_experiments(self, n_experiments=10, seed=None):
        seed = self.seed if seed is None else seed
        rng = np.random.default_rng(seed)
        n_experiments = min(n_experiments, len(self.dataset))
        indices = rng.choice(len(self.dataset), size=n_experiments, replace=False).tolist()

        loader = DataLoader(Subset(self.dataset, indices), batch_size=n_experiments, shuffle=False, collate_fn=collate_experiments)
        batch, predictions = self.predict_batch(next(iter(loader)))
        rate_names = self._rate_names(predictions)
        figures = {}

        for i in range(n_experiments):
            experiment_id = str(batch["experiment_id"][i])
            valid = batch["observation_mask"][i]

            target_names = [
                name for name in self.target_names
                if name in predictions and (self._plot_mask(batch, name, i) & valid).any()
            ]

            plot_items = [("target", name) for name in target_names] + [("rate", name) for name in rate_names]

            if not plot_items:
                continue

            n_cols = 3
            n_rows = int(np.ceil(len(plot_items) / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows))
            axes = np.asarray(axes).reshape(-1)

            for ax, (kind, name) in zip(axes, plot_items):
                x = self._temperature_C(batch, i, valid)

                if kind == "target":
                    pred = predictions[name][i, valid].detach().cpu().numpy()
                    ax.plot(x, pred, linewidth=1.5, label="Predicted")

                    if name == "no2_fraction_of_nox" and "S_NO2_eq" in predictions:
                        nox_in = batch["F_NO_in"][i] + batch["F_NO2_in"][i]
                        if nox_in > 0:
                            equilibrium = predictions["S_NO2_eq"][i, valid].detach().cpu().numpy()
                            ax.plot(x, equilibrium, "--", linewidth=1.5, label="Thermodynamic equilibrium")

                    measured_mask = self._plot_mask(batch, name, i) & valid
                    x_real = self._temperature_C(batch, i, measured_mask)
                    y_real = batch["targets"][name][i, measured_mask].detach().cpu().numpy()
                    ax.scatter(x_real, y_real, s=25, zorder=3, label="Measured")

                    ax.set_title(self.TARGET_LABELS[name])
                    ax.set_ylabel(self.TARGET_LABELS[name])
                    ax.legend()

                else:
                    rate = predictions[name][i, valid].detach().cpu().numpy()
                    finite = np.isfinite(rate)

                    if not finite.any():
                        ax.remove()
                        continue

                    ax.plot(x[finite], rate[finite], linewidth=1.5)
                    ax.set_title(name)
                    ax.set_ylabel("Rate / extent (mol min$^{-1}$)")

                ax.set_xlabel("Temperature (°C)")
                ax.grid(alpha=0.2)

            for ax in axes[len(plot_items):]:
                ax.remove()

            fig.suptitle(f"Experiment {experiment_id}", fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.97])

            if self.outdir is not None:
                fig.savefig(self.outdir / f"{experiment_id}.png", dpi=300, bbox_inches="tight")

            figures[experiment_id] = fig

        return figures

    def collect_parity_data(self):
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_experiments)
        parity = {name: {"actual": [], "predicted": []} for name in self.target_names}

        for batch in loader:
            batch, predictions = self.predict_batch(batch)

            for name in self.target_names:
                if name not in predictions:
                    continue

                mask = batch["masks"][name]

                if not mask.any():
                    continue

                actual = batch["targets"][name][mask].detach().cpu().numpy()
                predicted = predictions[name][mask].detach().cpu().numpy()
                parity[name]["actual"].append(actual)
                parity[name]["predicted"].append(predicted)

        for name in self.target_names:
            parity[name]["actual"] = np.concatenate(parity[name]["actual"]) if parity[name]["actual"] else np.array([])
            parity[name]["predicted"] = np.concatenate(parity[name]["predicted"]) if parity[name]["predicted"] else np.array([])

        return parity

    def plot_parity(self):
        parity = self.collect_parity_data()
        active_names = [name for name in self.target_names if len(parity[name]["actual"]) > 0]
        n_targets = len(active_names)
        n_cols = 3
        n_rows = int(np.ceil(n_targets / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.5 * n_rows))
        axes = np.asarray(axes).reshape(-1)

        for ax, name in zip(axes, active_names):
            actual = parity[name]["actual"]
            predicted = parity[name]["predicted"]

            finite = np.isfinite(actual) & np.isfinite(predicted)
            actual, predicted = actual[finite], predicted[finite]

            lower = min(actual.min(), predicted.min())
            upper = max(actual.max(), predicted.max())
            margin = 0.05 * (upper - lower if upper > lower else 1.0)
            lower, upper = lower - margin, upper + margin

            ax.scatter(actual, predicted, alpha=0.65, s=20)
            ax.plot([lower, upper], [lower, upper], "--", linewidth=1)
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
            ax.set_aspect("equal", adjustable="box")

            residual = predicted - actual
            rmse = np.sqrt(np.mean(residual ** 2))
            mae = np.mean(np.abs(residual))
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - actual.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            ax.text(0.04, 0.96, f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.3g}\nMAE = {mae:.3g}\nn = {len(actual)}", transform=ax.transAxes, va="top")
            label = self.TARGET_LABELS[name]
            ax.set_title(label)
            ax.set_xlabel(f"Measured {label}")
            ax.set_ylabel(f"Predicted {label}")
            ax.grid(alpha=0.2)

        for ax in axes[n_targets:]:
            ax.remove()

        fig.suptitle("Model parity — full dataset", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if self.outdir is not None:
            fig.savefig(self.outdir / "parity_plots.png", dpi=300, bbox_inches="tight")

        return fig, parity

    def analyse(self, n_experiments=10, seed=None):
        parameter_fig, parameter_data = self.plot_fitted_parameters()
        figs = self.plot_random_experiments(n_experiments=n_experiments, seed=seed)
        plt.close()
        parity_fig, parity = self.plot_parity()

        # return {
        #     "curves_figure": curves_fig,
        #     "parity_figure": parity_fig,
        #     "experiment_ids": experiment_ids,
        #     "parity_data": parity,
        #     "parameter_data": parameter_data
        # }
    
    def _rate_names(self, predictions):
        names = [name for name, value in predictions.items() if re.fullmatch(r"[rR]\d+", name) and torch.is_tensor(value) and value.ndim >= 2]
        return sorted(names, key=lambda name: (int(name[1:]), name[0].isupper()))
    
    def collect_fitted_parameters(self):
        parameter_info = getattr(self.model, "fitted_parameter_keys", {})

        if isinstance(parameter_info, (list, tuple)):
            parameter_info = {name: name for name in parameter_info}

        values = {name: [] for name in parameter_info}

        if not values:
            return values

        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_experiments)

        for batch in loader:
            batch, predictions = self.predict_batch(batch)
            observation_mask = batch["observation_mask"]

            for name in values:
                if name not in predictions:
                    continue

                value = predictions[name]

                if not torch.is_tensor(value):
                    continue

                if value.ndim == 0:
                    arr = np.array([value.detach().cpu().item()])

                elif value.ndim == 1:
                    arr = value.detach().cpu().numpy()

                else:
                    # Static parameters are repeated over the temperature trajectory.
                    # Take one value per experiment rather than counting every T point.
                    batch_values = []

                    for i in range(value.shape[0]):
                        valid = torch.where(observation_mask[i])[0]

                        if len(valid) == 0:
                            continue

                        batch_values.append(value[i, valid[0]].reshape(-1)[0])

                    if not batch_values:
                        continue

                    arr = torch.stack(batch_values).detach().cpu().numpy()

                arr = arr[np.isfinite(arr)]

                if len(arr):
                    values[name].append(arr)

        for name in values:
            values[name] = np.concatenate(values[name]) if values[name] else np.array([])

        return values
    
    def plot_fitted_parameters(self):
        data = self.collect_fitted_parameters()
        parameter_info = getattr(self.model, "fitted_parameter_keys", {})

        if isinstance(parameter_info, (list, tuple)):
            parameter_info = {name: name for name in parameter_info}

        active_names = [name for name, values in data.items() if len(values) > 0]

        if not active_names:
            return None, data

        n_cols = 3
        n_rows = int(np.ceil(len(active_names) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.2 * n_rows))
        axes = np.asarray(axes).reshape(-1)

        for ax, name in zip(axes, active_names):
            values = data[name]
            bins = min(20, max(5, int(np.sqrt(len(values)))))

            ax.hist(values, bins=bins)
            ax.axvline(np.median(values), linestyle="--", linewidth=1, label="Median")

            ax.set_title(parameter_info.get(name, name))
            ax.set_xlabel(parameter_info.get(name, name))
            ax.set_ylabel("Experiments")
            ax.grid(alpha=0.2)

            ax.text(
                0.97, 0.95,
                f"n = {len(values)}\nmean = {np.mean(values):.3g}\nmedian = {np.median(values):.3g}",
                transform=ax.transAxes,
                ha="right",
                va="top",
            )

            ax.legend()

        for ax in axes[len(active_names):]:
            ax.remove()

        fig.suptitle("Fitted parameter distributions — full dataset", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if self.outdir is not None:
            fig.savefig(self.outdir / "fitted_parameter_histograms.png", dpi=300, bbox_inches="tight")

        return fig, data