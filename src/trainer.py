import copy
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from .architectures.total_nox_model import NOxSelectivityModel
from .data_processing import ExperimentDataset, collate_experiments
from .loss import BalanceBasedLoss


class Trainer:
    LOSS_COMPONENTS = [
        "mass_soot_remaining_mg",
        "soot_oxidation_co2_concentration_ppm",
        "soot_oxidation_co_concentration_ppm",
        "soot_oxidation_co2_selectivity",
        "nox_ppm",
        "no2_ppm",
        "no_ppm",
        "n2_ppm",
        "no2_fraction_of_nox",
    ]

    def __init__(
        self,
        cfg: Dict,
        model: NOxSelectivityModel,
        train_dataset: ExperimentDataset,
        eval_dataset: ExperimentDataset,
        device=None,
        output_dir=None,
    ):
        self.cfg = cfg
        self.train_cfg = cfg["train"]

        self.model = model

        self.train_data = train_dataset
        self.eval_data = eval_dataset

        self.device = (
            torch.device(device)
            if device is not None
            else torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )
        self.model = self.model.to(self.device)
        self.loss_fn = BalanceBasedLoss(cfg["loss"]).to(self.device)

        self.train_loader = self.make_loader(self.train_data, shuffle=True)
        self.eval_loader = self.make_loader(self.eval_data, shuffle=False)
        self.optimiser = self.make_optimiser()

        self.epochs = self.train_cfg["epochs"]
        self.patience = self.train_cfg["early_stopping_patience"]
        self.output_dir = Path(output_dir) if output_dir is not None else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            "train_total": [],
            "train_carbon": [],
            "train_nitrogen": [],
            "eval_total": [],
            "eval_carbon": [],
            "eval_nitrogen": [],
        }
        for name in self.LOSS_COMPONENTS:
            self.history[f"train_{name}"] = []
            self.history[f"eval_{name}"] = []

        self.best_eval_loss = float("inf")
        self.best_epoch = None
        self.best_state_dict = None
        self.debug_cfg = self.train_cfg.get("debug", {})
        # self.debug = self.debug_cfg.get("enabled", False)
        self.debug = False
        self.current_epoch = 0
        self.current_batch = 0

    def make_loader(
        self,
        dataset,
        shuffle,
    ):
        return DataLoader(
            dataset,
            batch_size=self.train_cfg["batch_size"],
            shuffle=shuffle,
            collate_fn=collate_experiments,
        )

    def make_optimiser(self):
        optimiser_name = self.train_cfg["optimiser_name"]

        try:
            optimiser_class = getattr(torch.optim, optimiser_name)
        except AttributeError:
            raise ValueError(
                f"Unknown torch optimiser: "
                f"{optimiser_name}"
            )
        return optimiser_class(self.model.parameters(), **self.train_cfg["optimiser_kwargs"],)

    def move_to_device(self, value):
        if torch.is_tensor(value):
            return value.to(self.device)

        if isinstance(value, dict):
            return {
                key: self.move_to_device(sub_value)
                for key, sub_value in value.items()
            }

        if isinstance(value, list):
            return [
                self.move_to_device(item)
                for item in value
            ]

        return value
    
    def calculate_batch_loss(self, batch):
        batch = self.move_to_device(batch)
        try:
            predictions = self.model(batch)

        except Exception as e:
            if self.debug:
                print(
                    f"\nBatched ODE/model failure | "
                    f"epoch={self.current_epoch} "
                    f"batch={self.current_batch}"
                )

                print("Experiments in failed batch:")
                for experiment_id in batch["experiment_id"]:
                    print(f"  {experiment_id}")

                print(e)

            raise

        batch_size = len(batch["experiment_id"])

        experiment_losses = []


        for i in range(batch_size):

            predictions_i = {}

            for name, value in predictions.items():

                if (
                    torch.is_tensor(value)
                    and value.ndim > 0
                    and value.shape[0] == batch_size
                ):
                    predictions_i[name] = value[i]

                else:
                    # global scalar such as oxidisable_mass_frac
                    predictions_i[name] = value

            targets_i = {name: value[i] for name, value in batch["targets"].items()}
            masks_i = {name: value[i] for name, value in batch["masks"].items()}

            loss_dict = self.loss_fn(
                predictions=predictions_i,
                targets=targets_i,
                masks=masks_i,
            )

            if loss_dict["total"] is None:
                continue

            experiment_losses.append({
                "experiment_id": batch["experiment_id"][i],
                "loss": loss_dict,
            })

        if not experiment_losses: 
            return None
        total = torch.stack([item["loss"]["total"] for item in experiment_losses]).mean()


        carbon_losses = [item["loss"]["carbon"] for item in experiment_losses if item["loss"]["carbon"] is not None]
        nitrogen_losses = [item["loss"]["nitrogen"] for item in experiment_losses if item["loss"]["nitrogen"] is not None]

        carbon = torch.stack(carbon_losses).mean() if carbon_losses else None
        nitrogen = torch.stack(nitrogen_losses).mean() if nitrogen_losses else None
        
        components = {}

        for name in self.LOSS_COMPONENTS:
            values = []

            for item in experiment_losses:
                loss = item["loss"]

                if name in loss["carbon_components"]:
                    values.append(loss["carbon_components"][name])

                elif name in loss["nitrogen_components"]:
                    values.append(loss["nitrogen_components"][name])

            components[name] = (
                torch.stack(values).mean()
                if values
                else None
            )

        return {
            "total": total,
            "carbon": carbon,
            "nitrogen": nitrogen,
            "experiment_losses": experiment_losses,
            **components,
        }

    def train_epoch(self):
        self.model.train()

        losses = {
            name: []
            for name in ["total", "carbon", "nitrogen"] + self.LOSS_COMPONENTS
        }

        for batch_idx, batch in enumerate(self.train_loader):
            self.current_batch = batch_idx + 1

            self.optimiser.zero_grad()
            loss_dict = self.calculate_batch_loss(batch)
            if loss_dict is None:
                print(f"Batch {batch_idx} in {self.current_epoch} had no valid losses")
                continue
            loss_dict["total"].backward()

            grad_norm = None
            gradient_rows = None

            if self.debug:
                grad_norm, gradient_rows = self.get_gradient_summary()

                clip_norm = self.debug_cfg.get("gradient_clip_norm")

                if clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        clip_norm,
                    )

                self.print_batch_debug(
                    loss_dict,
                    grad_norm=grad_norm,
                    gradient_rows=gradient_rows,
                )

            self.optimiser.step()

            for name in ["total", "carbon", "nitrogen"] + self.LOSS_COMPONENTS:
                value = loss_dict[name]

                if value is not None:
                    losses[name].append(value.detach())

        return {
            name: self.mean_losses(values)
            for name, values in losses.items()
        }

    @torch.no_grad()
    def eval_epoch(self):
        self.model.eval()

        loss_names = ["total", "carbon", "nitrogen"] + self.LOSS_COMPONENTS
        losses = {name: [] for name in loss_names}

        for batch in self.eval_loader:
            loss_dict = self.calculate_batch_loss(batch)
            if loss_dict is None: 
                continue

            for name in loss_names:
                value = loss_dict[name]
                if value is not None:
                    losses[name].append(value)

        return {
            name: self.mean_losses(values)
            for name, values in losses.items()
        }

    def mean_losses(self, losses):
        if not losses:
            return None
        return torch.stack(losses).mean().item()

    def update_history(
        self,
        train_loss,
        eval_loss,
    ):
        for name in ["total", "carbon", "nitrogen"] + self.LOSS_COMPONENTS:
            self.history[f"train_{name}"].append(train_loss[name])
            self.history[f"eval_{name}"].append(eval_loss[name])

    def check_early_stopping(
        self,
        epoch,
        eval_loss,
    ):
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.best_epoch = epoch
            self.best_state_dict = copy.deepcopy(self.model.state_dict())

            if self.output_dir is not None:
                torch.save(self.best_state_dict, self.output_dir / "best_model.pt")
            return False

        epochs_without_improvement = epoch - self.best_epoch
        return epochs_without_improvement >= self.patience

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch
            train_loss = self.train_epoch()
            eval_loss = self.eval_epoch()

            self.update_history(train_loss, eval_loss)

            print(
                f"Epoch {epoch:4d} | "
                f"Train {train_loss['total']:.5f} | "
                f"Eval {eval_loss['total']:.5f} | "
                f"C {eval_loss['carbon']} | "
                f"N {eval_loss['nitrogen']} | "
                # f"f_C "
                # f"{self.model.get_carbon_fraction().item():.4f}"
            )

            stop = self.check_early_stopping(epoch, eval_loss["total"])

            if stop:
                print(
                    f"Early stopping at epoch "
                    f"{epoch}. "
                    f"Best epoch: {self.best_epoch}"
                )
                break

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        return self.history

    # def get_gradient_summary(self):
    #     rows = []
    #     total_norm_sq = 0.0

    #     for name, param in self.model.named_parameters():
    #         if param.grad is None:
    #             continue

    #         grad = param.grad.detach()

    #         grad_norm = grad.norm().item()
    #         grad_max = grad.abs().max().item()
    #         param_norm = param.detach().norm().item()

    #         total_norm_sq += grad_norm ** 2

    #         rows.append({
    #             "name": name,
    #             "grad_norm": grad_norm,
    #             "grad_max": grad_max,
    #             "param_norm": param_norm,
    #         })

    #     return total_norm_sq ** 0.5, rows
    
    # def print_batch_debug(self, loss_dict, grad_norm=None, gradient_rows=None):
    #     print(
    #         f"\nDEBUG | epoch={self.current_epoch} "
    #         f"batch={self.current_batch} "
    #         f"batch_loss={loss_dict['total'].item():.6g}"
    #     )

    #     experiment_losses = sorted(
    #         loss_dict["experiment_losses"],
    #         key=lambda x: x["loss"]["total"].item(),
    #         reverse=True,
    #     )

    #     n = self.debug_cfg.get("print_top_experiments", 5)

    #     print("Top experiment losses:")

    #     for item in experiment_losses[:n]:
    #         loss = item["loss"]

    #         print(
    #             f"  {item['experiment_id']} "
    #             f"total={loss['total'].item():.6g} "
    #             f"C={loss['carbon'].item() if loss['carbon'] is not None else None} "
    #             f"N={loss['nitrogen'].item() if loss['nitrogen'] is not None else None}"
    #         )

    #     if grad_norm is not None:
    #         print(f"Total gradient norm: {grad_norm:.6g}")

    #     if gradient_rows is not None:
    #         print("Largest parameter gradients:")

    #         for row in sorted(
    #             gradient_rows,
    #             key=lambda x: x["grad_norm"],
    #             reverse=True,
    #         )[:10]:
    #             print(
    #                 f"  {row['name']:35s} "
    #                 f"grad_norm={row['grad_norm']:.6g} "
    #                 f"grad_max={row['grad_max']:.6g} "
    #                 f"param_norm={row['param_norm']:.6g}"
    #             )


    # def debug_experiment_state(self, experiment):
    #     experiment = self.move_experiment_to_device(experiment)

    #     state = torch.stack([
    #         torch.tensor(0.0, device=self.device),
    #         experiment["start_temp_K"],
    #     ])

    #     with torch.no_grad():
    #         rates = self.model.calculate_rates(
    #             static_inputs_scaled=experiment["static_inputs_scaled"],
    #             m_C_unscaled_state=experiment["m_C_initial"],
    #             T_unscaled_state=experiment["start_temp_K"],
    #             F_total=experiment["F_total"],
    #             F_NO_in=experiment["F_NO_in"],
    #             F_NO2_in=experiment["F_NO2_in"],
    #             o2_fraction=experiment["o2_fraction"],
    #         )

    #         rhs = self.model.ode_rhs(
    #             t=torch.tensor(0.0, device=self.device),
    #             state=state,
    #             static_inputs_scaled=experiment["static_inputs_scaled"],
    #             ramp_rate=experiment["ramp_rate_K_min"],
    #             m_C_initial=experiment["m_C_initial"],
    #             F_total=experiment["F_total"],
    #             F_NO_in=experiment["F_NO_in"],
    #             F_NO2_in=experiment["F_NO2_in"],
    #             o2_fraction=experiment["o2_fraction"],
    #         )

    #     print("Initial-state diagnostics:")
    #     print("  m_C:", experiment["m_C_initial"].item())
    #     print("  T:", experiment["start_temp_K"].item())
    #     print("  du/dt:", rhs[0].item())
    #     print("  r1:", rates["r1"].item())
    #     print("  r2:", rates["r2"].item())
    #     print("  r5:", rates["r5"].item())
    #     print("  dm_C_dt:", rates["dm_C_dt"].item())
    #     print("  F_NO2_potential:", rates["F_NO2_potential"].item())


    # def debug_experiment_trajectory(self, experiment):
    #     T = experiment["temperature_K"]
    #     target_mass = experiment["targets"]["mass_soot_remaining_mg"]
    #     mass_mask = experiment["masks"]["mass_soot_remaining_mg"]

    #     print("Trajectory diagnostics:")

    #     for i in range(len(T)):
    #         if mass_mask[i]:
    #             m_C = target_mass[i]
    #         else:
    #             m_C = experiment["m_C_initial"]

    #         with torch.no_grad():
    #             rates = self.model.calculate_rates(
    #                 static_inputs_scaled=experiment["static_inputs_scaled"],
    #                 m_C_unscaled_state=m_C,
    #                 T_unscaled_state=T[i],
    #                 F_total=experiment["F_total"],
    #                 F_NO_in=experiment["F_NO_in"],
    #                 F_NO2_in=experiment["F_NO2_in"],
    #                 o2_fraction=experiment["o2_fraction"],
    #             )

    #         du_dt = -rates["dm_C_dt"].item() / max(m_C.item(), 1e-12)

    #         print(
    #             f"  T={T[i].item():.1f} K "
    #             f"m={m_C.item():.4g} "
    #             f"du/dt={du_dt:.4g} "
    #             f"r1={rates['r1'].item():.4g} "
    #             f"r2={rates['r2'].item():.4g} "
    #             f"r5={rates['r5'].item():.4g}"
    #         )