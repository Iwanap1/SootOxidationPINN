import torch
import torch.nn as nn
from typing import Dict


class BalanceBasedLoss(nn.Module):
    Info = """
    Ltotal = 0.5 * Lc + 0.5 * Ln

    Lc = mean available (
        L_mass_soot,
        L_CO2,
        L_CO,
        L_SCO2
    )

    Ln = mean available (
        L_NO2,
        L_NO,
        L_NOx,
        L_N2,
        L_SNO2
    )

    For each target i:

        L_i = mean [
            ((y_pred_i - y_true_i) / scale_i)^2
        ]

    where the mean is calculated only over measured/known points
    according to the target mask.

    If a target is completely unavailable, it is excluded from the
    corresponding balance rather than assigned zero loss.

    If both carbon and nitrogen losses are available:
        Ltotal = 0.5 * Lc + 0.5 * Ln

    If only one balance is available:
        Ltotal = Lc or Ln
    """

    CARBON_TARGETS = [
        "mass_soot_remaining_mg",
        "soot_oxidation_co2_concentration_ppm",
        "soot_oxidation_co_concentration_ppm",
        "soot_oxidation_co2_selectivity",
    ]

    NITROGEN_TARGETS = [
        "no2_ppm",
        "no_ppm",
        "nox_ppm",
        "n2_ppm",
        "no2_fraction_of_nox",
    ]

    def __init__(self, loss_config: Dict):
        super().__init__()

        self.target_scales = loss_config.get("target_scales",{})
        self.target_weights = loss_config.get("target_weights", {})
        self.carbon_weight = loss_config.get("carbon_weight", 0.5)
        self.nitrogen_weight = loss_config.get("nitrogen_weight", 0.5)


    def target_loss(self, prediction, target, mask, target_name):
        weight = self.target_weights.get(target_name, 1.0)

        if weight <= 0 or not mask.any():
            return None

        prediction = prediction[mask]
        target = target[mask]
        scale = self.target_scales.get(target_name, 1.0)
        residual = (prediction - target) / scale

        return weight * torch.mean(residual ** 2)


    def balance_loss(
        self,
        predictions,
        targets,
        masks,
        target_names,
    ):
        losses = {}

        for name in target_names:
            if name not in predictions:
                continue

            if name not in targets:
                continue

            if name not in masks:
                continue

            loss = self.target_loss(
                prediction=predictions[name],
                target=targets[name],
                mask=masks[name],
                target_name=name,
            )

            if loss is not None:
                losses[name] = loss

        if not losses:
            return None, losses

        balance_loss = torch.stack(list(losses.values())).mean()
        return balance_loss, losses

    def forward(
        self,
        predictions,
        targets,
        masks,
    ):
        carbon_loss, carbon_components = (
            self.balance_loss(
                predictions=predictions,
                targets=targets,
                masks=masks,
                target_names=self.CARBON_TARGETS,
            )
        )

        nitrogen_loss, nitrogen_components = (
            self.balance_loss(
                predictions=predictions,
                targets=targets,
                masks=masks,
                target_names=self.NITROGEN_TARGETS,
            )
        )

        if carbon_loss is not None and nitrogen_loss is not None:
            total_loss = self.carbon_weight * carbon_loss + self.nitrogen_weight * nitrogen_loss

        elif carbon_loss is not None:
            total_loss = carbon_loss

        elif nitrogen_loss is not None:
            total_loss = nitrogen_loss

        else:
            total_loss = None

        return {
            "total": total_loss,
            "carbon": carbon_loss,
            "nitrogen": nitrogen_loss,
            "carbon_components": carbon_components,
            "nitrogen_components": nitrogen_components,
        }