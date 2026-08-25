import torch
import torch.nn.functional as F

from .mlp import MLP
from .only_s_nox import OnlySelectivityModel


class ArrheniusOnlySelectivityModel(OnlySelectivityModel):
    def __init__(self, config, nn_input_dim, scaler):
        super().__init__(config, nn_input_dim, scaler)

        del self.nn

        nn_cfg = config["nn"].copy()

        arrhenius_cfg = nn_cfg.copy()
        arrhenius_cfg["output_dim"] = 6
        self.arrhenius_nn = MLP(input_dim=self.static_input_dim, cfg=arrhenius_cfg)

        co2_cfg = nn_cfg.copy()
        co2_cfg["output_dim"] = 1
        self.co2_nn = MLP(input_dim=nn_input_dim, cfg=co2_cfg)

        cfg_arr = config["physics"].get("arrhenius", {})
        self.T_ref = cfg_arr.get("T_ref_K", 700.0)
        self.Ea_scale = cfg_arr.get("Ea_scale_kJ_mol", 50.0) * 1000.0
        self.log_ref_bound = cfg_arr.get("log_ref_multiplier_bound", 2.0)

        self.diagnostic_output_keys = [
            "y1",
            "Da2",
            "y3",
            "Da3",
            "Ea1_kJ_mol",
            "Ea2_kJ_mol",
            "Ea3_kJ_mol",
        ]

    def calculate_model_terms(self, static_inputs_scaled, state_scaled, nn_inputs, m_C, T):
        z_b1, z_Ea1, z_b2, z_Ea2, z_b3, z_Ea3 = self.arrhenius_nn(static_inputs_scaled).unbind(dim=-1)

        b1 = self.log_ref_bound * torch.tanh(z_b1)
        b2 = self.log_ref_bound * torch.tanh(z_b2)
        b3 = self.log_ref_bound * torch.tanh(z_b3)

        Ea1 = self.Ea_scale * F.softplus(z_Ea1)
        Ea2 = self.Ea_scale * F.softplus(z_Ea2)
        Ea3 = self.Ea_scale * F.softplus(z_Ea3)

        R = self.physics_calculator.R

        arr1 = Ea1 / R * (1.0 / self.T_ref - 1.0 / T)
        arr2 = Ea2 / R * (1.0 / self.T_ref - 1.0 / T)
        arr3 = Ea3 / R * (1.0 / self.T_ref - 1.0 / T)

        y1 = self.y1_scale * torch.exp(b1 + arr1)
        Da2 = self.Da2_scale * torch.exp(b2 + arr2)
        y3 = self.y3_scale * torch.exp(b3 + arr3)

        S_CO2 = torch.sigmoid(self.co2_nn(nn_inputs).squeeze(-1))

        return {
            "y1": y1,
            "Da2": Da2,
            "y3": y3,
            "S_CO2": S_CO2,

            "diagnostics": {
                "Ea1_kJ_mol": Ea1 / 1000.0,
                "Ea2_kJ_mol": Ea2 / 1000.0,
                "Ea3_kJ_mol": Ea3 / 1000.0,
            },
        }