import torch
import torch.nn.functional as F

from .mlp import MLP
from .total_nox_base import FullModelNeuralODE


class ArrheniusFullModel(FullModelNeuralODE):
    def __init__(self, config, nn_input_dim, scaler):
        super().__init__(config, nn_input_dim, scaler)

        nn_cfg = config["nn"].copy()

        static_cfg = nn_cfg.copy()
        static_cfg["output_dim"] = 6
        self.arrhenius_nn = MLP(input_dim=self.static_input_dim, cfg=static_cfg)

        state_cfg = nn_cfg.copy()
        state_cfg["output_dim"] = 6
        self.state_nn = MLP(input_dim=nn_input_dim, cfg=state_cfg)

        arr_cfg = config["physics"].get("arrhenius", {})
        self.T_ref = arr_cfg.get("T_ref_K", 700.0)
        self.Ea_scale = arr_cfg.get("Ea_scale_kJ_mol", 50.0) * 1000
        self.log_kref_bound = arr_cfg.get("log_kref_multiplier_bound", 2.0)
        self.state_correction_limit = arr_cfg.get("state_correction_limit", 0.5)

        self.diagnostic_output_keys = ["k1", "k2", "k5", "Ea1_kJ_mol", "Ea2_kJ_mol", "Ea5_kJ_mol"]

    def calculate_model_terms(self, static_inputs_scaled, state_scaled, nn_inputs, m_C, T):
        z_b1, z_Ea1, z_b2, z_Ea2, z_b5, z_Ea5 = self.arrhenius_nn(static_inputs_scaled).unbind(dim=-1)

        b1 = self.log_kref_bound * torch.tanh(z_b1)
        b2 = self.log_kref_bound * torch.tanh(z_b2)
        b5 = self.log_kref_bound * torch.tanh(z_b5)

        Ea1 = self.Ea_scale * F.softplus(z_Ea1)
        Ea2 = self.Ea_scale * F.softplus(z_Ea2)
        Ea5 = self.Ea_scale * F.softplus(z_Ea5)

        z_d1, z_d2, z3, z4, z_d5, z_co2 = self.state_nn(nn_inputs).unbind(dim=-1)

        d1 = self.state_correction_limit * torch.tanh(z_d1)
        d2 = self.state_correction_limit * torch.tanh(z_d2)
        d5 = self.state_correction_limit * torch.tanh(z_d5)

        R = self.physics_calculator.R
        arr1 = Ea1 / R * (1 / self.T_ref - 1 / T)
        arr2 = Ea2 / R * (1 / self.T_ref - 1 / T)
        arr5 = Ea5 / R * (1 / self.T_ref - 1 / T)

        k1 = self.k1_scale * torch.exp(b1 + arr1 + d1)
        k2 = self.k2_scale * torch.exp(b2 + arr2 + d2)
        k5 = self.k5_scale * torch.exp(b5 + arr5 + d5)

        return {
            "k1": k1,
            "k2": k2,
            "eta3": torch.sigmoid(z3),
            "z4": z4,
            "k5": k5,
            "S_CO2": torch.sigmoid(z_co2),

            "diagnostics": {
                "Ea1_kJ_mol": Ea1 / 1000,
                "Ea2_kJ_mol": Ea2 / 1000,
                "Ea5_kJ_mol": Ea5 / 1000,
            },
        }