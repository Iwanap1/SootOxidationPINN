import torch
import torch.nn.functional as F

from .mlp import MLP
from .total_nox_base import SootNOxNeuralODE


class NOxSelectivityModel(SootNOxNeuralODE):
    def __init__(self, config, nn_input_dim, scaler):
        super().__init__(config, nn_input_dim, scaler)

        nn_cfg = config["nn"].copy()
        nn_cfg["output_dim"] = 6

        self.nn = MLP(input_dim=nn_input_dim, cfg=nn_cfg)

    def calculate_model_terms(self, static_inputs_scaled, state_scaled, nn_inputs, m_C, T):
        z1, z2, z3, z4, z5, z_co2 = self.nn(nn_inputs).unbind(dim=-1)

        return {
            "k1": self.k1_scale * F.softplus(z1),
            "k2": self.k2_scale * F.softplus(z2),
            "eta3": torch.sigmoid(z3),
            "z4": z4,
            "k5": self.k5_scale * F.softplus(z5),
            "S_CO2": torch.sigmoid(z_co2),
        }