import torch
import torch.nn as nn
from torchdiffeq import odeint


class NODE(nn.Module):
    def __init__(self, config, nn_input_dim, scaler):
        super().__init__()

        self.config = config
        self.nn_input_dim = nn_input_dim
        self.static_input_dim = nn_input_dim - len(config["nn"]["state"])
        self.solver_options = config["physics"]["solver"]
        self.eps = 1e-12

        state_mean, state_std = scaler.get_mean_std(config["nn"]["state"])
        self.fitted_parameter_keys = {}
        self.register_buffer("state_mean", torch.tensor(state_mean, dtype=torch.float32))
        self.register_buffer("state_std", torch.tensor(state_std, dtype=torch.float32))

    def as_tensor(self, x, dtype, device):
        if torch.is_tensor(x):
            return x.to(dtype=dtype, device=device)
        return torch.tensor(x, dtype=dtype, device=device)

    def scale_state(self, *state_values):
        state = torch.stack(state_values, dim=-1)
        return (state - self.state_mean) / self.state_std

    def initial_state(self, batch):
        raise NotImplementedError

    def ode_rhs(self, t, state, batch):
        raise NotImplementedError

    def decode_solution(self, solution, batch):
        raise NotImplementedError

    def calculate_outputs_from_trajectory(self, batch, trajectory):
        raise NotImplementedError

    def integrate(self, batch):
        y0 = self.initial_state(batch)
        time = batch["shared_time"].to(device=y0.device, dtype=y0.dtype)
        solution = odeint(lambda t, state: self.ode_rhs(t, state, batch), y0, time, method=self.solver_options["method"], options=self.solver_options["options"])
        return self.decode_solution(solution, batch)

    def forward(self, batch):
        trajectory = self.integrate(batch)
        return self.calculate_outputs_from_trajectory(batch, trajectory)

    def count_parameters(self, trainable_only=True):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())