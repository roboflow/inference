from typing import Callable
import torch

from lidra.model.backbone.generator.ddpm.model import DDPM
from lidra.model.backbone.generator.ddpm.schedule import Schedule
from lidra.model.backbone.generator.ddpm.solver import Solver, Noise


# https://arxiv.org/pdf/2010.02502
class DDIM(DDPM):
    def __init__(
        self,
        reverse_fn: Callable,
        training_schedule: Schedule,
        generative_schedule: Schedule,
        eta: float = 0.0,  # 1.0 = stochastic, 0.0 = deterministic generation
        solver: Solver = Noise(),
        **kwargs,
    ):
        super().__init__(reverse_fn, training_schedule, solver, **kwargs)

        self.generative_schedule = generative_schedule
        self.eta = eta

    # equation (16)
    def _get_sigma(self, alpha_bar_t, alpha_bar_tm1):
        sigma = self.eta
        sigma *= torch.sqrt((1 - alpha_bar_tm1) * (alpha_bar_tm1 - alpha_bar_t))
        sigma /= torch.sqrt((1 - alpha_bar_t) * alpha_bar_tm1)
        return sigma

    @staticmethod
    def _check_bound_at_0(array, t_idx, value_at_0, device):
        return array[t_idx] if t_idx > -1 else torch.tensor(value_at_0, device=device)

    # equation (12)
    def _step(
        self,
        t,
        x_t,
        device,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        t_idx = t - 1
        t = torch.tensor([t], device=device)

        prediction = self.reverse_fn(
            x_t,
            self.schedule.time[t_idx],
            *args_conditionals,
            **kwargs_conditionals,
        )

        noise = self.solver.get_noise(x_t, prediction, t, self.schedule)
        x0 = self.solver.get_x0(x_t, prediction, t, self.schedule)

        # get constants
        alpha_bar_tm1 = DDIM._check_bound_at_0(
            self.schedule.alphas_bar,
            t_idx - 1,
            1,
            device,
        )
        alpha_bar_t = self.schedule.alphas_bar[t_idx]
        sigma = self._get_sigma(alpha_bar_t, alpha_bar_tm1)

        x0_weight = torch.sqrt(alpha_bar_tm1)
        noise_weight = torch.sqrt(1 - alpha_bar_tm1 - sigma**2)
        x_tm1 = x0_weight * x0 + noise_weight * noise

        if (self.eta > 0) and (t_idx > 0):
            z = self._generate_noise(x_tm1.shape, device)
            x_tm1 += sigma * z
        else:
            z = None

        aux = (noise, z, x0)

        return x_tm1, aux

    def generate_iter(
        self,
        x_shape,
        x_device,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        # replace current schedule by generative (usually shorter) version
        _schedule = self.schedule
        self.schedule = self.generative_schedule
        yield from super().generate_iter(
            x_shape,
            x_device,
            *args_conditionals,
            **kwargs_conditionals,
        )
        self.schedule = _schedule
