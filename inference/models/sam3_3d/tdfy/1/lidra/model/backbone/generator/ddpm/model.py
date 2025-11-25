from typing import Callable
import torch

from lidra.model.backbone.generator.base import Base
from lidra.model.backbone.generator.ddpm.schedule import Schedule
from lidra.model.backbone.generator.ddpm.solver import Solver, Noise
from lidra.data.utils import right_broadcasting


# https://arxiv.org/pdf/2006.11239
class DDPM(Base):
    def __init__(
        self,
        reverse_fn: Callable,
        schedule: Schedule,
        solver: Solver = Noise(),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.reverse_fn = reverse_fn
        self.schedule = schedule
        self.solver = solver

    def _get_signal_noise_weights(self, t):
        t_idx = t - 1
        selected_alpha_bar = self.schedule.alphas_bar[t_idx]
        signal_weights = torch.sqrt(selected_alpha_bar)
        noise_weights = torch.sqrt(1 - selected_alpha_bar)

        return signal_weights, noise_weights

    def _generate_noise(self, x_shape, x_device):
        return torch.randn(x_shape, generator=self.random_generator).to(x_device)

    def _forward_process(self, x0, t):
        noise = self._generate_noise(x0.shape, x0.device)
        signal_weights, noise_weights = self._get_signal_noise_weights(t)
        signal_weights = right_broadcasting(signal_weights, x0)
        noise_weights = right_broadcasting(noise_weights, x0)
        x_t = signal_weights * x0 + noise_weights * noise
        return x_t, noise

    # algorithm 1 (in paper)
    def loss(self, x0: torch.Tensor, *args_conditionals, **kwargs_conditionals):
        batch_size = x0.shape[0]
        t = torch.randint(
            low=1,
            high=self.schedule.max_t + 1,
            size=(batch_size,),
            generator=self.random_generator,
        )
        t = t.to(x0.device)
        t_idx = t - 1

        x_t, noise = self._forward_process(x0, t)

        target = self.solver.target(x0, noise, t, self.schedule)

        prediction = self.reverse_fn(
            x_t,
            self.schedule.time[t_idx],
            *args_conditionals,
            **kwargs_conditionals,
        )

        loss = torch.nn.functional.mse_loss(prediction, target, reduction="mean")

        return loss

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
        signal_weight = 1.0 / torch.sqrt(self.schedule.alphas[t_idx])
        noise_weight = torch.divide(
            self.schedule.alphas[t_idx] - 1.0,
            torch.sqrt(
                self.schedule.alphas[t_idx] * (1.0 - self.schedule.alphas_bar[t_idx])
            ),
        )
        prediction = self.reverse_fn(
            x_t,
            self.schedule.time[t_idx],
            *args_conditionals,
            **kwargs_conditionals,
        )
        noise = self.solver.get_noise(x_t, prediction, t, self.schedule)
        x_tm1 = signal_weight * x_t + noise_weight * noise

        if t_idx > 0:
            z = self._generate_noise(x_tm1.shape, device)
            # TODO(Pierre) : handle other sigma case (interpolation coeff ?)
            sigma = torch.sqrt(self.schedule.betas[t_idx])
            x_tm1 += sigma * z
        else:
            z = None

        aux = (noise, z)

        return x_tm1, aux

    # algorithm 2 (in paper)
    def generate_iter(
        self,
        x_shape,
        x_device,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        x_t = self._generate_noise(x_shape, x_device)
        for t in reversed(range(1, self.schedule.max_t + 1)):
            x_t, aux = self._step(
                t,
                x_t,
                x_device,
                *args_conditionals,
                **kwargs_conditionals,
            )
            yield t, x_t, aux
