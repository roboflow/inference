from typing import Callable
import torch


def check_bound(data, min_v, max_v, error_message):
    if isinstance(data, torch.Tensor):
        success = torch.logical_and(min_v <= data, (data <= max_v)).all().item()
    else:
        success = min_v <= data <= max_v

    assert success, error_message


def linear(beta_1=1e-4, beta_T=0.02, T=1000):
    def schedule_fn(t):
        check_bound(
            t,
            min_v=1,
            max_v=T,
            error_message=f"the schedule function can only take a time integer between 1 and {T}",
        )
        r = (t - 1) / (T - 1)
        return (1 - r) * beta_1 + r * beta_T

    return schedule_fn


class BaseSchedule(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._max_t = None

    @property
    def max_t(self):
        return self._max_t

    @max_t.setter
    def max_t(self, t: int):
        if t != self._max_t:
            self._max_t = t

            time, alphas_bar, alphas, betas = self.compute_schedule_weights(self.max_t)

            self.register_buffer("time", time, persistent=False)
            self.register_buffer("alphas_bar", alphas_bar, persistent=False)
            self.register_buffer("alphas", alphas, persistent=False)
            self.register_buffer("betas", betas, persistent=False)

    def compute_schedule_weights(self, max_t):
        raise NotImplementedError


class Schedule(BaseSchedule):
    def __init__(self, schedule_fn: Callable, max_t: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.schedule_fn = schedule_fn
        self.max_t = max_t  # init base schedule

    def compute_schedule_weights(self, max_t):
        t = torch.arange(1, max_t + 1)

        betas = self.schedule_fn(t)
        alphas = 1 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        time = t

        return time, alphas_bar, alphas, betas


class LinearSubSchedule(BaseSchedule):
    def __init__(self, schedule: Schedule, max_t: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.schedule = schedule
        self.max_t = max_t  # init base schedule

    def compute_schedule_weights(self, max_t):
        assert max_t <= self.schedule.max_t

        c = self.schedule.max_t / max_t
        t = torch.arange(1, max_t + 1)

        t_idx = torch.floor(c * t) - 1
        t_idx = t_idx.to(torch.int)

        time = self.schedule.time[t_idx]
        betas = self.schedule.betas[t_idx]
        alphas = self.schedule.alphas[t_idx]
        alphas_bar = self.schedule.alphas_bar[t_idx]

        return time, alphas_bar, alphas, betas
