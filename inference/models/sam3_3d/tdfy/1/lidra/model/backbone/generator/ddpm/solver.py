import torch

from lidra.data.utils import right_broadcasting


class Solver:
    def target(self, x0, noise, t, schedule):
        raise NotImplementedError

    def get_noise(self, xt, prediction, t, schedule):
        raise NotImplementedError

    def get_x0(self, xt, prediction, t, schedule):
        raise NotImplementedError


class Noise(Solver):
    def target(self, x0, noise, t, schedule):
        return noise

    def get_noise(self, xt, prediction, t, schedule):
        noise = prediction
        return noise

    def get_x0(self, xt, prediction, t, schedule):
        noise = prediction
        t_idx = t - 1
        alpha_bar_t = schedule.alphas_bar[t_idx]
        x0 = xt - torch.sqrt(1 - alpha_bar_t) * noise
        x0 /= torch.sqrt(alpha_bar_t)
        return x0


# TODO(Pierre)
class Sample(Solver):
    def target(self, x0, noise, t, schedule):
        return x0


class Velocity(Solver):
    def target(self, x0, noise, t, schedule):
        t_idx = t - 1
        noise_weight = torch.sqrt(schedule.alphas_bar[t_idx])
        x0_weight = torch.sqrt(1 - schedule.alphas_bar[t_idx])

        noise_weight = right_broadcasting(noise_weight, noise)
        x0_weight = right_broadcasting(x0_weight, x0)

        v = noise_weight * noise - x0_weight * x0
        return v

    def get_noise(self, xt, prediction, t, schedule):
        t_idx = t - 1
        velocity = prediction
        velocity_weight = torch.sqrt(schedule.alphas_bar[t_idx])
        sample_weight = torch.sqrt(1 - schedule.alphas_bar[t_idx])
        noise = velocity_weight * velocity + sample_weight * xt
        return noise

    def get_x0(self, xt, prediction, t, schedule):
        t_idx = t - 1
        velocity = prediction
        velocity_weight = torch.sqrt(1 - schedule.alphas_bar[t_idx])
        sample_weight = torch.sqrt(schedule.alphas_bar[t_idx])
        x0 = sample_weight * xt - velocity_weight * velocity
        return x0
