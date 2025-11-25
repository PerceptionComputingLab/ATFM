import torch
from einops import rearrange


class ConditionalFlow:
    """Conditional Flow base class."""

    def sample_p_t(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Samples points x_t conditioned on x_1.

        Check eq. (10) in paper: https://arxiv.org/abs/2210.02747

        Args:
            x_0: Samples from base distribution, [batch_size, 1, h, w].
            x_1: Samples from target distribution, [batch_size, 1, h, w].
            t: Time samples, [batch_size].

        Returns:
            Samples x_t from distribution p_t(x), [batch_size, 1, h, w].
        """
        raise NotImplementedError

    def get_conditional_vector_field(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Counts conditional vector field based on samples from base and target distributions.

        Check eq. (15) in paper: https://arxiv.org/abs/2210.02747

        Args:
            x_0: Samples from base distribution, [batch_size, 1, h, w].
            x_1: Samples from target distribution, [batch_size, 1, h, w].
            t: Time samples, [batch_size].

        Returns:
            Target conditional vector field, [batch_size, 1, h, w].
        """
        raise NotImplementedError


class OTConditionalFlow(ConditionalFlow):
    def __init__(self, sigma_min) -> None:
        """Inits optimal transport conditional flow object.

        Args:
            sigma_min: Minimal Gaussian distribution std for target distribution points.
        """
        self.sigma_min = sigma_min

    def sample_p_t(self, x_0, x_1, t):
        """Samples points x_t conditioned on x_1 using optimal transport path.

        Check eq. (20) in paper: https://arxiv.org/abs/2210.02747

        Args:
            x_0: Samples from base distribution, [batch_size, 1, h, w].
            x_1: Samples from target distribution, [batch_size, 1, h, w].
            t: Time samples, [batch_size].

        Returns:
            Samples x_t from distribution p_t(x).
        """
        t = rearrange(t, "b -> b 1 1 1")
        mu_t = t * x_1
        sigma_t = 1 - (1 - self.sigma_min) * t

        x_t = mu_t + sigma_t * x_0

        return x_t

    def get_conditional_vector_field(self, x_0, x_1, t):
        """Counts optimal transport conditional vector field.

        Check eq. (23) in paper: https://arxiv.org/abs/2210.02747

        Args:
            x_0: Samples from base distribution, [batch_size, 1, h, w].
            x_1: Samples from target distribution, [batch_size, 1, h, w].
            t: Time samples, [batch_size].

        Returns:
            Target optimal transport conditional vector field, [batch_size, 1, h, w].
        """
        return x_1 - (1 - self.sigma_min) * x_0
