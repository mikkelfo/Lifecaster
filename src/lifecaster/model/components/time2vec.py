import torch


class Time2Vec(torch.nn.Module):
    """Implements Time2vec (https://arxiv.org/abs/1907.05321)"""

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 768,
        function: callable = torch.cos,
        init_scale: float = 1,
        clip_min: float = None,
        clip_max: float = None,
    ):
        super().__init__()
        self.f = function
        self.init_scale = init_scale
        self.clip_min = clip_min
        self.clip_max = clip_max
        # for i = 0
        self.w0 = torch.nn.Parameter(torch.randn(input_dim, 1))
        self.phi0 = torch.nn.Parameter(torch.randn(1))
        # for 1 <= i <= k (input_size)
        self.w = torch.nn.Parameter(torch.randn(input_dim, output_dim - 1))
        self.phi = torch.nn.Parameter(torch.randn(output_dim - 1))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """Linear layers with bias (through nn.Parameter) followed clipping and a periodic function"""
        if self.init_scale is not None:
            tau = tau * self.init_scale
        tau = tau.unsqueeze(-1)

        linear_1 = torch.matmul(tau, self.w0) + self.phi0
        linear_2 = torch.matmul(tau, self.w)

        if self.clip_min is not None or self.clip_max is not None:
            linear_1 = torch.clamp(linear_1, min=self.clip_min, max=self.clip_max)
            linear_2 = torch.clamp(linear_2, min=self.clip_min, max=self.clip_max)

        periodic = self.f(linear_2 + self.phi)

        return torch.cat((linear_1, periodic), dim=-1)
