from torch import nn
import torch


class BaseCNNEncoder(nn.Module):
    """
    Encode the input to the latent space.
    Q(z|X)
    """

    def __init__(self, dim_input: int, dim_latent: int, out_channels_convolution: int = 32) -> None:
        super(BaseCNNEncoder, self).__init__()
        self.dim_input = dim_input
        self.dim_latent = dim_latent
        self.out_channels_convolution = out_channels_convolution
        # For the random encoder use this base_net and a linear layer to map the output of the base_net to the mean and std of the variables (which is of size 2*dim_latent).
        # For the deterministic encoder use this base_net and a linear layer to map the output of the base_net to the latent space (which is of size dim_latent).
        self.base_net = nn.Sequential(
            nn.Conv2d(1, self.out_channels_convolution, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.out_channels_convolution,
                      self.out_channels_convolution * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.out_channels_convolution * 2),
            nn.ReLU(True),
            nn.Conv2d(self.out_channels_convolution * 2,
                      self.out_channels_convolution * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.out_channels_convolution * 4),
            nn.ReLU(True),
            nn.Conv2d(self.out_channels_convolution * 4,
                      self.out_channels_convolution * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.out_channels_convolution * 8),
            nn.ReLU(True),
            nn.Flatten(start_dim=1),
        )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError


class DeterministicCNNEncoder(BaseCNNEncoder):
    """
    Here, the conditional distribution of Q(z|X) is deterministic, i.e. Q(z|X) = delta_{G(X)}(z).
    """

    def __init__(self, dim_input: int, dim_latent: int, out_channels_convolution: int = 32) -> None:
        super().__init__(
            dim_input, dim_latent, out_channels_convolution=out_channels_convolution)

        self.net = self.base_net.append(nn.Linear(256, self.dim_latent))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x


class RandomCNNEncoder(BaseCNNEncoder):
    """
    Here, the conditional distribution of Q(z|X) is probabilistic, i.e. Q(z|X) = N(\mu(X), diag(\sigma_1, \ldots,\sigma_D)).
    """

    def __init__(self, dim_input: int, dim_latent: int, out_channels_convolution: int = 32) -> None:
        super().__init__(
            dim_input, dim_latent, out_channels_convolution=out_channels_convolution)

        self.net = self.base_net.append(nn.Linear(256, 2*self.dim_latent))

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: torch.Tensor) -> tuple(torch.Tensor, torch.Tensor, torch.Tensor):
        x = self.net(x)
        mu, log_var = x[:, :self.dim_latent], x[:, self.dim_latent:]
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var


class CNNDecoder(nn.Module):
    """
    G(Z), where Z is the tensor from the latent space.
    """

    def __init__(self, dim_latent: int, dim_output: int, out_channels_convolution: int = 32) -> None:
        super(CNNDecoder, self).__init__()
        self.dim_latent = dim_latent
        self.dim_output = dim_output
        self.dim_h = out_channels_convolution
        self.net = self.get_nn()

    def get_nn(self) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, 1, 4, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.Linear(self.dim_latent, self.dim_h * 8 * 7 * 7)(x)
        x = nn.ReLU()(x)
        x = x.view(-1, self.dim_h * 8, 7, 7)
        x = self.net(x)
        return x


class SimpleDeterministicEncoder(nn.Module):
    """
    Encode the input to the latent space.
    Q(z|X)
    """

    def __init__(self, dim_input: int, dim_latent: int) -> None:
        super().__init__()
        self.dim_input = dim_input
        self.dim_latent = dim_latent
        self.base_net = nn.Sequential(
            nn.Linear(self.dim_input, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.dim_latent),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_net(x)
        return x


class SimpleDecoder(nn.Module):
    """
    G(Z), where Z is the tensor from the latent space.
    """

    def __init__(self, dim_latent: int, dim_output: int) -> None:
        super().__init__()
        self.dim_latent = dim_latent
        self.dim_output = dim_output
        self.base_net = nn.Sequential(
            nn.Linear(self.dim_latent, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.dim_output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_net(x)
        return x
