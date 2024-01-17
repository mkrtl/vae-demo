from torch.utils import data
import torch
import math


class CircleData(data.Dataset):
    """
    A simple dataset of points on a circle perturbed by Gaussian noise.
    """

    def __init__(self, n_samples: int, dim: int = 2, radius: float = 1., sigma: float = 0.1, seed: int = 0) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.dim = dim
        self.radius = radius
        self.sigma = sigma
        self.seed = seed
        self.data = self.generate_data()

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index) -> torch.Tensor:
        return self.data[:, index]

    def generate_data(self) -> torch.Tensor:
        torch.manual_seed(self.seed)
        random_angles = 2 * math.pi * torch.rand(self.n_samples)
        data = torch.vstack((
            self.radius * torch.cos(random_angles),
                            self.radius *
                            torch.sin(
                                random_angles)
                            )
                            )
        data += self.sigma * torch.randn(self.dim, self.n_samples)
        return data
