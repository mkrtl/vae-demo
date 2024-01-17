import torch


def rbf_kernel_dist(X: torch.Tensor, Z: torch.Tensor, bandwidths: list[float] = [.1, .2, .5, 1., 2., 5., 10.]) -> torch.Tensor:
    """
    Calculate MMD distance for several bandwidths and take their sum.
    X and Z must have the same number of columns. 
    """
    assert X.shape[1] == Z.shape[1]
    distances_matrix_x = torch.cdist(X, X, p=2).pow(2)
    distances_matrix_z = torch.cdist(Z, Z, p=2).pow(2)
    distances_matrix_xz = torch.cdist(X, Z, p=2).pow(2)

    dist = torch.zeros(len(bandwidths))
    for k, bandwidth in enumerate(bandwidths):
        dist[k] = (1/(X.shape[0] * Z.shape[0])) * (torch.exp(-0.5 * distances_matrix_x / bandwidth).sum()
                                                   + torch.exp(-0.5 * distances_matrix_z / bandwidth).sum()
                                                   - 2 * torch.exp(-0.5 * distances_matrix_xz / bandwidth).sum())

    return dist.sum()
