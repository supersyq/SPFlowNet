import torch

def pairwise_distance(src, dst, normalized=True):
    """Calculates squared Euclidean distance between each two points.
    Args:
        src (torch tensor): source data, [b, n, c]
        dst (torch tensor): target data, [b, m, c]
        normalized (bool): distance computation can be more efficient 
    Returns:
        dist (torch tensor): per-point square distance, [b, n, m]
    """

    if len(src.shape) == 2:
        src = src.unsqueeze(0)
        dst = dst.unsqueeze(0)

    B, N, _ = src.shape
    _, M, _ = dst.shape
    
    # Minus such that smaller value still means closer 
    dist = -torch.matmul(src, dst.permute(0, 2, 1))

    # If inputs are normalized just add 1 otherwise compute the norms 
    if not normalized:
        dist *= 2 
        dist += torch.sum(src ** 2, dim=-1)[:, :, None]
        dist += torch.sum(dst ** 2, dim=-1)[:, None, :]
    
    else:
        dist += 1.0
    
    # Distances can get negative due to numerical precision
    dist = torch.clamp(dist, min=0.0, max=None)
    
    return dist


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z