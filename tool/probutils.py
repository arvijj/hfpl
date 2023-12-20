import math
import torch


def sample_probs(probs, num_samples, logits=None):
    """
    Aggregates probs by importance sampling. For each image and class, samples
    num_samples number of pixels, and returns the corresponding elements in
    probs. If logits is provided, instead returns the elements in logits.
    """
    assert probs.dim() == 4, f'Dimension of probs must be 4, got {probs.dim()}.'
    n, c, h, w = probs.size()
    probs = probs.view(n*c, h*w)
    indices = torch.multinomial(probs, num_samples=num_samples, replacement=True)
    if logits is not None:
        samples = torch.gather(logits.view(n*c, h*w), 1, indices)
    else:
        samples = torch.gather(probs, 1, indices)
    samples = samples.view(n, c, num_samples)
    return samples


def feature_similarity_loss(x, y, mu, sigma=None, px=1.0, py=2.0, fast=False, max_dist=10):
    """
    Computes the feature similarity loss between features x and predictions y.
    The tensors x and y must have shapes BxC1xHxW and BxC2xHxW respectively.
    """
    assert x.dim() == y.dim() == 4, f'Both x and y must have 4 dimensions, but they had {x.dim()} and {y.dim()}.'
    assert x.size()[0] == y.size()[0], 'x and y must have the same batch dimensions.'
    assert x.size()[2:] == y.size()[2:], 'x and y must have the same spatial dimensions.'

    # Format input dimensions
    n, cx, h, w = x.size()
    x = x.clone().view(n, -1, h*w).transpose(1, 2).contiguous().clamp(0., 1.)
    y = y.clone().view(n, -1, h*w).transpose(1, 2).contiguous()

    # Compute spatial distances
    if fast or sigma is not None:
        h_range = torch.arange(0, h).to(x.device)
        w_range = torch.arange(0, w).to(x.device)
        hh, ww = torch.meshgrid(h_range, w_range)
        hh, ww = (hh.type(torch.float), ww.type(torch.float))
        pos = torch.cat([hh.unsqueeze(-1), ww.unsqueeze(-1)], dim=-1).view(1, -1, 2)
        dp = torch.cdist(pos, pos, p=2.0)

    # Compute feature (dx) and prediction (dy) distances
    dx = torch.cdist(x, x, p=px)
    dy = torch.cdist(y, y, p=py)

    # Ignore distant pixel pairs
    if fast:
        dp = torch.where(dp > max_dist, torch.tensor(0.).to(dp.device), dp)
        indices = torch.nonzero(dp).T
        dx = dx[indices[0], indices[1], indices[2]]
        dy = dy[indices[0], indices[1], indices[2]]
        dp = dp[indices[0], indices[1], indices[2]]

    # Compute the loss
    dx = dx / (cx ** (1/px))
    dx = dx.clamp(0., 1.)
    dx = torch.tanh(mu + torch.log((dx + 1e-5) / (1 - dx + 1e-5)))
    dy = 0.5 * dy ** py
    if sigma is not None:
        dp = torch.exp(-0.5*(dp/sigma)**2) / (2.*math.pi*sigma**2)
    else:
        dp = 1 / (h*w)
    fs_loss = -(1/(n*h*w)) * dp * dy * dx
    return torch.sum(fs_loss)
