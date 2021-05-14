import torch

"""
  Functions that act as replacements for ArgMin or ArgMax across dimension 1
  Intended for use-cases when dealing with embedding selectors that need to be differentiable
"""

def hard_gumbel_softmin(x: torch.Tensor) -> torch.Tensor:
    scaled_x = torch.softmax(-1 * x, dim=1)
    indices_hard = torch.argmin(x, dim=1, keepdim=True)
    indices = torch.zeros_like(scaled_x).scatter_(1, indices_hard, 1.0)
    indices = indices - scaled_x.detach() + scaled_x
    return indices


def hard_gumbel_softmax(x: torch.Tensor) -> torch.Tensor:
    scaled_x = torch.softmax(x, dim=1)
    indices_hard = torch.argmax(x, dim=1, keepdim=True)
    indices = torch.zeros_like(scaled_x).scatter_(1, indices_hard, 1.0)
    indices = indices - scaled_x.detach() + scaled_x
    return indices
