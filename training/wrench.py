"""CHORD-inspired geometric capability, deliberately separate from measured force."""

import math

import torch
from torch.nn import functional as F


def basis_directions(count=64, seed=0):
    generator = torch.Generator().manual_seed(seed)
    base = F.normalize(
        torch.randn((max((count + 1) // 2, 1), 6), generator=generator), dim=-1
    )
    return torch.cat((base, -base), dim=0)[:count]


def wrench_support(points, directions, active, basis, radius, friction=0.8, edges=8):
    """Inputs [...,contacts,3], object-local; output [...,basis]. Unit force rays.

    This directional descriptor is not a torque-limited feasible net-wrench set.
    Live directions follow the pinned runtime's measured-force-direction convention.
    """
    if points.shape[-2] == 0:
        return points.new_zeros((*points.shape[:-2], len(basis)))
    normals = F.normalize(directions, dim=-1, eps=1e-8)
    valid = active.bool() & (directions.norm(dim=-1) > 1e-6)
    helper = F.one_hot(normals.abs().argmin(dim=-1), 3).to(points)
    tangent = F.normalize(torch.linalg.cross(normals, helper), dim=-1)
    bitangent = torch.linalg.cross(normals, tangent)
    angles = torch.arange(edges, device=points.device, dtype=points.dtype) * (
        2 * math.pi / edges
    )
    forces = F.normalize(
        normals[..., None, :]
        + friction
        * (
            angles.cos()[..., None] * tangent[..., None, :]
            + angles.sin()[..., None] * bitangent[..., None, :]
        ),
        dim=-1,
    )
    torques = (
        torch.linalg.cross(points[..., None, :].expand_as(forces), forces) / radius
    )
    wrenches = torch.cat((forces, torques), dim=-1)
    support = torch.einsum("...cew,bw->...ceb", wrenches, basis)
    support = support.masked_fill(~valid[..., None, None], -1e6)
    result = support.flatten(-3, -2).amax(dim=-2)
    return torch.where(
        valid.any(dim=-1, keepdim=True), result, torch.zeros_like(result)
    )


def capability_reward(actual, target, confidence, tolerance=0.2):
    low = target - tolerance * target.abs()
    high = target + tolerance * target.abs()
    error = (F.relu(low - actual).square() + F.relu(actual - high).square()).mean(
        dim=-1
    )
    return confidence * torch.exp(-error / 0.25)
