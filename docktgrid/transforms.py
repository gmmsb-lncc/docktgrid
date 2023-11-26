from typing import Protocol

import torch
from scipy.spatial.transform import Rotation

from .config import DTYPE

__all__ = ["RandomRotation", "Transform"]


class Transform(Protocol):
    """Interface for implementing transformations."""

    def __call__(self, *args, **kwargs):
        ...


class RandomRotation:
    def __init__(self, inplace: bool = True, **kwargs) -> None:
        self.inplace = inplace

    def _get_rn_matrix(self) -> torch.Tensor:
        """Get random rotation matrix."""
        rotation = Rotation.random().as_matrix()
        return torch.from_numpy(rotation).to(dtype=DTYPE)

    def __call__(
        self, coords: torch.Tensor, ligand_center: torch.Tensor
    ) -> tuple[torch.Tensor] | None:
        matrix = self._get_rn_matrix()

        rotated_coords = torch.matmul(matrix, coords)
        rotated_center = torch.matmul(matrix, ligand_center)

        if not self.inplace:
            return rotated_coords, rotated_center

        coords.copy_(rotated_coords)
        ligand_center.copy_(rotated_center)
