import torch

from .config import DTYPE

__all__ = ["Grid3D"]


class Grid3D:
    """Generates a 3D grid of points centered at the origin.

    Attributes:
        axes:
            A tuple with the axes (x, y, z).
        points:
            A tuple with the coordinates of the grid points (x, y, z).
        axes_dims:
            A tuple with the size of the grid in each dimension (x, y, z).
    """

    def __init__(self, vox_size: float, box_dims: list[float]):
        """Initialize a 3D grid of points.

        Args:
            vox_size: Voxel size.
            box_dims: Dimensions of the box containing the grid.
        """
        self._vox_size = vox_size
        self._box_dims = torch.tensor(box_dims, dtype=DTYPE)
        self.axes = self._build_grid_axes()
        self.points = self._build_grid()

    @property
    def axes_dims(self):
        """Get the size of the grid in each dimension (x, y, z)."""
        dim1 = self.axes[0].shape[0]
        dim2 = self.axes[1].shape[0]
        dim3 = self.axes[2].shape[0]

        return (dim1, dim2, dim3)

    def _build_grid_axes(self):
        """Return coordinate axes (x, y, z)."""
        nx, ny, nz = self._box_dims / self._vox_size  # n. of discrete points
        xm, ym, zm = self._box_dims / 2  # center ~ at origin

        x = torch.arange(0, nx, dtype=DTYPE) * self._vox_size - xm
        y = torch.arange(0, ny, dtype=DTYPE) * self._vox_size - ym
        z = torch.arange(0, nz, dtype=DTYPE) * self._vox_size - zm

        return x, y, z

    def _build_grid(self):
        """Return x, y, and z coordinates of the grid points.

        Returns:
            A tuple with grid points coords (x, y, z).

        """
        x, y, z = self.axes

        points = (
            torch.repeat_interleave(x, y.shape[0] * z.shape[0]),
            torch.repeat_interleave(y, z.shape[0]).repeat(x.shape[0]),
            z.repeat(x.shape[0] * y.shape[0]),
        )

        return points
