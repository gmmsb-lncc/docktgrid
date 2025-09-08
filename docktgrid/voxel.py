from typing import List

import torch

from docktgrid.config import DEVICE, DTYPE
from docktgrid.grid import Grid3D
from docktgrid.view import View

torch.set_float32_matmul_precision("high")

__all__ = ["VoxelGrid"]


def _calc_vdw_occupancies_fn(
    P3: torch.Tensor,  # (P, 3) grid points (shifted by center), DTYPE on DEVICE
    A3: torch.Tensor,  # (A, 3) atom coords, DTYPE on DEVICE
    vdws12: torch.Tensor,  # (A,) vdw radii^12, DTYPE on DEVICE
    chs: torch.Tensor,  # (C, A) bool on DEVICE
) -> torch.Tensor:  # returns (C, P)
    # compute squared distances via norm identities and a single GEMM
    p2 = (P3 * P3).sum(dim=1, keepdim=True)  # (P,1)
    a2 = (A3 * A3).sum(dim=1, keepdim=False).unsqueeze(0)  # (1,A)
    dot = P3 @ A3.transpose(0, 1)  # (P,A)
    dist2 = (p2 + a2 - 2.0 * dot).clamp_min(1e-12)  # (P,A)

    inv_d2 = dist2.reciprocal()
    inv_d4 = inv_d2 * inv_d2
    inv_d8 = inv_d4 * inv_d4
    inv_d12 = inv_d8 * inv_d4  # (P,A)

    occs = 1 - torch.exp(-(vdws12 * inv_d12))  # (P,A)

    masked = occs.unsqueeze(0).masked_fill(~chs[:, None, :], float("-inf"))  # (C,P,A)
    out = masked.amax(dim=-1)  # (C,P)
    out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    return out


_compiled_calc = torch.compile(
    _calc_vdw_occupancies_fn, dynamic=True, mode="max-autotune"
)


class VoxelGrid:
    """Class to generate voxel representations of protein-ligand complexes.

    Attributes:
        grid:
            A Grid3D object.
        views:
            List of Views.
        num_channels:
            Total number of channels for the chosen view configuration.
        shape:
            Voxel grid shape with channels first (n_channels, dim1, dim2, dim3).
        occupancy_func:
            Occupancy function to use.
    """

    def __init__(
        self,
        views: List[View],
        vox_size: float,
        box_dims: List[float],
        occupancy: str = "vdw",
    ):
        """Initialize voxel grid.

        Args:
            views: List of views.
            vox_size: Voxel size.
            box_dims: Dimensions of the box containing the grid.
            occupancy: Occupancy function to use.

        """
        if occupancy != "vdw":
            raise NotImplementedError(
                f"Occupancy function for {occupancy} is not implemented yet."
            )
        self.grid = Grid3D(vox_size, box_dims)
        self.views = views

        # cache grid points on device for fast matmuls
        px, py, pz = self.grid.points
        self._P3_base = torch.stack((px, py, pz), dim=1).to(device=DEVICE, dtype=DTYPE)

    @property
    def num_channels(self):
        """Get total number of channels for the chosen view configuration."""
        return sum([v.get_num_channels() for v in self.views])

    @property
    def shape(self):
        """Get voxel grid shape with channels first.

        Voxel grid has shape (n_channels, dim1, dim2, dim3).

        """
        n_channels = self.num_channels
        dim1, dim2, dim3 = self.grid.axes_dims
        return (n_channels, dim1, dim2, dim3)

    def get_channels_mask(self, molecule):
        """Build channels mask.

        Each channel is a boolean mask that indicates which atoms are present in the
        channel.

        Args:
            molecule (docktgrid.molecule.MolecularComplex)

        Returns:
            A torch.Tensor with shape (n_channels, n_atoms) type bool

        """
        return torch.cat([v(molecule) for v in self.views])

    def voxelize(self, molecule, out=None, channels=None, requires_grad=False):
        """Voxelize protein-ligand complex and return voxel grid (features).

        Args:
            molecule: docktgrid.molecule.MolecularComplex.

            out (array-like or None): Alternate output array in which to place the result.
            The default is None; if provided, it must have shape corresponding to
            (n_channels, nvoxels).

            channels (array-like or None): Must have shape (n_channels, n_atoms); if
            provided overrides channels
            created from `view`.

        Returns:
            A torch tensor of shape (n_channels, dim1, dim2, dim3). Each element
            corresponds to voxel values, calculated according to the occupancy model.

        """
        if out is None:
            out = torch.zeros(
                self.shape, dtype=DTYPE, device=DEVICE, requires_grad=requires_grad
            )
        else:
            if out.shape != self.shape:
                raise ValueError(
                    " ".join(
                        (
                            f"`out` shape must be == {self.shape},",
                            f"currently it is {out.shape}",
                        )
                    )
                )
            out = torch.as_tensor(
                out, dtype=DTYPE, device=DEVICE, requires_grad=requires_grad
            )

        if channels is None:
            channels = self.get_channels_mask(molecule)
        else:
            cshape = (self.num_channels, molecule.n_atoms)
            if channels.shape != cshape:
                raise ValueError(
                    " ".join(
                        (
                            f"`channels` shape must be == {cshape},",
                            f"currently it is {channels.shape}",
                        )
                    )
                )
        channels = torch.as_tensor(
            channels, dtype=torch.bool, device=DEVICE
        ).contiguous()

        self._voxelize_vdw(molecule, out, channels)
        return out.view(self.shape)

    @torch.no_grad()
    def _voxelize_vdw(self, molecule, out, channels) -> None:
        # shift cached grid points by ligand center
        center = molecule.ligand_center.to(device=DEVICE, dtype=DTYPE)  # (3,)
        P3 = self._P3_base + center  # (P,3)

        # build atoms matrix (A,3) and vdW^12 vector
        ax, ay, az = molecule.coords
        A3 = torch.stack((ax, ay, az), dim=1).to(device=DEVICE, dtype=DTYPE)  # (A,3)
        vdws12 = molecule.vdw_radii.to(device=DEVICE, dtype=DTYPE).pow(12)  # (A,)

        # print("THIS IS THE OPTIMIZED VERSION" * 10)
        out_ = _compiled_calc(P3, A3, vdws12, channels)

        # copy back into provided buffer with shape (C, P)
        C = channels.shape[0]
        P = P3.shape[0]
        out.view(C, P).copy_(out_)
