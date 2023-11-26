import numpy as np
import torch

import docktgrid as dockt


def test_axes_have_correct_shape():
    grid = dockt.Grid3D(vox_size=0.5, box_dims=[2.0, 6.0, 8.0])
    assert grid.axes_dims == (4, 12, 16)  # check length of each axis


def test_axes_tensors_have_correct_values():
    grid = dockt.Grid3D(vox_size=1.0, box_dims=[8.0, 8.0, 8.0])
    x, y, z = grid.axes
    j = torch.tensor([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

    assert torch.equal(x.cpu(), j)
    assert torch.equal(x.cpu(), z.cpu())
    assert torch.equal(x.cpu(), y.cpu())


def test_grid_points_have_correct_values():
    xyz = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]

    points = list()
    for i in xyz:
        for j in xyz:
            for k in xyz:
                points.append([i, j, k])

    points = np.asarray(points, dtype=np.float32)

    grid = dockt.Grid3D(vox_size=0.5, box_dims=[4.0, 4.0, 4.0])

    np.all(grid.points[0].cpu().numpy() == points[:, 0])
    np.all(grid.points[1].cpu().numpy() == points[:, 1])
    np.all(grid.points[2].cpu().numpy() == points[:, 2])
