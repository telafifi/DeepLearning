from functools import cached_property

import numpy as np


def homogeneous(points: np.ndarray) -> np.ndarray:
    """
    Args:
        points (np.ndarray): points with shape (n, d)

    Returns:
        np.ndarray: homogeneous (n, d+1)
    """
    return np.concatenate([points, np.ones((len(points), 1))], axis=1)


def interpolate_smooth(
    points: np.ndarray,
    fixed_distance: float | None = None,
    fixed_number: int | None = None,
):
    if fixed_distance is None and fixed_number is None:
        raise ValueError("Either fixed_distance or fixed_number must be provided")

    dists = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    cumulative = np.concatenate(([0], np.cumsum(dists)))

    if fixed_distance is not None:
        sample = np.arange(0, cumulative[-1], fixed_distance)
    elif fixed_number is not None:
        sample = np.linspace(0, cumulative[-1], fixed_number)

    return np.array([np.interp(sample, cumulative, points[:, i]) for i in range(points.shape[1])]).T


class Track:
    def __init__(
        self,
        path_distance: np.ndarray,
        path_nodes: np.ndarray,
        path_width: np.ndarray,
        interpolate: bool = True,
        fixed_distance: float = 0.5,
    ):
        self.path_distance = path_distance
        self.path_nodes = path_nodes
        self.path_width = path_width

        center = path_nodes[:, 0] + 1e-5 * np.random.randn(*path_nodes[:, 0].shape)
        width = path_width

        # loop around
        center = np.concatenate([center, center[:1]])
        width = np.concatenate([width, width[:1]])

        if interpolate:
            center = interpolate_smooth(center, fixed_distance=fixed_distance)
            width = interpolate_smooth(width, fixed_number=len(center))

        d = np.diff(center, axis=0, append=center[:1])
        n = np.stack([-d[:, 2], np.zeros_like(d[:, 0]), d[:, 0]], axis=1)
        n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-5)

        left = center + n * (width / 2)
        right = center - n * (width / 2)

        self.center = center
        self.left = left
        self.right = right

    @cached_property
    def track(self):
        return homogeneous(self.center)

    @cached_property
    def track_left(self):
        return homogeneous(self.left)

    @cached_property
    def track_right(self):
        return homogeneous(self.right)
