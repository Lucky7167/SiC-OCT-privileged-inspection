from __future__ import annotations

import numpy as np


def pairwise_distances(points):
    coordinates = np.asarray(points, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] < 1:
        raise ValueError("points must be a two-dimensional coordinate array")
    if not np.isfinite(coordinates).all():
        raise ValueError("points must be finite")
    return np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)


def nearest_neighbor(points, start=0):
    coordinates = np.asarray(points, dtype=float)
    distances = pairwise_distances(coordinates)
    count = len(coordinates)
    if count == 0:
        return []
    if np.isscalar(start):
        first = int(start)
        if first < 0 or first >= count:
            raise ValueError("start index is out of range")
    else:
        origin = np.asarray(start, dtype=float)
        if origin.shape != (coordinates.shape[1],):
            raise ValueError("start coordinates must match the point dimension")
        first = int(np.argmin(np.linalg.norm(coordinates - origin, axis=1)))
    route = [first]
    remaining = set(range(count)) - {first}
    while remaining:
        nearest = min(remaining, key=lambda index: (distances[route[-1], index], index))
        route.append(nearest)
        remaining.remove(nearest)
    return route


def route_length(points, route, return_to_start=False):
    distances = pairwise_distances(points)
    ordered = list(map(int, route))
    if len(ordered) != len(distances) or set(ordered) != set(range(len(distances))):
        raise ValueError("route must visit every point exactly once")
    if len(ordered) < 2:
        return 0.0
    total = sum(distances[a, b] for a, b in zip(ordered[:-1], ordered[1:]))
    if return_to_start:
        total += distances[ordered[-1], ordered[0]]
    return float(total)


def two_opt(points, route=None, return_to_start=False, max_passes=None):
    coordinates = np.asarray(points, dtype=float)
    distances = pairwise_distances(coordinates)
    ordered = nearest_neighbor(coordinates) if route is None else list(map(int, route))
    count = len(ordered)
    if len(ordered) != count or set(ordered) != set(range(count)):
        raise ValueError("route must visit every point exactly once")
    if max_passes is not None and max_passes < 1:
        raise ValueError("max_passes must be positive")
    passes = 0
    improved = True
    while improved and (max_passes is None or passes < max_passes):
        improved = False
        passes += 1
        for i in range(1, count - 1):
            for j in range(i + 1, count):
                before = ordered[i - 1]
                first = ordered[i]
                last = ordered[j]
                after = ordered[j + 1] if j + 1 < count else None
                old = distances[before, first]
                new = distances[before, last]
                if after is not None:
                    old += distances[last, after]
                    new += distances[first, after]
                elif return_to_start:
                    old += distances[last, ordered[0]]
                    new += distances[first, ordered[0]]
                if new < old - 1e-12:
                    ordered[i:j + 1] = reversed(ordered[i:j + 1])
                    improved = True
        if count < 4:
            break
    return ordered


def nearest_neighbor_2opt(points, start=0, return_to_start=False, max_passes=None):
    route = nearest_neighbor(points, start)
    return two_opt(points, route, return_to_start, max_passes)


def plan_route(points, start=0, return_to_start=False, max_passes=None):
    return nearest_neighbor_2opt(points, start, return_to_start, max_passes)
