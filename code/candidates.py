from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.cluster import KMeans

K_CLUSTERS = 2


def per_wafer_kmeans(embeddings, wafer_ids, random_state=0, n_init=10):
    features = np.asarray(embeddings, dtype=float)
    wafers = np.asarray(wafer_ids).astype(str)
    if features.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")
    if len(features) != len(wafers):
        raise ValueError("wafer_ids must have one entry per embedding")
    if not np.isfinite(features).all():
        raise ValueError("embeddings must be finite")
    labels = np.full(len(features), -1, dtype=int)
    scores = np.zeros(len(features), dtype=float)
    anomalous = np.zeros(len(features), dtype=bool)
    background = {}
    for wafer in sorted(np.unique(wafers)):
        indices = np.flatnonzero(wafers == wafer)
        if len(indices) < K_CLUSTERS:
            raise ValueError(f"wafer {wafer} has fewer than two embeddings")
        if np.unique(features[indices], axis=0).shape[0] < K_CLUSTERS:
            raise ValueError(f"wafer {wafer} has fewer than two distinct embeddings")
        model = KMeans(
            n_clusters=K_CLUSTERS,
            random_state=random_state,
            n_init=n_init,
        )
        wafer_labels = model.fit_predict(features[indices])
        counts = np.bincount(wafer_labels, minlength=K_CLUSTERS)
        normal = int(np.argmax(counts))
        background[wafer] = normal
        labels[indices] = wafer_labels
        anomalous[indices] = wafer_labels != normal
        scores[indices] = np.linalg.norm(
            features[indices] - model.cluster_centers_[normal], axis=1
        )
    return {
        "cluster": labels,
        "anomalous": anomalous,
        "anomaly_score": scores,
        "background_cluster": background,
    }


def eight_connected_components(anomalous):
    mask = np.asarray(anomalous, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("anomalous must be a two-dimensional mask")
    return ndimage.label(mask, structure=np.ones((3, 3), dtype=int))


def merge_anomalous_patches(anomalous, scores, max_width=None, max_height=None):
    mask = np.asarray(anomalous, dtype=bool)
    values = np.asarray(scores, dtype=float)
    if mask.shape != values.shape or mask.ndim != 2:
        raise ValueError("anomalous and scores must be matching two-dimensional arrays")
    labels, count = eight_connected_components(mask)
    records = []
    for component in range(1, count + 1):
        rows, cols = np.nonzero(labels == component)
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        height = r1 - r0 if max_height is None else min(r1 - r0, int(max_height))
        width = c1 - c0 if max_width is None else min(c1 - c0, int(max_width))
        if height <= 0 or width <= 0:
            raise ValueError("maximum ROI dimensions must be positive")
        for top in range(r0, r1, height):
            for left in range(c0, c1, width):
                bottom = min(top + height, r1)
                right = min(left + width, c1)
                selected = labels[top:bottom, left:right] == component
                if not selected.any():
                    continue
                local_rows, local_cols = np.nonzero(selected)
                y0 = top + int(local_rows.min())
                y1 = top + int(local_rows.max()) + 1
                x0 = left + int(local_cols.min())
                x1 = left + int(local_cols.max()) + 1
                component_scores = values[top:bottom, left:right][selected]
                records.append({
                    "x": (x0 + x1) / 2.0,
                    "y": (y0 + y1) / 2.0,
                    "w": x1 - x0,
                    "h": y1 - y0,
                    "a": float(np.max(component_scores)),
                    "x0": x0,
                    "x1": x1,
                    "y0": y0,
                    "y1": y1,
                    "n_patches": int(selected.sum()),
                })
    return records


def detect_candidates(
    embeddings,
    patch_rows,
    patch_cols,
    wafer_ids,
    random_state=0,
    n_init=10,
    max_width=None,
    max_height=None,
):
    rows = np.asarray(patch_rows, dtype=int)
    cols = np.asarray(patch_cols, dtype=int)
    wafers = np.asarray(wafer_ids).astype(str)
    if rows.ndim != 1 or cols.shape != rows.shape or wafers.shape != rows.shape:
        raise ValueError("patch coordinates and wafer_ids must be one-dimensional and aligned")
    if len(rows) != len(embeddings):
        raise ValueError("patch coordinates must have one entry per embedding")
    if (rows < 0).any() or (cols < 0).any():
        raise ValueError("patch coordinates must be non-negative")
    clustering = per_wafer_kmeans(embeddings, wafers, random_state, n_init)
    records = []
    for wafer in sorted(np.unique(wafers)):
        indices = np.flatnonzero(wafers == wafer)
        shape = (int(rows[indices].max()) + 1, int(cols[indices].max()) + 1)
        mask = np.zeros(shape, dtype=bool)
        scores = np.full(shape, np.nan, dtype=float)
        mask[rows[indices], cols[indices]] = clustering["anomalous"][indices]
        scores[rows[indices], cols[indices]] = clustering["anomaly_score"][indices]
        for record in merge_anomalous_patches(mask, scores, max_width, max_height):
            records.append({"wafer": wafer, **record})
    columns = ["wafer", "x", "y", "w", "h", "a", "x0", "x1", "y0", "y1", "n_patches"]
    return {**clustering, "rois": pd.DataFrame(records, columns=columns)}


def oct_only_en_face_candidates(
    embeddings,
    patch_rows,
    patch_cols,
    wafer_ids,
    random_state=0,
    n_init=10,
    max_width=None,
    max_height=None,
):
    return detect_candidates(
        embeddings,
        patch_rows,
        patch_cols,
        wafer_ids,
        random_state,
        n_init,
        max_width,
        max_height,
    )
