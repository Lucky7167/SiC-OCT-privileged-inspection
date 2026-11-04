from __future__ import annotations

import numpy as np

CANDIDATE_THRESHOLD_SD = 3.0
MIN_CONSECUTIVE_BSCANS = 3
MAD_SCALE = 1.4826
SNR_REF_DB = 6.0
CONTINUITY_REFERENCE_RUN = 5
MIN_DEPTH_INTERVAL_OVERLAP = 0.30
MAX_CENTROID_DEPTH_DIFFERENCE_UM = 10.0
SHADOW_ATTENUATION_DB = 6.0
SATURATION_FULL_SCALE_FRACTION = 0.98
GROUP_REFRACTIVE_INDEX = 2.59
MIN_SUBSURFACE_DEPTH_UM = 10.0


def raw_oct_candidate_mask(
    volume,
    local_mean,
    local_std,
    threshold_sd=CANDIDATE_THRESHOLD_SD,
    min_consecutive_bscans=MIN_CONSECUTIVE_BSCANS,
):
    intensities = np.asarray(volume, dtype=float)
    mean = np.asarray(local_mean, dtype=float)
    std = np.asarray(local_std, dtype=float)
    if intensities.ndim != 3:
        raise ValueError("volume must have B-scan, depth, and A-line axes")
    if min_consecutive_bscans < 1:
        raise ValueError("min_consecutive_bscans must be positive")
    try:
        above = intensities > mean + float(threshold_sd) * std
    except ValueError as error:
        raise ValueError("local background statistics must broadcast to the volume") from error
    persistent = np.zeros_like(above, dtype=bool)
    run = np.zeros(above.shape[1:], dtype=int)
    for bscan in range(len(above)):
        run = (run + 1) * above[bscan]
        completed = run >= min_consecutive_bscans
        for offset in range(min_consecutive_bscans):
            persistent[bscan - offset] |= completed
    return persistent


def detect_raw_oct_candidates(
    volume,
    background_volumes,
    threshold_sd=CANDIDATE_THRESHOLD_SD,
    min_consecutive_bscans=MIN_CONSECUTIVE_BSCANS,
):
    background = np.asarray(background_volumes, dtype=float)
    if background.ndim != 4:
        raise ValueError("background_volumes must have sample, B-scan, depth, and A-line axes")
    if background.shape[1:] != np.shape(volume):
        raise ValueError("background_volumes and volume must have matching scan dimensions")
    return raw_oct_candidate_mask(
        volume,
        background.mean(axis=0),
        background.std(axis=0),
        threshold_sd,
        min_consecutive_bscans,
    )


def snr_db(defect_intensity, background_intensity, epsilon=1e-12):
    defect = np.asarray(defect_intensity, dtype=float)
    background = np.asarray(background_intensity, dtype=float)
    if defect.size == 0 or background.size == 0:
        raise ValueError("defect and background samples must be non-empty")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    defect_median = float(np.median(defect))
    background_median = float(np.median(background))
    mad = float(np.median(np.abs(background - background_median)))
    signal = max(defect_median - background_median, 0.0)
    denominator = MAD_SCALE * mad + epsilon
    if signal == 0:
        return -np.inf
    return float(20.0 * np.log10(signal / denominator))


def shadow_fraction(
    post_structure,
    local_background,
    defect_alines=None,
    attenuation_db=SHADOW_ATTENUATION_DB,
    epsilon=1e-12,
):
    post = np.asarray(post_structure, dtype=float)
    background = np.asarray(local_background, dtype=float)
    if post.shape != background.shape or post.ndim == 0:
        raise ValueError("post_structure and local_background must have matching non-scalar shapes")
    if defect_alines is not None:
        selected = np.asarray(defect_alines, dtype=bool)
        if selected.ndim != 1 or len(selected) != len(post):
            raise ValueError("defect_alines must select entries on the first axis")
        post = post[selected]
        background = background[selected]
    if len(post) == 0:
        return 0.0
    axes = tuple(range(1, post.ndim))
    post_level = post.mean(axis=axes) if axes else post
    background_level = background.mean(axis=axes) if axes else background
    attenuation = 20.0 * np.log10(
        (np.maximum(background_level, 0.0) + epsilon)
        / (np.maximum(post_level, 0.0) + epsilon)
    )
    return float(np.mean(attenuation > attenuation_db))


def saturation_fraction(
    raw_detector,
    defect_mask,
    full_scale,
    full_scale_fraction=SATURATION_FULL_SCALE_FRACTION,
):
    raw = np.asarray(raw_detector, dtype=float)
    mask = np.asarray(defect_mask, dtype=bool)
    if raw.shape != mask.shape:
        raise ValueError("raw_detector and defect_mask must have matching shapes")
    if full_scale <= 0:
        raise ValueError("full_scale must be positive")
    if not mask.any():
        return 0.0
    return float(np.mean(raw[mask] >= full_scale_fraction * full_scale))


def _depth_interval(structure):
    if isinstance(structure, dict):
        lower = structure.get("z0", structure.get("top"))
        upper = structure.get("z1", structure.get("bottom"))
    else:
        lower, upper = structure
    lower, upper = float(lower), float(upper)
    if not np.isfinite([lower, upper]).all() or upper <= lower:
        raise ValueError("depth intervals must be finite with positive extent")
    return lower, upper


def structures_linked(
    first,
    second,
    min_overlap=MIN_DEPTH_INTERVAL_OVERLAP,
    max_centroid_difference_um=MAX_CENTROID_DEPTH_DIFFERENCE_UM,
):
    first_lower, first_upper = _depth_interval(first)
    second_lower, second_upper = _depth_interval(second)
    overlap = max(0.0, min(first_upper, second_upper) - max(first_lower, second_lower))
    fraction = overlap / min(first_upper - first_lower, second_upper - second_lower)
    first_centroid = (first_lower + first_upper) / 2.0
    second_centroid = (second_lower + second_upper) / 2.0
    return bool(
        fraction >= min_overlap
        and abs(first_centroid - second_centroid) <= max_centroid_difference_um
    )


def longest_linked_run(bscan_structures):
    previous_lengths = []
    longest = 0
    for structures in bscan_structures:
        current_lengths = []
        for current in structures:
            linked = [
                length
                for previous, length in zip(previous_structures, previous_lengths)
                if structures_linked(previous, current)
            ] if longest else []
            current_lengths.append(1 + max(linked, default=0))
        if current_lengths:
            longest = max(longest, max(current_lengths))
        previous_structures = list(structures)
        previous_lengths = current_lengths
    return int(longest)


def contrast_continuity(bscan_structures):
    return float(
        min(longest_linked_run(bscan_structures) / CONTINUITY_REFERENCE_RUN, 1.0)
    )


def effective_depth_um(optical_path_length_um, group_index=GROUP_REFRACTIVE_INDEX):
    if group_index <= 0:
        raise ValueError("group_index must be positive")
    return np.asarray(optical_path_length_um, dtype=float) / group_index


def q_star_intermediates(
    deff,
    dmax,
    snr,
    rho_shadow,
    rho_sat,
    c_cont,
    has_depth_structure=True,
    snr_ref=SNR_REF_DB,
):
    if snr_ref <= 0:
        raise ValueError("snr_ref must be positive")
    present = bool(has_depth_structure)
    depth_term = float(deff) / float(dmax) if present and dmax != 0 else 0.0
    snr_term = float(np.clip(float(snr) / snr_ref, 0.0, 1.0)) if present else 0.0
    shadow_term = 1.0 - float(rho_shadow)
    saturation_term = 1.0 - float(rho_sat)
    continuity_term = float(c_cont) if present else 0.0
    q_star = float(np.clip(
        (depth_term + snr_term + shadow_term + saturation_term + continuity_term) / 5.0,
        0.0,
        1.0,
    ))
    return {
        "deff": float(deff),
        "dmax": float(dmax),
        "snr": float(snr),
        "snr_ref": float(snr_ref),
        "rho_shadow": float(rho_shadow),
        "rho_sat": float(rho_sat),
        "c_cont": float(c_cont),
        "has_depth_structure": present,
        "depth_term": depth_term,
        "snr_term": snr_term,
        "shadow_term": shadow_term,
        "saturation_term": saturation_term,
        "continuity_term": continuity_term,
        "q_star": q_star,
    }


def depth_evidence_score(*args, **kwargs):
    return q_star_intermediates(*args, **kwargs)["q_star"]
