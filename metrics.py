"""Distance-based verification metrics for ReID embeddings."""

import numpy as np


def _rates(positive_distances, negative_distances, threshold):
    """Return TAR and FAR when distances below ``threshold`` are matches."""
    tar = np.mean(positive_distances < threshold)
    far = np.mean(negative_distances < threshold)
    return float(tar), float(far)


def _tar_at_far(positive_distances, negative_distances, target_far):
    """Find the most permissive empirical threshold within a FAR budget."""
    allowed_false_matches = int(np.floor(target_far * len(negative_distances)))
    sorted_negatives = np.sort(negative_distances)
    if allowed_false_matches >= len(sorted_negatives):
        threshold = float("inf")
    else:
        # Matching uses a strict '<', so this value admits only negatives below it.
        threshold = float(sorted_negatives[allowed_false_matches])
    tar, far = _rates(positive_distances, negative_distances, threshold)
    return tar, far, threshold


def _roc_auc(positive_distances, negative_distances):
    """Probability that a random positive has a smaller distance than a negative."""
    sorted_negatives = np.sort(negative_distances)
    lower = np.searchsorted(sorted_negatives, positive_distances, side="left")
    upper = np.searchsorted(sorted_negatives, positive_distances, side="right")
    greater = len(sorted_negatives) - upper
    ties = upper - lower
    return float(np.mean((greater + 0.5 * ties) / len(sorted_negatives)))


def _equal_error_rate(positive_distances, negative_distances):
    """Return the empirical operating point where FAR and FRR are closest."""
    sorted_positives = np.sort(positive_distances)
    sorted_negatives = np.sort(negative_distances)
    candidates = np.concatenate((
        [float("-inf")], np.unique(np.concatenate(
            (positive_distances, negative_distances))), [float("inf")]))
    tar = np.searchsorted(sorted_positives, candidates, side="left") / len(
        sorted_positives)
    far = np.searchsorted(sorted_negatives, candidates, side="left") / len(
        sorted_negatives)
    frr = 1.0 - tar
    eer = (far + frr) / 2.0
    best = np.lexsort((candidates, eer, np.abs(far - frr)))[0]
    return float(eer[best]), float(candidates[best])


def verification_metrics(positive_distances, negative_distances, threshold,
                         target_fars=(0.01,)):
    """Calculate fixed-threshold and threshold-independent ReID metrics."""
    positive = np.asarray(positive_distances, dtype=np.float64).reshape(-1)
    negative = np.asarray(negative_distances, dtype=np.float64).reshape(-1)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("positive and negative distances must not be empty")
    if not np.all(np.isfinite(positive)) or not np.all(np.isfinite(negative)):
        raise ValueError("distances must contain only finite values")

    eer, eer_threshold = _equal_error_rate(positive, negative)
    # With no deployment threshold configured, use the empirical EER point for
    # fixed-threshold diagnostic metrics. Low-FAR thresholds remain separate.
    threshold = eer_threshold if threshold is None else float(threshold)
    tar, far = _rates(positive, negative, threshold)
    metrics = {
        "positive_mean_distance": float(np.mean(positive)),
        "positive_p95_distance": float(np.percentile(positive, 95)),
        "negative_mean_distance": float(np.mean(negative)),
        "negative_p05_distance": float(np.percentile(negative, 5)),
        "verification_threshold": threshold,
        "tar": tar,
        "frr": 1.0 - tar,
        "tnr": 1.0 - far,
        "far": far,
        "threshold_accuracy": float(np.mean(
            (positive < threshold) & (negative >= threshold)))
            if positive.size == negative.size else None,
        "roc_auc": _roc_auc(positive, negative),
    }
    metrics["eer"] = eer
    metrics["eer_threshold"] = eer_threshold

    for target_far in target_fars:
        tar_at_far, empirical_far, operating_threshold = _tar_at_far(
            positive, negative, target_far)
        percent = f"{target_far * 100:.10f}".rstrip("0").rstrip(".")
        suffix = percent.replace(".", "p")
        metrics[f"tar_at_far_{suffix}pct"] = tar_at_far
        metrics[f"far_at_target_{suffix}pct"] = empirical_far
        metrics[f"threshold_at_far_{suffix}pct"] = operating_threshold
    return metrics
