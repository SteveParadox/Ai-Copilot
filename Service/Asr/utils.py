# utils.py
import numpy as np
from collections import Counter
import logging
from typing import List, Optional, Iterable, Generator, Hashable, Any

# ===================== LOGGING CONFIG =====================
log = logging.getLogger("Utils")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# ===================== MAJORITY VOTE =====================
def majority_vote(items: List[Hashable]) -> Optional[Hashable]:
    """
    Return the most common element in a list.
    Tie-breaker: first encountered among the most common.
    Args:
        items: list of hashable items
    Returns:
        most common item or None if list is empty
    """
    if not items:
        log.debug("Empty list provided to majority_vote.")
        return None

    counter = Counter(items)
    most_common_items = counter.most_common()
    max_count = most_common_items[0][1]
    tied = [item for item, count in most_common_items if count == max_count]
    if len(tied) > 1:
        log.info(f"Tie detected among items {tied}, returning first encountered.")
    return tied[0]

# ===================== WEIGHTED VOTE =====================
def weighted_vote(items: List[Hashable], weights: List[float]) -> Hashable:
    """
    Weighted majority vote.
    Args:
        items: list of elements
        weights: list of corresponding weights (floats)
    Returns:
        element with highest weighted sum
    Raises:
        ValueError if items and weights have different lengths
    """
    if len(items) != len(weights):
        raise ValueError("Items and weights must have the same length")

    vote_scores: dict[Hashable, float] = {}
    for item, weight in zip(items, weights):
        vote_scores[item] = vote_scores.get(item, 0.0) + weight

    winner = max(vote_scores.items(), key=lambda x: x[1])[0]
    log.debug(f"Weighted votes: {vote_scores}, winner: {winner}")
    return winner

# ===================== ARRAY NORMALIZATION =====================
def normalize_array(arr: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """
    Min-max normalize a numpy array to the specified range.
    Args:
        arr: numpy array
        min_val: minimum of normalized range
        max_val: maximum of normalized range
    Returns:
        normalized array
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        log.debug("Empty array provided to normalize_array.")
        return np.array([], dtype=np.float32)

    min_a = arr.min()
    max_a = arr.max()
    if max_a - min_a == 0:
        log.debug("Array has zero variance, returning array filled with min_val.")
        return np.full_like(arr, min_val)
    return (arr - min_a) / (max_a - min_a) * (max_val - min_val) + min_val

# ===================== SMOOTHING =====================
def smooth_predictions(preds: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Simple moving average smoothing of predictions.
    Args:
        preds: numpy array of predictions
        window_size: size of smoothing window
    Returns:
        smoothed numpy array
    """
    if len(preds) < window_size:
        log.debug(f"Window size {window_size} > preds length {len(preds)}, returning original array.")
        return preds

    smoothed = np.convolve(preds, np.ones(window_size)/window_size, mode='valid')
    pad_left = window_size // 2
    pad_right = len(preds) - len(smoothed) - pad_left
    smoothed = np.pad(smoothed, (pad_left, pad_right), mode='edge')
    return smoothed

# ===================== LIST CHUNKING =====================
def chunk_list(lst: List[Any], chunk_size: int) -> Generator[List[Any], None, None]:
    """
    Split list into chunks of given size.
    Args:
        lst: list to split
        chunk_size: size of each chunk
    Yields:
        successive chunks as lists
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]
