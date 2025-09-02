# utils.py
import re
import numpy as np
from collections import Counter, defaultdict
import logging
from typing import List, Optional, Iterable, Generator, Hashable, Any

# ===================== LOGGING CONFIG =====================
log = logging.getLogger("Utils")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# ===================== MAJORITY VOTE =====================

_normalize_re = re.compile(r"[^\w\s]")
MIN_CHARS = 6
junk = []
def _normalize_text(s: str) -> str:
    """Lower, remove punctuation, collapse whitespace for voting/compare."""
    if not s:
        return ""
    s = s.lower().strip()
    s = _normalize_re.sub("", s)
    s = " ".join(s.split())
    return s

# ===================== MAJORITY VOTE =====================

def majority_vote(
    items: List[str],
    weights: Optional[List[float]] = None,
    min_conf: float = 0.0,
    tie_strategy: str = "first"
) -> Optional[str]:
    """
    Robust majority vote (weighted or unweighted).

    Args:
        items: list of candidate strings
        weights: optional weights/confidences, defaults to uniform
        min_conf: discard votes with weight < min_conf
        tie_strategy: tie-breaking rule
            - "first": first-seen among tied
            - "highest_weight": item with highest single vote weight
            - "random": random choice among tied

    Returns:
        str | None: normalized winning candidate
    """
    if not items:
        return None
    if weights and len(weights) != len(items):
        raise ValueError("Items and weights must have same length")

    weights = weights or [1.0] * len(items)

    scores = defaultdict(float)
    item_max_weight = defaultdict(float)

    for item, w in zip(items, weights):
        norm_item = _normalize_text(item)
        if not norm_item or w < min_conf:
            continue
        scores[norm_item] += w
        item_max_weight[norm_item] = max(item_max_weight[norm_item], w)

    if not scores:
        return None

    max_score = max(scores.values())
    candidates = [it for it, sc in scores.items() if sc == max_score]

    if len(candidates) == 1:
        return candidates[0]

    if tie_strategy == "highest_weight":
        return max(candidates, key=lambda it: item_max_weight[it])
    elif tie_strategy == "first":
        for item in items:
            if _normalize_text(item) in candidates:
                return _normalize_text(item)
    elif tie_strategy == "random":
        return np.random.choice(candidates)

    return candidates[0]  # default fallback


# ===================== WEIGHTED VOTE =====================

def weighted_vote(items: List[str], weights: List[float]) -> Optional[str]:
    """
    Weighted majority vote with text normalization and stability improvements.
    
    Args:
        items: list of strings
        weights: list of corresponding weights (floats)
    Returns:
        Most likely item (string) with highest weighted score, or None
    """
    if not items or not weights:
        return None
    if len(items) != len(weights):
        raise ValueError("Items and weights must have the same length")

    vote_scores: dict[str, float] = {}

    for item, weight in zip(items, weights):
        # Normalize + trim whitespace + lowercase
        norm_item = _normalize_text(item).strip().lower()

        # Skip empties or too-short garbage
        if not norm_item or len(norm_item) < 2:
            continue

        # Collapse runs of duplicate tokens (fixes "asus asus asus")
        deduped = " ".join(dict.fromkeys(norm_item.split()))

        vote_scores[deduped] = vote_scores.get(deduped, 0.0) + weight

    if not vote_scores:
        return None

    # Pick the winner with highest score
    winner, score = max(vote_scores.items(), key=lambda x: x[1])

    # Debug info for traceability
    log.debug(f"Weighted votes: {vote_scores}, winner: {winner} (score={score:.2f})")

    return winner


import logging

log = logging.getLogger(__name__)

def weighted_majority(
    items: List[str],
    weights: List[float],
    min_conf: float = 0.0,
    tie_strategy: str = "highest_weight"
) -> Optional[str]:
    """
    Weighted majority vote with normalization and better tie-breaking.
    
    Args:
        items (List[str]): list of candidate strings
        weights (List[float]): corresponding weights/confidences
        min_conf (float): discard votes with weight < min_conf
        tie_strategy (str): how to break ties
            - "highest_weight": winner is item with highest single weight
            - "first": winner is first-seen among tied
            - "random": choose randomly among tied
    
    Returns:
        str | None: normalized winner
    """
    if not items or not weights or len(items) != len(weights):
        return None

    scores = defaultdict(float)
    item_max_weight = defaultdict(float)

    for item, w in zip(items, weights):
        norm_item = _normalize_text(item)
        if not norm_item or w < min_conf:
            continue
        scores[norm_item] += w
        item_max_weight[norm_item] = max(item_max_weight[norm_item], w)

    if not scores:
        return None

    # Find top score(s)
    max_score = max(scores.values())
    candidates = [it for it, sc in scores.items() if sc == max_score]

    if len(candidates) == 1:
        winner = candidates[0]
    else:
        if tie_strategy == "highest_weight":
            winner = max(candidates, key=lambda it: item_max_weight[it])
        elif tie_strategy == "first":
            for it, _ in zip(items, weights):
                if _normalize_text(it) in candidates:
                    winner = _normalize_text(it)
                    break
        elif tie_strategy == "random":
            winner = np.random.choice(candidates)
        else:
            winner = candidates[0]

    log.debug(f"Weighted scores={dict(scores)}, candidates={candidates}, winner={winner}")
    return winner

def weighted_majority_vote(transcripts: list[str], weights: list[float] = None) -> str:
    """Improved weighted majority vote with fallback."""
    if not transcripts:
        return ""
    
    from collections import defaultdict
    scores = defaultdict(float)
    weights = weights or [1.0] * len(transcripts)
    
    for t, w in zip(transcripts, weights):
        scores[t.strip()] += w
    
    # Pick best by score
    best, best_score = max(scores.items(), key=lambda x: x[1])
    
    # If score is low, fallback to last transcript
    if best_score < 1.5:  # tweak threshold
        return transcripts[-1]
    
    return best


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

def dedup_text(text: str) -> str:
    words = text.split()
    cleaned = []
    for w in words:
        if not cleaned or w != cleaned[-1]:
            cleaned.append(w)
    return " ".join(cleaned)
