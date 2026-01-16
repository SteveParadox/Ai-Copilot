"""
ASR Utilities - Extended Edition
Comprehensive utilities for ASR processing including voting algorithms, normalization,
fuzzy matching, N-gram analysis, and beam search decoding.
"""

import re
import logging
import hashlib
from collections import Counter, defaultdict, deque
from typing import List, Optional, Tuple, Generator, Any, Dict, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import heapq

import numpy as np

# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

class VotingConfig:
    """Configuration for voting algorithms."""
    MIN_CHARS: int = 6
    MIN_CONFIDENCE: float = 0.0
    NORMALIZATION_AGGRESSIVE: bool = True
    
    # Text cleaning
    REMOVE_PUNCTUATION: bool = True
    LOWERCASE: bool = True
    COLLAPSE_WHITESPACE: bool = True
    DEDUPE_TOKENS: bool = True


class FuzzyMatchConfig:
    """Configuration for fuzzy matching."""
    # Levenshtein thresholds
    EXACT_MATCH_THRESHOLD: float = 1.0
    STRONG_MATCH_THRESHOLD: float = 0.85
    WEAK_MATCH_THRESHOLD: float = 0.70
    
    # Edit weights
    INSERTION_COST: float = 1.0
    DELETION_COST: float = 1.0
    SUBSTITUTION_COST: float = 1.0
    
    # Optimization
    MAX_DISTANCE_CUTOFF: Optional[int] = None  # Early termination


class NGramConfig:
    """Configuration for N-gram analysis."""
    DEFAULT_N: int = 2
    MAX_N: int = 5
    MIN_FREQUENCY: int = 2
    
    # Quality metrics
    PERPLEXITY_SMOOTHING: float = 1e-10


# ============================================================================
# Text Normalization (from previous version)
# ============================================================================

_PUNCTUATION_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")
_REPEATED_WORDS_RE = re.compile(r'\b(\w+)(\s+\1)+\b', re.IGNORECASE)


def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    collapse_whitespace: bool = True,
    dedupe_consecutive: bool = True,
) -> str:
    """Normalize text for comparison and voting."""
    if not text:
        return ""
    
    if lowercase:
        text = text.lower()
    
    text = text.strip()
    
    if remove_punctuation:
        text = _PUNCTUATION_RE.sub("", text)
    
    if dedupe_consecutive:
        text = _REPEATED_WORDS_RE.sub(r"\1", text)
    
    if collapse_whitespace:
        text = _WHITESPACE_RE.sub(" ", text)
    
    return text.strip()


def dedupe_tokens(text: str) -> str:
    """Remove consecutive duplicate tokens while preserving order."""
    if not text:
        return ""
    
    tokens = text.split()
    if not tokens:
        return ""
    
    deduped = list(dict.fromkeys(tokens))
    return " ".join(deduped)


# ============================================================================
# Levenshtein Distance & Fuzzy Matching
# ============================================================================

def levenshtein_distance(
    s1: str,
    s2: str,
    insertion_cost: float = 1.0,
    deletion_cost: float = 1.0,
    substitution_cost: float = 1.0,
    max_distance: Optional[int] = None,
) -> int:
    """
    Compute Levenshtein (edit) distance between two strings.
    
    Uses dynamic programming with optional early termination for efficiency.
    
    Args:
        s1: First string
        s2: Second string
        insertion_cost: Cost of inserting a character
        deletion_cost: Cost of deleting a character
        substitution_cost: Cost of substituting a character
        max_distance: Early termination if distance exceeds this
    
    Returns:
        Minimum edit distance
    
    Examples:
        >>> levenshtein_distance("kitten", "sitting")
        3
        >>> levenshtein_distance("hello", "hello")
        0
        >>> levenshtein_distance("hello", "hallo")
        1
    
    Time Complexity: O(m*n) where m, n are string lengths
    Space Complexity: O(min(m, n)) with optimization
    """
    if s1 == s2:
        return 0
    
    if not s1:
        return len(s2) * insertion_cost
    if not s2:
        return len(s1) * deletion_cost
    
    # Ensure s1 is shorter for space optimization
    if len(s1) > len(s2):
        s1, s2 = s2, s1
        insertion_cost, deletion_cost = deletion_cost, insertion_cost
    
    m, n = len(s1), len(s2)
    
    # Early termination if max_distance exceeded
    if max_distance is not None and abs(m - n) > max_distance:
        return max_distance + 1
    
    # Use two rows instead of full matrix (space optimization)
    prev_row = list(range(n + 1))
    curr_row = [0] * (n + 1)
    
    for i in range(1, m + 1):
        curr_row[0] = i * deletion_cost
        
        # Track minimum in row for early termination
        min_in_row = curr_row[0]
        
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                # Characters match, no cost
                cost = 0
                curr_row[j] = prev_row[j - 1]
            else:
                # Take minimum of three operations
                curr_row[j] = min(
                    prev_row[j] + deletion_cost,      # Delete from s1
                    curr_row[j - 1] + insertion_cost,  # Insert into s1
                    prev_row[j - 1] + substitution_cost  # Substitute
                )
            
            min_in_row = min(min_in_row, curr_row[j])
        
        # Early termination: if best possible distance in row exceeds max
        if max_distance is not None and min_in_row > max_distance:
            return max_distance + 1
        
        # Swap rows
        prev_row, curr_row = curr_row, prev_row
    
    return int(prev_row[n])


def normalized_levenshtein(s1: str, s2: str, **kwargs) -> float:
    """
    Compute normalized Levenshtein distance (0 = different, 1 = identical).
    
    Args:
        s1: First string
        s2: Second string
        **kwargs: Additional arguments for levenshtein_distance
    
    Returns:
        Similarity score between 0 and 1
    
    Examples:
        >>> normalized_levenshtein("kitten", "sitting")
        0.571...
        >>> normalized_levenshtein("hello", "hello")
        1.0
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    max_len = max(len(s1), len(s2))
    distance = levenshtein_distance(s1, s2, **kwargs)
    
    return 1.0 - (distance / max_len)


def fuzzy_match(
    query: str,
    candidates: List[str],
    threshold: float = FuzzyMatchConfig.WEAK_MATCH_THRESHOLD,
    normalize: bool = True,
    return_scores: bool = False,
) -> List[str] | List[Tuple[str, float]]:
    """
    Find fuzzy matches for query in candidates using Levenshtein distance.
    
    Args:
        query: Query string
        candidates: List of candidate strings
        threshold: Minimum similarity score (0-1)
        normalize: Whether to normalize texts before matching
        return_scores: Whether to return (text, score) tuples
    
    Returns:
        List of matching candidates (or tuples if return_scores=True)
    
    Examples:
        >>> candidates = ["hello world", "helo world", "goodbye"]
        >>> fuzzy_match("hello world", candidates, threshold=0.8)
        ['hello world', 'helo world']
        >>> fuzzy_match("hello world", candidates, threshold=0.8, return_scores=True)
        [('hello world', 1.0), ('helo world', 0.909)]
    """
    if normalize:
        query_norm = normalize_text(query)
        candidates_norm = [normalize_text(c) for c in candidates]
    else:
        query_norm = query
        candidates_norm = candidates
    
    matches = []
    
    for i, (cand_norm, cand_orig) in enumerate(zip(candidates_norm, candidates)):
        score = normalized_levenshtein(query_norm, cand_norm)
        
        if score >= threshold:
            if return_scores:
                matches.append((cand_orig, score))
            else:
                matches.append(cand_orig)
    
    # Sort by score if returning scores
    if return_scores:
        matches.sort(key=lambda x: x[1], reverse=True)
    
    return matches


def find_closest_match(
    query: str,
    candidates: List[str],
    normalize: bool = True,
) -> Tuple[Optional[str], float]:
    """
    Find the closest match to query in candidates.
    
    Args:
        query: Query string
        candidates: List of candidate strings
        normalize: Whether to normalize texts
    
    Returns:
        (best_match, similarity_score) or (None, 0.0) if no candidates
    
    Examples:
        >>> candidates = ["hello", "helo", "goodbye"]
        >>> find_closest_match("hallo", candidates)
        ('hello', 0.8)
    """
    if not candidates:
        return None, 0.0
    
    if normalize:
        query_norm = normalize_text(query)
        candidates_norm = [normalize_text(c) for c in candidates]
    else:
        query_norm = query
        candidates_norm = candidates
    
    best_match = None
    best_score = 0.0
    
    for cand_norm, cand_orig in zip(candidates_norm, candidates):
        score = normalized_levenshtein(query_norm, cand_norm)
        
        if score > best_score:
            best_score = score
            best_match = cand_orig
    
    return best_match, best_score


@dataclass
class FuzzyMatchResult:
    """Result of fuzzy matching operation."""
    query: str
    matches: List[Tuple[str, float]]
    best_match: Optional[str] = None
    best_score: float = 0.0
    threshold: float = 0.0
    
    def __post_init__(self):
        if self.matches:
            self.best_match, self.best_score = self.matches[0]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "best_match": self.best_match,
            "best_score": round(self.best_score, 3),
            "num_matches": len(self.matches),
            "threshold": self.threshold,
            "top_3_matches": [
                {"text": text, "score": round(score, 3)}
                for text, score in self.matches[:3]
            ],
        }


def fuzzy_match_advanced(
    query: str,
    candidates: List[str],
    threshold: float = FuzzyMatchConfig.WEAK_MATCH_THRESHOLD,
    normalize: bool = True,
    max_results: Optional[int] = None,
) -> FuzzyMatchResult:
    """
    Advanced fuzzy matching with detailed results.
    
    Args:
        query: Query string
        candidates: List of candidate strings
        threshold: Minimum similarity score
        normalize: Whether to normalize texts
        max_results: Maximum number of matches to return
    
    Returns:
        FuzzyMatchResult with detailed matching information
    """
    matches = fuzzy_match(
        query, candidates,
        threshold=threshold,
        normalize=normalize,
        return_scores=True
    )
    
    if max_results:
        matches = matches[:max_results]
    
    return FuzzyMatchResult(
        query=query,
        matches=matches,
        threshold=threshold,
    )


# ============================================================================
# N-gram Analysis
# ============================================================================

@dataclass
class NGram:
    """Represents an N-gram with metadata."""
    tokens: Tuple[str, ...]
    n: int = field(init=False)
    
    def __post_init__(self):
        self.n = len(self.tokens)
    
    def __str__(self) -> str:
        return " ".join(self.tokens)
    
    def __hash__(self) -> int:
        return hash(self.tokens)


class NGramCounter:
    """Efficient N-gram counting and analysis."""
    
    def __init__(self, n: int = NGramConfig.DEFAULT_N):
        """
        Initialize N-gram counter.
        
        Args:
            n: Size of N-grams to extract
        """
        if n < 1:
            raise ValueError("n must be positive")
        if n > NGramConfig.MAX_N:
            log.warning(f"n={n} exceeds recommended max {NGramConfig.MAX_N}")
        
        self.n = n
        self.counts: Counter = Counter()
        self.total_ngrams = 0
    
    def add_text(self, text: str, normalize: bool = True) -> int:
        """
        Extract and count N-grams from text.
        
        Args:
            text: Input text
            normalize: Whether to normalize text first
        
        Returns:
            Number of N-grams added
        """
        if normalize:
            text = normalize_text(text)
        
        tokens = text.split()
        
        if len(tokens) < self.n:
            log.debug(f"Text too short for {self.n}-grams: {len(tokens)} tokens")
            return 0
        
        ngrams_added = 0
        
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            self.counts[ngram] += 1
            self.total_ngrams += 1
            ngrams_added += 1
        
        return ngrams_added
    
    def add_texts(self, texts: List[str], normalize: bool = True) -> int:
        """Add multiple texts."""
        total_added = 0
        for text in texts:
            total_added += self.add_text(text, normalize)
        return total_added
    
    def most_common(self, k: int = 10) -> List[Tuple[Tuple[str, ...], int]]:
        """
        Get k most common N-grams.
        
        Args:
            k: Number of N-grams to return
        
        Returns:
            List of (ngram_tuple, count) pairs
        """
        return self.counts.most_common(k)
    
    def get_frequency(self, ngram: Tuple[str, ...]) -> float:
        """
        Get relative frequency of N-gram.
        
        Args:
            ngram: N-gram tuple
        
        Returns:
            Frequency between 0 and 1
        """
        if self.total_ngrams == 0:
            return 0.0
        return self.counts[ngram] / self.total_ngrams
    
    def get_probability(
        self,
        ngram: Tuple[str, ...],
        smoothing: float = NGramConfig.PERPLEXITY_SMOOTHING,
    ) -> float:
        """
        Get smoothed probability of N-gram.
        
        Args:
            ngram: N-gram tuple
            smoothing: Add-k smoothing parameter
        
        Returns:
            Smoothed probability
        """
        vocab_size = len(self.counts)
        numerator = self.counts[ngram] + smoothing
        denominator = self.total_ngrams + smoothing * vocab_size
        return numerator / denominator if denominator > 0 else 0.0
    
    def filter_rare(self, min_count: int = NGramConfig.MIN_FREQUENCY) -> int:
        """
        Remove N-grams below minimum frequency.
        
        Args:
            min_count: Minimum count threshold
        
        Returns:
            Number of N-grams removed
        """
        before = len(self.counts)
        self.counts = Counter({
            ng: cnt for ng, cnt in self.counts.items()
            if cnt >= min_count
        })
        removed = before - len(self.counts)
        log.debug(f"Filtered {removed} rare N-grams (threshold={min_count})")
        return removed
    
    def to_dict(self) -> Dict[str, Any]:
        """Export statistics."""
        return {
            "n": self.n,
            "total_ngrams": self.total_ngrams,
            "unique_ngrams": len(self.counts),
            "top_10": [
                {" ".join(ng): cnt}
                for ng, cnt in self.most_common(10)
            ],
        }


def extract_ngrams(
    text: str,
    n: int = NGramConfig.DEFAULT_N,
    normalize: bool = True,
) -> List[Tuple[str, ...]]:
    """
    Extract all N-grams from text.
    
    Args:
        text: Input text
        n: Size of N-grams
        normalize: Whether to normalize text
    
    Returns:
        List of N-gram tuples
    
    Examples:
        >>> extract_ngrams("hello world foo", n=2)
        [('hello', 'world'), ('world', 'foo')]
        >>> extract_ngrams("one two three", n=3)
        [('one', 'two', 'three')]
    """
    if normalize:
        text = normalize_text(text)
    
    tokens = text.split()
    
    if len(tokens) < n:
        return []
    
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def ngram_overlap(
    text1: str,
    text2: str,
    n: int = NGramConfig.DEFAULT_N,
    normalize: bool = True,
) -> float:
    """
    Compute N-gram overlap between two texts (Jaccard similarity).
    
    Args:
        text1: First text
        text2: Second text
        n: Size of N-grams
        normalize: Whether to normalize texts
    
    Returns:
        Overlap score between 0 and 1
    
    Examples:
        >>> ngram_overlap("hello world", "hello there", n=1)
        0.333...  # 1 common word out of 3 unique
        >>> ngram_overlap("hello world foo", "hello world bar", n=2)
        0.333...  # 1 common bigram out of 3 unique
    """
    ngrams1 = set(extract_ngrams(text1, n, normalize))
    ngrams2 = set(extract_ngrams(text2, n, normalize))
    
    if not ngrams1 and not ngrams2:
        return 1.0
    if not ngrams1 or not ngrams2:
        return 0.0
    
    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)
    
    return len(intersection) / len(union)


def compute_bleu_score(
    hypothesis: str,
    reference: str,
    max_n: int = 4,
    normalize: bool = True,
) -> float:
    """
    Compute BLEU score (simplified version without brevity penalty).
    
    BLEU measures N-gram precision averaged over multiple N values.
    
    Args:
        hypothesis: Generated text
        reference: Reference (ground truth) text
        max_n: Maximum N-gram size
        normalize: Whether to normalize texts
    
    Returns:
        BLEU score between 0 and 1
    
    Examples:
        >>> compute_bleu_score("hello world", "hello world")
        1.0
        >>> compute_bleu_score("hello there", "hello world")
        0.5  # 50% precision
    """
    if normalize:
        hypothesis = normalize_text(hypothesis)
        reference = normalize_text(reference)
    
    precisions = []
    
    for n in range(1, max_n + 1):
        hyp_ngrams = extract_ngrams(hypothesis, n, normalize=False)
        ref_ngrams = extract_ngrams(reference, n, normalize=False)
        
        if not hyp_ngrams:
            precisions.append(0.0)
            continue
        
        hyp_counts = Counter(hyp_ngrams)
        ref_counts = Counter(ref_ngrams)
        
        # Clipped counts
        clipped = 0
        total = 0
        
        for ngram, count in hyp_counts.items():
            clipped += min(count, ref_counts[ngram])
            total += count
        
        precision = clipped / total if total > 0 else 0.0
        precisions.append(precision)
    
    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        return 0.0
    
    return np.exp(np.mean(np.log(precisions)))


@dataclass
class TextQualityMetrics:
    """Comprehensive text quality metrics."""
    text: str
    length_chars: int
    length_tokens: int
    unique_tokens: int
    lexical_diversity: float  # unique / total tokens
    avg_token_length: float
    ngram_stats: Dict[int, Dict[str, Any]]
    
    @classmethod
    def from_text(
        cls,
        text: str,
        max_n: int = 3,
        normalize: bool = True,
    ) -> 'TextQualityMetrics':
        """
        Compute quality metrics for text.
        
        Args:
            text: Input text
            max_n: Maximum N-gram size to analyze
            normalize: Whether to normalize text
        
        Returns:
            TextQualityMetrics instance
        """
        if normalize:
            text_clean = normalize_text(text)
        else:
            text_clean = text
        
        tokens = text_clean.split()
        
        # Basic stats
        length_chars = len(text_clean)
        length_tokens = len(tokens)
        unique_tokens = len(set(tokens))
        
        lexical_diversity = unique_tokens / length_tokens if length_tokens > 0 else 0.0
        avg_token_length = np.mean([len(t) for t in tokens]) if tokens else 0.0
        
        # N-gram stats
        ngram_stats = {}
        for n in range(1, max_n + 1):
            counter = NGramCounter(n)
            counter.add_text(text, normalize=normalize)
            ngram_stats[n] = {
                "total": counter.total_ngrams,
                "unique": len(counter.counts),
                "diversity": len(counter.counts) / counter.total_ngrams if counter.total_ngrams > 0 else 0.0,
            }
        
        return cls(
            text=text,
            length_chars=length_chars,
            length_tokens=length_tokens,
            unique_tokens=unique_tokens,
            lexical_diversity=lexical_diversity,
            avg_token_length=avg_token_length,
            ngram_stats=ngram_stats,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "length_chars": self.length_chars,
            "length_tokens": self.length_tokens,
            "unique_tokens": self.unique_tokens,
            "lexical_diversity": round(self.lexical_diversity, 3),
            "avg_token_length": round(self.avg_token_length, 2),
            "ngram_stats": self.ngram_stats,
        }


# ============================================================================
# Beam Search Decoding
# ============================================================================

@dataclass(order=True)
class BeamHypothesis:
    """A hypothesis in beam search."""
    score: float = field(compare=True)
    sequence: List[Any] = field(compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)
    
    def __repr__(self) -> str:
        return f"BeamHypothesis(score={self.score:.3f}, len={len(self.sequence)})"


class BeamSearchDecoder:
    """
    Beam search decoder for sequence generation.
    
    Maintains top-k hypotheses at each step, expanding the most promising ones.
    """
    
    def __init__(
        self,
        beam_width: int = 5,
        max_length: int = 100,
        min_length: int = 1,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
    ):
        """
        Initialize beam search decoder.
        
        Args:
            beam_width: Number of beams to maintain
            max_length: Maximum sequence length
            min_length: Minimum sequence length
            length_penalty: Length normalization (> 1 favors longer sequences)
            early_stopping: Stop when beam_width complete sequences found
        """
        if beam_width < 1:
            raise ValueError("beam_width must be positive")
        if max_length < min_length:
            raise ValueError("max_length must be >= min_length")
        
        self.beam_width = beam_width
        self.max_length = max_length
        self.min_length = min_length
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        
        self.reset()
    
    def reset(self):
        """Reset decoder state."""
        self.beams: List[BeamHypothesis] = []
        self.finished: List[BeamHypothesis] = []
        self.step_count = 0
    
    def normalize_score(self, score: float, length: int) -> float:
        """
        Apply length normalization to score.
        
        Args:
            score: Log probability score
            length: Sequence length
        
        Returns:
            Normalized score
        """
        if self.length_penalty == 1.0:
            return score
        
        # Length penalty from Wu et al. (2016)
        # https://arxiv.org/abs/1609.08144
        return score / (length ** self.length_penalty)
    
    def step(
        self,
        expand_fn: Callable[[BeamHypothesis], List[Tuple[Any, float]]],
        is_complete_fn: Optional[Callable[[BeamHypothesis], bool]] = None,
    ) -> bool:
        """
        Perform one beam search step.
        
        Args:
            expand_fn: Function that takes a hypothesis and returns
                       list of (next_token, log_prob) tuples
            is_complete_fn: Optional function to check if hypothesis is complete
        
        Returns:
            True if search should continue, False if done
        """
        self.step_count += 1
        
        # Initialize with empty hypothesis
        if not self.beams and not self.finished:
            self.beams = [BeamHypothesis(score=0.0, sequence=[])]
        
        # Check early stopping
        if (
            self.early_stopping
            and len(self.finished) >= self.beam_width
        ):
            return False
        
        # Check max length
        if self.step_count >= self.max_length:
            # Move remaining beams to finished
            self.finished.extend(self.beams)
            self.beams = []
            return False
        
        # Expand all current beams
        candidates: List[BeamHypothesis] = []
        
        for beam in self.beams:
            # Check if already complete
            if is_complete_fn and is_complete_fn(beam):
                if len(beam.sequence) >= self.min_length:
                    self.finished.append(beam)
                continue
            
            # Expand beam
            try:
                expansions = expand_fn(beam)
            except Exception as e:
                log.warning(f"Expansion failed for beam: {e}")
                continue
            
            for token, log_prob in expansions:
                new_sequence = beam.sequence + [token]
                new_score = beam.score + log_prob
                
                candidates.append(BeamHypothesis(
                    score=new_score,
                    sequence=new_sequence,
                    metadata={"log_prob_last": log_prob},
                ))
        
        # No candidates, we're done
        if not candidates:
            return False
        
        # Select top beam_width candidates
        # Use negative score for max-heap behavior
        candidates.sort(
            key=lambda h: -self.normalize_score(h.score, len(h.sequence))
        )
        
        self.beams = candidates[:self.beam_width]
        
        return True
    
    def decode(
        self,
        expand_fn: Callable[[BeamHypothesis], List[Tuple[Any, float]]],
        is_complete_fn: Optional[Callable[[BeamHypothesis], bool]] = None,
        max_steps: Optional[int] = None,
    ) -> List[BeamHypothesis]:
        """
        Run complete beam search.
        
        Args:
            expand_fn: Expansion function
            is_complete_fn: Completion check function
            max_steps: Maximum number of steps (defaults to max_length)
        
        Returns:
            List of completed hypotheses, sorted by score
        """
        self.reset()
        
        max_steps = max_steps or self.max_length
        
        for _ in range(max_steps):
            should_continue = self.step(expand_fn, is_complete_fn)
            if not should_continue:
                break
        
        # Collect all hypotheses
        all_hyps = self.finished + self.beams
        
        # Sort by normalized score
        all_hyps.sort(
            key=lambda h: -self.normalize_score(h.score, len(h.sequence))
        )
        
        return all_hyps[:self.beam_width]
    
    def get_best(self) -> Optional[BeamHypothesis]:
        """Get best hypothesis."""
        all_hyps = self.finished + self.beams
        
        if not all_hyps:
            return None
        
        return max(
            all_hyps,
            key=lambda h: self.normalize_score(h.score, len(h.sequence))
        )


def beam_search_text(
    start_tokens: List[str],
    next_token_fn: Callable[[List[str]], List[Tuple[str, float]]],
    beam_width: int = 5,
    max_length: int = 50,
    end_tokens: Set[str] = None,
) -> List[Tuple[List[str], float]]:
    """
    Beam search for text generation.
    
    Args:
        start_tokens: Initial tokens
        next_token_
        next_token_fn: Function that takes token sequence and returns
                   list of (next_token, log_prob) tuples
    beam_width: Number of beams
    max_length: Maximum sequence length
    end_tokens: Set of tokens that mark sequence end
    
    Returns:
        List of (token_sequence, score) tuples
    
    Examples:
        >>> def next_token_fn(seq):
        ...     # Mock function returning top 3 candidates
        ...     if not seq:
        ...         return [("hello", -0.1), ("hi", -0.5), ("hey", -1.0)]
        ...     return [("world", -0.2), ("there", -0.4), (".", -0.8)]
        >>> results = beam_search_text([], next_token_fn, beam_width=2, max_length=2)
        >>> len(results)
        2
    """
    end_tokens = end_tokens or {"<eos>", "</s>", ".", "!", "?"}
    
    decoder = BeamSearchDecoder(
        beam_width=beam_width,
        max_length=max_length,
        length_penalty=1.0,
    )
    
    def expand_fn(hyp: BeamHypothesis) -> List[Tuple[str, float]]:
        """Expand hypothesis with next tokens."""
        current_seq = start_tokens + hyp.sequence
        return next_token_fn(current_seq)
    
    def is_complete_fn(hyp: BeamHypothesis) -> bool:
        """Check if sequence is complete."""
        if not hyp.sequence:
            return False
        return hyp.sequence[-1] in end_tokens
    
    # Run beam search
    results = decoder.decode(expand_fn, is_complete_fn)
    
    # Convert to token sequences with scores
    return [
        (start_tokens + hyp.sequence, hyp.score)
        for hyp in results
    ]


class ConstrainedBeamSearch:
    """
    Beam search with constraints (forced tokens, banned tokens, etc.).
    """
    
    def __init__(
        self,
        beam_width: int = 5,
        max_length: int = 100,
        forced_tokens: Optional[Dict[int, List[Any]]] = None,
        banned_tokens: Optional[Set[Any]] = None,
        min_token_score: float = -float('inf'),
    ):
        """
        Initialize constrained beam search.
        
        Args:
            beam_width: Number of beams
            max_length: Maximum sequence length
            forced_tokens: Dict mapping position -> list of required tokens
            banned_tokens: Set of tokens that cannot be generated
            min_token_score: Minimum score threshold for tokens
        """
        self.decoder = BeamSearchDecoder(beam_width, max_length)
        self.forced_tokens = forced_tokens or {}
        self.banned_tokens = banned_tokens or set()
        self.min_token_score = min_token_score
    
    def filter_candidates(
        self,
        candidates: List[Tuple[Any, float]],
        position: int,
    ) -> List[Tuple[Any, float]]:
        """
        Filter candidates based on constraints.
        
        Args:
            candidates: List of (token, score) tuples
            position: Current position in sequence
        
        Returns:
            Filtered candidate list
        """
        # Check forced tokens
        if position in self.forced_tokens:
            required = set(self.forced_tokens[position])
            candidates = [(t, s) for t, s in candidates if t in required]
        
        # Filter banned tokens
        candidates = [(t, s) for t, s in candidates if t not in self.banned_tokens]
        
        # Filter by minimum score
        candidates = [(t, s) for t, s in candidates if s >= self.min_token_score]
        
        return candidates
    
    def decode(
        self,
        expand_fn: Callable[[BeamHypothesis], List[Tuple[Any, float]]],
        is_complete_fn: Optional[Callable[[BeamHypothesis], bool]] = None,
    ) -> List[BeamHypothesis]:
        """
        Run constrained beam search.
        
        Args:
            expand_fn: Expansion function
            is_complete_fn: Completion check function
        
        Returns:
            List of completed hypotheses
        """
        def constrained_expand_fn(hyp: BeamHypothesis) -> List[Tuple[Any, float]]:
            candidates = expand_fn(hyp)
            position = len(hyp.sequence)
            return self.filter_candidates(candidates, position)
        
        return self.decoder.decode(constrained_expand_fn, is_complete_fn)


# ============================================================================
# Voting Algorithms (from previous version)
# ============================================================================

class TieStrategy(Enum):
    """Strategies for breaking ties in voting."""
    FIRST = "first"
    HIGHEST_WEIGHT = "highest_weight"
    LONGEST = "longest"
    RANDOM = "random"


@dataclass
class VotingResult:
    """Result of a voting operation with metadata."""
    winner: Optional[str]
    scores: Dict[str, float]
    total_votes: int
    tied: bool = False
    confidence: float = field(init=False)
    
    def __post_init__(self):
        self.confidence = self._compute_confidence()
    
    def _compute_confidence(self) -> float:
        """Compute confidence based on score distribution."""
        if not self.scores or not self.winner:
            return 0.0
        
        sorted_scores = sorted(self.scores.values(), reverse=True)
        
        if len(sorted_scores) == 1:
            return 1.0
        
        top_score = sorted_scores[0]
        second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0
        
        if top_score == 0:
            return 0.0
        
        return (top_score - second_score) / top_score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "winner": self.winner,
            "confidence": round(self.confidence, 3),
            "tied": self.tied,
            "total_votes": self.total_votes,
            "num_candidates": len(self.scores),
            "top_3_scores": dict(
                sorted(self.scores.items(), key=lambda x: x[1], reverse=True)[:3]
            ),
        }


def majority_vote(
    items: List[str],
    weights: Optional[List[float]] = None,
    min_confidence: float = 0.0,
    min_length: int = VotingConfig.MIN_CHARS,
    tie_strategy: TieStrategy = TieStrategy.FIRST,
    normalize: bool = True,
) -> VotingResult:
    """Robust majority voting with weighted support."""
    if not items:
        return VotingResult(None, {}, 0)
    
    if weights is not None and len(weights) != len(items):
        raise ValueError(
            f"Items ({len(items)}) and weights ({len(weights)}) must have same length"
        )
    
    if weights is None:
        weights = [1.0] * len(items)
    
    scores: Dict[str, float] = defaultdict(float)
    item_max_weight: Dict[str, float] = defaultdict(float)
    original_texts: Dict[str, str] = {}
    
    total_votes = 0
    
    for item, weight in zip(items, weights):
        if weight < min_confidence:
            continue
        
        norm_item = normalize_text(item) if normalize else item.strip()
        
        if len(norm_item) < min_length:
            log.debug(f"Filtered short item: '{norm_item}' (len={len(norm_item)})")
            continue
        
        scores[norm_item] += weight
        item_max_weight[norm_item] = max(item_max_weight[norm_item], weight)
        
        if norm_item not in original_texts:
            original_texts[norm_item] = item.strip()
        
        total_votes += 1
    
    if not scores:
        log.warning("No valid votes after filtering")
        return VotingResult(None, {}, total_votes)
    
    max_score = max(scores.values())
    candidates = [item for item, score in scores.items() if score == max_score]
    
    if len(candidates) == 1:
        winner_norm = candidates[0]
        winner = original_texts[winner_norm]
        log.debug(f"Clear winner: '{winner}' (score={max_score:.2f})")
        return VotingResult(winner, dict(scores), total_votes, tied=False)
    
    log.debug(f"Tie between {len(candidates)} candidates, using {tie_strategy.value}")
    
    if tie_strategy == TieStrategy.HIGHEST_WEIGHT:
        winner_norm = max(candidates, key=lambda x: item_max_weight[x])
    elif tie_strategy == TieStrategy.LONGEST:
        winner_norm = max(candidates, key=len)
    elif tie_strategy == TieStrategy.RANDOM:
        winner_norm = np.random.choice(candidates)
    else:  # TieStrategy.FIRST
        for item in items:
            norm = normalize_text(item) if normalize else item.strip()
            if norm in candidates:
                winner_norm = norm
                break
        else:
            winner_norm = candidates[0]
    
    winner = original_texts[winner_norm]
    log.debug(f"Tie broken: '{winner}' (score={max_score:.2f})")
    
    return VotingResult(winner, dict(scores), total_votes, tied=True)


def weighted_vote(items: List[str], weights: List[float], **kwargs) -> Optional[str]:
    """Weighted majority vote (convenience wrapper)."""
    result = majority_vote(items, weights=weights, **kwargs)
    return result.winner


# ============================================================================
# Array Operations (from previous version)
# ============================================================================

def normalize_array(
    arr: np.ndarray,
    min_val: float = 0.0,
    max_val: float = 1.0,
    clip: bool = False,
) -> np.ndarray:
    """Min-max normalize array to specified range."""
    if min_val >= max_val:
        raise ValueError(f"min_val ({min_val}) must be < max_val ({max_val})")
    
    arr = np.asarray(arr, dtype=np.float32)
    
    if arr.size == 0:
        log.debug("Empty array provided to normalize_array")
        return np.array([], dtype=np.float32)
    
    arr_min = arr.min()
    arr_max = arr.max()
    
    if arr_max - arr_min == 0:
        log.debug("Array has zero variance, returning array filled with min_val")
        return np.full_like(arr, min_val, dtype=np.float32)
    
    normalized = (arr - arr_min) / (arr_max - arr_min)
    normalized = normalized * (max_val - min_val) + min_val
    
    if clip:
        normalized = np.clip(normalized, min_val, max_val)
    
    return normalized


def smooth_array(
    arr: np.ndarray,
    window_size: int = 3,
    method: str = "mean",
) -> np.ndarray:
    """Smooth array using moving window."""
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    if window_size < 1:
        raise ValueError("window_size must be positive")
    
    arr = np.asarray(arr, dtype=np.float32)
    
    if len(arr) < window_size:
        log.debug(
            f"Array length ({len(arr)}) < window_size ({window_size}), "
            f"returning original"
        )
        return arr
    
    if method == "mean":
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(arr, kernel, mode='valid')
    else:
        raise ValueError(f"Unknown method: {method}")
    
    pad_left = window_size // 2
    pad_right = len(arr) - len(smoothed) - pad_left
    smoothed = np.pad(smoothed, (pad_left, pad_right), mode='edge')
    
    return smoothed


def chunk_list(
    lst: List[Any],
    chunk_size: int,
    drop_last: bool = False,
) -> Generator[List[Any], None, None]:
    """Split list into chunks of given size."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    
    for i in range(0, len(lst), chunk_size):
        chunk = lst[i:i + chunk_size]
        
        if drop_last and len(chunk) < chunk_size:
            continue
        
        yield chunk


# ============================================================================
# Backward Compatibility Aliases
# ============================================================================

_normalize_text = normalize_text
dedup_text = dedupe_tokens
smooth_predictions = smooth_array
weighted_majority = weighted_vote
weighted_majority_vote = weighted_vote


# ============================================================================
# Comprehensive Testing & Examples
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ASR UTILITIES - COMPREHENSIVE DEMO")
    print("=" * 80)
    
    # ========================================================================
    # 1. Text Normalization
    # ========================================================================
    print("\n" + "=" * 80)
    print("1. TEXT NORMALIZATION")
    print("=" * 80)
    
    text = "Hello,  World! World!  The the answer is 42."
    print(f"Original: '{text}'")
    print(f"Normalized: '{normalize_text(text)}'")
    print(f"Deduped: '{dedupe_tokens(text)}'")
    
    # ========================================================================
    # 2. Levenshtein Distance & Fuzzy Matching
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. LEVENSHTEIN DISTANCE & FUZZY MATCHING")
    print("=" * 80)
    
    s1, s2 = "kitten", "sitting"
    dist = levenshtein_distance(s1, s2)
    sim = normalized_levenshtein(s1, s2)
    print(f"Distance('{s1}', '{s2}'): {dist}")
    print(f"Similarity: {sim:.3f}")
    
    query = "hello world"
    candidates = ["hello world", "helo world", "goodbye world", "hello there"]
    print(f"\nFuzzy matching '{query}' against candidates:")
    
    matches = fuzzy_match(query, candidates, threshold=0.7, return_scores=True)
    for text, score in matches:
        print(f"  {score:.3f} - {text}")
    
    # Advanced fuzzy matching
    result = fuzzy_match_advanced(query, candidates, threshold=0.7)
    print(f"\nBest match: {result.best_match} (score={result.best_score:.3f})")
    
    # ========================================================================
    # 3. N-gram Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. N-GRAM ANALYSIS")
    print("=" * 80)
    
    text = "hello world foo bar hello world"
    
    # Extract N-grams
    bigrams = extract_ngrams(text, n=2)
    print(f"Bigrams: {bigrams}")
    
    # N-gram counter
    counter = NGramCounter(n=2)
    counter.add_text(text)
    print(f"\nN-gram counts:")
    for ngram, count in counter.most_common(5):
        print(f"  {' '.join(ngram)}: {count}")
    
    # N-gram overlap
    text1 = "hello world foo"
    text2 = "hello world bar"
    overlap = ngram_overlap(text1, text2, n=2)
    print(f"\nBigram overlap('{text1}', '{text2}'): {overlap:.3f}")
    
    # BLEU score
    hyp = "the cat sat on the mat"
    ref = "the cat is on the mat"
    bleu = compute_bleu_score(hyp, ref, max_n=4)
    print(f"\nBLEU score:")
    print(f"  Hypothesis: {hyp}")
    print(f"  Reference: {ref}")
    print(f"  Score: {bleu:.3f}")
    
    # Text quality metrics
    metrics = TextQualityMetrics.from_text(text)
    print(f"\nText quality metrics:")
    print(f"  Tokens: {metrics.length_tokens}")
    print(f"  Unique tokens: {metrics.unique_tokens}")
    print(f"  Lexical diversity: {metrics.lexical_diversity:.3f}")
    
    # ========================================================================
    # 4. Beam Search
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. BEAM SEARCH DECODING")
    print("=" * 80)
    
    # Mock next-token function
    def mock_next_token_fn(seq: List[str]) -> List[Tuple[str, float]]:
        """Mock function for demonstration."""
        if not seq:
            return [("hello", -0.1), ("hi", -0.5), ("hey", -1.0)]
        elif seq[-1] in ["hello", "hi", "hey"]:
            return [("world", -0.2), ("there", -0.4), ("friend", -0.8)]
        else:
            return [(".", -0.1), ("!", -0.5)]
    
    results = beam_search_text(
        start_tokens=[],
        next_token_fn=mock_next_token_fn,
        beam_width=3,
        max_length=3,
    )
    
    print("Beam search results:")
    for i, (seq, score) in enumerate(results, 1):
        print(f"  {i}. {' '.join(seq)} (score={score:.3f})")
    
    # ========================================================================
    # 5. Voting Algorithms
    # ========================================================================
    print("\n" + "=" * 80)
    print("5. VOTING ALGORITHMS")
    print("=" * 80)
    
    items = ["hello world", "hello world", "goodbye world", "hello world"]
    weights = [0.9, 0.8, 0.7, 0.85]
    
    result = majority_vote(items, weights)
    print(f"Items: {items}")
    print(f"Weights: {weights}")
    print(f"Winner: {result.winner}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Scores: {result.scores}")
    
    # ========================================================================
    # 6. Integration Example: ASR Hypothesis Selection
    # ========================================================================
    print("\n" + "=" * 80)
    print("6. INTEGRATION EXAMPLE: ASR HYPOTHESIS SELECTION")
    print("=" * 80)
    
    # Simulate multiple ASR hypotheses with confidences
    hypotheses = [
        "recognize speech",
        "recognize speach",  # typo
        "recognize speech",
        "recognise speech",  # British spelling
    ]
    confidences = [0.9, 0.6, 0.85, 0.7]
    
    print("ASR Hypotheses:")
    for hyp, conf in zip(hypotheses, confidences):
        print(f"  {conf:.2f} - {hyp}")
    
    # Method 1: Weighted voting
    vote_result = majority_vote(hypotheses, confidences, min_confidence=0.5)
    print(f"\nWeighted voting winner: {vote_result.winner}")
    print(f"Confidence: {vote_result.confidence:.3f}")
    
    # Method 2: Fuzzy matching to cluster similar hypotheses
    print(f"\nFuzzy matching analysis:")
    for i, hyp in enumerate(hypotheses):
        similar = fuzzy_match(hyp, hypotheses, threshold=0.8, return_scores=True)
        print(f"  '{hyp}' has {len(similar)} similar matches")
    
    # Method 3: N-gram quality scoring
    print(f"\nN-gram quality scores:")
    for hyp in set(hypotheses):
        quality = TextQualityMetrics.from_text(hyp, max_n=2)
        print(f"  '{hyp}': diversity={quality.lexical_diversity:.3f}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
