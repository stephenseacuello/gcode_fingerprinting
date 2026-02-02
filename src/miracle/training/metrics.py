"""
Metrics for G-code prediction evaluation.

Includes:
- Classification metrics (accuracy, precision, recall, F1)
- Sequence metrics (BLEU, edit distance)
- Per-token metrics
- Confusion matrix utilities
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

__all__ = [
    "compute_classification_metrics",
    "compute_sequence_metrics",
    "compute_bleu",
    "compute_perplexity",
    "compute_confusion_matrix",
    "compute_comprehensive_metrics",
    "MetricsTracker",
]


def compute_classification_metrics(
    predictions: torch.Tensor,  # [B, num_classes] or [B]
    targets: torch.Tensor,      # [B]
    num_classes: Optional[int] = None,
    ignore_index: int = -100,
) -> Dict[str, float]:
    """
    Compute classification metrics (accuracy, precision, recall, F1).

    Args:
        predictions: Model predictions (logits or class indices)
        targets: Ground truth labels
        num_classes: Number of classes (auto-detected if None)
        ignore_index: Label to ignore in metrics

    Returns:
        Dictionary with metrics: accuracy, precision, recall, f1
    """
    # Convert logits to predictions if needed
    if predictions.dim() > 1:
        predictions = torch.argmax(predictions, dim=-1)

    # Filter out ignored indices
    mask = targets != ignore_index
    predictions = predictions[mask]
    targets = targets[mask]

    if len(targets) == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Accuracy
    correct = (predictions == targets).float()
    accuracy = correct.mean().item()

    # Per-class precision, recall, F1
    if num_classes is None:
        num_classes = max(int(targets.max().item()) + 1, int(predictions.max().item()) + 1)

    precision_scores = []
    recall_scores = []
    f1_scores = []

    for c in range(num_classes):
        # True positives, false positives, false negatives
        tp = ((predictions == c) & (targets == c)).sum().float()
        fp = ((predictions == c) & (targets != c)).sum().float()
        fn = ((predictions != c) & (targets == c)).sum().float()

        # Precision
        precision = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0.0
        precision_scores.append(precision)

        # Recall
        recall = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0
        recall_scores.append(recall)

        # F1
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    # Macro-averaged metrics
    precision = np.mean([p for p in precision_scores if p > 0])
    recall = np.mean([r for r in recall_scores if r > 0])
    f1 = np.mean([f for f in f1_scores if f > 0])

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_perplexity(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = 0) -> float:
    """
    Compute perplexity for language modeling.

    Args:
        logits: Model output logits [B, T, V]
        targets: Target token ids [B, T]
        ignore_index: Token to ignore (usually PAD)

    Returns:
        Perplexity value
    """
    # Flatten
    logits_flat = logits.reshape(-1, logits.size(-1))  # [B*T, V]
    targets_flat = targets.reshape(-1)  # [B*T]

    # Compute cross-entropy loss
    loss = torch.nn.functional.cross_entropy(
        logits_flat, targets_flat,
        ignore_index=ignore_index,
        reduction='mean'
    )

    # Perplexity = exp(loss)
    perplexity = torch.exp(loss).item()

    return perplexity


def compute_sequence_metrics(
    predicted_sequences: List[List[int]],
    target_sequences: List[List[int]],
    pad_token: int = 0,
) -> Dict[str, float]:
    """
    Compute sequence-level metrics.

    Args:
        predicted_sequences: List of predicted token sequences
        target_sequences: List of target token sequences
        pad_token: Padding token to ignore

    Returns:
        Dictionary with: exact_match, edit_distance, token_accuracy
    """
    exact_matches = 0
    total_edit_dist = 0
    total_tokens = 0
    correct_tokens = 0

    for pred_seq, target_seq in zip(predicted_sequences, target_sequences):
        # Remove padding
        pred_seq = [t for t in pred_seq if t != pad_token]
        target_seq = [t for t in target_seq if t != pad_token]

        # Exact match
        if pred_seq == target_seq:
            exact_matches += 1

        # Edit distance (Levenshtein)
        edit_dist = levenshtein_distance(pred_seq, target_seq)
        total_edit_dist += edit_dist

        # Token-level accuracy
        for p, t in zip(pred_seq, target_seq):
            total_tokens += 1
            if p == t:
                correct_tokens += 1

        # Account for length difference
        total_tokens += abs(len(pred_seq) - len(target_seq))

    n_sequences = len(predicted_sequences)

    return {
        "exact_match": exact_matches / n_sequences if n_sequences > 0 else 0.0,
        "avg_edit_distance": total_edit_dist / n_sequences if n_sequences > 0 else 0.0,
        "token_accuracy": correct_tokens / total_tokens if total_tokens > 0 else 0.0,
    }


def levenshtein_distance(seq1: List[int], seq2: List[int]) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    m, n = len(seq1), len(seq2)

    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1],  # substitution
                )

    return dp[m][n]


def compute_bleu(
    predicted_sequences: List[List[int]],
    target_sequences: List[List[int]],
    max_n: int = 4,
    pad_token: int = 0,
) -> Dict[str, float]:
    """
    Compute corpus-level BLEU score (BLEU-1 through BLEU-4).

    Args:
        predicted_sequences: List of predicted token sequences
        target_sequences: List of target token sequences
        max_n: Maximum n-gram order (default 4)
        pad_token: Padding token to strip

    Returns:
        Dictionary with bleu_1, bleu_2, bleu_3, bleu_4, bleu (geometric mean)
    """
    from collections import Counter
    import math

    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    bp_ref_len = 0
    bp_hyp_len = 0

    for pred_seq, ref_seq in zip(predicted_sequences, target_sequences):
        pred_seq = [t for t in pred_seq if t != pad_token]
        ref_seq = [t for t in ref_seq if t != pad_token]

        bp_hyp_len += len(pred_seq)
        bp_ref_len += len(ref_seq)

        for n in range(1, max_n + 1):
            pred_ngrams = Counter(
                tuple(pred_seq[i:i + n]) for i in range(len(pred_seq) - n + 1)
            )
            ref_ngrams = Counter(
                tuple(ref_seq[i:i + n]) for i in range(len(ref_seq) - n + 1)
            )
            clipped = sum(
                min(count, ref_ngrams[ng]) for ng, count in pred_ngrams.items()
            )
            total = sum(pred_ngrams.values())
            clipped_counts[n - 1] += clipped
            total_counts[n - 1] += total

    # Brevity penalty
    if bp_hyp_len == 0:
        bp = 0.0
    elif bp_hyp_len >= bp_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - bp_ref_len / bp_hyp_len)

    # Per-n precisions
    precisions = {}
    log_avg = 0.0
    valid_n = 0
    for n in range(max_n):
        if total_counts[n] > 0 and clipped_counts[n] > 0:
            p = clipped_counts[n] / total_counts[n]
            precisions[f"bleu_{n + 1}"] = p
            log_avg += math.log(p)
            valid_n += 1
        else:
            precisions[f"bleu_{n + 1}"] = 0.0

    if valid_n > 0:
        precisions["bleu"] = bp * math.exp(log_avg / valid_n)
    else:
        precisions["bleu"] = 0.0

    return precisions


def compute_comprehensive_metrics(
    all_predictions: torch.Tensor,
    all_targets: torch.Tensor,
    predicted_sequences: List[List[int]],
    target_sequences: List[List[int]],
    pad_token: int = 0,
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute all metrics at once: accuracy, precision, recall, F1, BLEU, edit distance.

    Args:
        all_predictions: Flat tensor of all predicted token IDs
        all_targets: Flat tensor of all target token IDs
        predicted_sequences: List of predicted token sequences (for sequence-level metrics)
        target_sequences: List of target token sequences
        pad_token: Padding token to ignore
        num_classes: Number of classes for classification metrics

    Returns:
        Dictionary with all metrics
    """
    results = {}

    # Flatten to 1D if needed (callers may pass [B, T] tensors)
    if all_predictions.dim() > 1:
        all_predictions = all_predictions.reshape(-1)
    if all_targets.dim() > 1:
        all_targets = all_targets.reshape(-1)

    # Classification metrics (token-level)
    cls_metrics = compute_classification_metrics(
        all_predictions, all_targets,
        num_classes=num_classes, ignore_index=pad_token,
    )
    results.update(cls_metrics)

    # Sequence metrics (edit distance, exact match)
    seq_metrics = compute_sequence_metrics(
        predicted_sequences, target_sequences, pad_token=pad_token,
    )
    results.update(seq_metrics)

    # BLEU
    bleu_metrics = compute_bleu(
        predicted_sequences, target_sequences, pad_token=pad_token,
    )
    results.update(bleu_metrics)

    return results


def compute_confusion_matrix(
    predictions: torch.Tensor,  # [N]
    targets: torch.Tensor,      # [N]
    num_classes: int,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        predictions: Predicted class indices
        targets: Target class indices
        num_classes: Number of classes

    Returns:
        Confusion matrix [num_classes, num_classes]
    """
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    # Handle both tensor and numpy inputs
    if hasattr(predictions, 'cpu'):
        predictions = predictions.cpu().numpy()
    elif isinstance(predictions, np.ndarray):
        predictions = predictions
    else:
        predictions = np.array(predictions)

    if hasattr(targets, 'cpu'):
        targets = targets.cpu().numpy()
    elif isinstance(targets, np.ndarray):
        targets = targets
    else:
        targets = np.array(targets)

    for pred, target in zip(predictions, targets):
        if 0 <= pred < num_classes and 0 <= target < num_classes:
            confusion[target, pred] += 1

    return confusion


class MetricsTracker:
    """Track and aggregate metrics over training/evaluation."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracked metrics."""
        self.metrics = defaultdict(list)
        self.counts = defaultdict(int)

    def update(self, metrics: Dict[str, float], count: int = 1):
        """Update metrics with new values."""
        for key, value in metrics.items():
            self.metrics[key].append(value)
            self.counts[key] += count

    def compute(self) -> Dict[str, float]:
        """Compute average metrics."""
        avg_metrics = {}
        for key, values in self.metrics.items():
            if len(values) > 0:
                avg_metrics[key] = np.mean(values)
            else:
                avg_metrics[key] = 0.0
        return avg_metrics

    def get_latest(self) -> Dict[str, float]:
        """Get latest metric values."""
        latest = {}
        for key, values in self.metrics.items():
            if len(values) > 0:
                latest[key] = values[-1]
            else:
                latest[key] = 0.0
        return latest

    def __repr__(self) -> str:
        metrics = self.compute()
        return ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])


def compute_top_k_accuracy(
    logits: torch.Tensor,  # [B, num_classes]
    targets: torch.Tensor,  # [B]
    k: int = 5,
) -> float:
    """
    Compute top-k accuracy.

    Args:
        logits: Model predictions
        targets: Ground truth labels
        k: Number of top predictions to consider

    Returns:
        Top-k accuracy
    """
    with torch.no_grad():
        # Get top k predictions
        _, top_k_preds = logits.topk(k, dim=-1)  # [B, k]

        # Check if target is in top k
        targets_expanded = targets.unsqueeze(-1).expand_as(top_k_preds)
        correct = (top_k_preds == targets_expanded).any(dim=-1)

        accuracy = correct.float().mean().item()

    return accuracy
