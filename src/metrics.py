"""
STEP 5 - MODEL METRICS
======================
Everything is computed from P(Parasitized) and a decision threshold
(the LED turns on when P >= threshold).

Metrics
-------
accuracy      : fraction correct
sensitivity   : infected cells caught            TP / (TP + FN)   <- most important
specificity   : healthy cells correctly cleared  TN / (TN + FP)
precision     : LED-on events that are truly infected
f1            : balance of precision and sensitivity
auroc         : threshold-free ranking quality
ece           : expected calibration error - does a 90% confidence mean 90% correct?
brier         : mean squared error of the probabilities
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(labels, prob_pos, n_bins: int = 15) -> float:
    """ECE on the confidence of the predicted class (standard definition)."""
    labels = np.asarray(labels)
    prob_pos = np.asarray(prob_pos)
    pred = (prob_pos >= 0.5).astype(int)
    conf = np.where(pred == 1, prob_pos, 1 - prob_pos)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def compute_metrics(labels, prob_pos, threshold: float = 0.5) -> dict:
    labels = np.asarray(labels).astype(int)
    prob_pos = np.asarray(prob_pos, dtype=np.float64)
    pred = (prob_pos >= threshold).astype(int)

    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-12)
    try:
        auroc = roc_auc_score(labels, prob_pos)
    except ValueError:      # only one class present
        auroc = float("nan")

    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / len(labels),
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "auroc": float(auroc),
        "ece": expected_calibration_error(labels, prob_pos),
        "brier": float(np.mean((prob_pos - labels) ** 2)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def threshold_for_sensitivity(labels, prob_pos, target: float) -> float:
    """
    Highest threshold whose sensitivity is still >= target.
    (Highest = fewest false alarms while catching enough infections.)
    """
    labels = np.asarray(labels).astype(int)
    pos_scores = np.sort(np.asarray(prob_pos)[labels == 1])
    if len(pos_scores) == 0:
        return 0.5
    # allow at most floor((1 - target) * n_pos) positives below the threshold
    k = int(np.floor((1 - target) * len(pos_scores)))
    return float(pos_scores[k])


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    """
    Temperature scaling: find T > 0 that minimizes NLL of softmax(logits / T)
    on the validation set. One number; does not change which class wins.
    """
    x = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(x / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())
