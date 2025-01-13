import warnings

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from torch import nn


def top_k_accuracy(y_true, y_pred, k=5):
    """
    Calculate top-k accuracy for multi-label classification.

    Parameters:
    y_true (np.array): Binary matrix of ground truth labels (shape: num_samples x num_classes)
    y_pred (np.array): Predicted scores/probabilities for each class (shape: num_samples x num_classes)
    k (int): Number of top elements to consider for evaluation

    Returns:
    accuracy (float): Top-k accuracy
    """
    num_samples = y_true.shape[0]
    correct_count = 0

    for i in range(num_samples):
        true_labels = np.where(y_true[i] == 1)[0]
        pred_labels = np.argsort(y_pred[i])[-k:]  # Get top-k predicted labels

        # Check if any true label is in the top-k predicted labels
        if np.intersect1d(pred_labels, true_labels).size > 0:
            correct_count += 1

    accuracy = correct_count / num_samples

    return accuracy


def top_k_precision(y_true, y_pred, k=5):
    num_samples = y_true.shape[0]
    precision_sum = 0

    for i in range(num_samples):
        true_labels = np.where(y_true[i] == 1)[0]
        pred_labels = np.argsort(y_pred[i])[-k:]  # Get top-k predicted labels

        true_positives = np.intersect1d(pred_labels, true_labels).size
        precision_sum += true_positives / k

    precision = precision_sum / num_samples
    return precision


def top_k_recall(y_true, y_pred, k=5):
    num_samples = y_true.shape[0]
    recall_sum = 0

    for i in range(num_samples):
        true_labels = np.where(y_true[i] == 1)[0]
        pred_labels = np.argsort(y_pred[i])[-k:]  # Get top-k predicted labels

        true_positives = np.intersect1d(pred_labels, true_labels).size
        recall_sum += true_positives / true_labels.size

    recall = recall_sum / num_samples
    return recall


def top_k_f1_score(y_true, y_pred, k=5):
    precision = top_k_precision(y_true, y_pred, k)
    recall = top_k_recall(y_true, y_pred, k)
    if precision + recall == 0:
        return 0.0
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def Metrics(y_true: np.ndarray, y_scores: np.ndarray):
    y_pred = y_scores >= 0.5
    acc = np.zeros(y_pred.shape[-1])

    for i in range(y_pred.shape[-1]):
        acc[i] = accuracy_score(y_true[:, i], y_pred[:, i])

    return acc.tolist(), np.mean(acc).tolist()


def AUC(y_true: np.ndarray, y_pred: np.ndarray, verbose: bool = False) -> float:
    aucs = []
    assert (
            len(y_true.shape) == 2 and len(y_pred.shape) == 2
    ), "Predictions and labels must be 2D."
    for col in range(y_true.shape[1]):
        try:
            aucs.append(roc_auc_score(y_true[:, col], y_pred[:, col]))
        except ValueError as e:
            if verbose:
                print(
                    f"Value error encountered for label {col}, likely due to using mixup or "
                    f"lack of full label presence. Setting AUC to accuracy. "
                    f"Original error was: {str(e)}."
                )
            aucs.append((y_pred == y_true).sum() / len(y_pred))
    return np.array(aucs).tolist()


def multi_threshold_precision_recall(y_true: np.ndarray, y_pred: np.ndarray, thresholds: np.ndarray):
    # Expand analysis to number of thresholds
    y_pred_bin = (
            np.repeat(y_pred[None, :, :], len(thresholds), axis=0)
            >= thresholds[:, None, None]
    )
    y_true_bin = np.repeat(y_true[None, :, :], len(thresholds), axis=0)

    # Compute true positives
    TP = np.sum(np.logical_and(y_true, y_pred_bin), axis=2)

    # Compute macro-average precision handling all warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        den = np.sum(y_pred_bin, axis=2)
        precision = TP / den
        precision[den == 0] = np.nan
        # precision[den == 0] = 0
        with warnings.catch_warnings():  # for nan slices
            warnings.simplefilter("ignore", category=RuntimeWarning)
            av_precision = np.nanmean(precision, axis=1)

    # Compute macro-average recall
    recall = TP / np.sum(y_true_bin, axis=2)
    av_recall = np.mean(recall, axis=1)

    return av_precision, av_recall


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray, num_thresholds: int = 10):
    thresholds = np.arange(0.00, 1.01, 1.0 / (num_thresholds - 1), float)
    average_precisions, average_recalls = multi_threshold_precision_recall(
        y_true, y_pred, thresholds
    )
    f_scores = (
            2
            * (average_precisions * average_recalls)
            / (average_precisions + average_recalls)
    )
    auc = np.array(AUC(y_true, y_pred, verbose=True)).mean().tolist()
    return (
        f_scores[np.nanargmax(f_scores).tolist()],
        average_recalls[np.nanargmax(average_recalls).tolist()],
        f_scores.tolist(),
        average_precisions.tolist(),
        average_recalls.tolist(),
        thresholds.tolist(),
    )

#focal loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.sigmoid = nn.Sigmoid()

    def forward(self, y_pred, y_true):
        y_pred = self.sigmoid(y_pred)#之前有sigmoid的话记得注释掉这一句
        fl = - self.alpha * y_true * torch.log(y_pred) * ((1.0 - y_pred) ** self.gamma) - (1.0 - self.alpha) * (1.0 - y_true) * torch.log(1.0 - y_pred) * (y_pred ** self.gamma)
        fl_sum = fl.sum()
        return fl_sum