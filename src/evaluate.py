from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
import numpy as np

def evaluate(clf, data):
    """
    Evaluates performance of a classifier. Within-run evaluation if
    data contains one run/group. Cross-run evaluation if data contains 
    multiple runsgroups.
    
    Parameters
    ----------
    clf : sklearn.pipeline.Pipeline
        Pipeline containing CSP for feature extraction and LDA
        for classification.
    data : dict
        x : np.ndarray
            EEG data of shape (n_epochs, n_channels, n_times).
        y : np.ndarray
            Labels for each epoch of shape (n_epochs).
        groups : np.ndarray
            Numeric run identifier of shape (n_epochs). Differentiate
            epochs between runs

    Returns
    ----------
    mean_acc : float
        Mean accuracy of all scores
    scores : np.ndarray
        Accuracy for each fold
    """

    x, y , groups = data["x"], data["y"], data["groups"]
    n_unique_groups = len(np.unique(groups))

    # Within-run evaluation if train_data contains only one run/group.
    if n_unique_groups <= 1:
        cv = StratifiedKFold(n_splits=5, shuffle=True)
        scores = cross_val_score(clf, x, y, cv=cv)
    else: # Cross-run evaluation if train_data contains multiple runs/groups.
        cv = StratifiedGroupKFold(n_splits=min(n_unique_groups, 5), shuffle=True)
        scores = cross_val_score(clf, x, y, groups=groups, cv=cv)
    
    mean = scores.mean()
    return mean, scores
