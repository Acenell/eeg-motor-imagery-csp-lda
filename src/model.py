from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

def csp_lda_classifier(n_components=6):
    """
    Creates and return a Sklearn Pipeline object with csp and lda

    Parameters
    ----------
    n_components : int
        Used by common spatial patterns to specify how many virtual
        channels to create. Default as 6 to find patterns without
        overfitting.

    Returns
    -------
    clf : sklearn.pipeline.Pipeline
        Pipeline containing CSP for feature extraction and LDA
        for classification.
    """
    # Common spatial with n_components for feature extraction
    csp = CSP(
    n_components=n_components,
    reg=None,
    log=True,
    norm_trace=False
    )

    # LDA for classification
    lda = LinearDiscriminantAnalysis()

    return Pipeline([
        ("csp", csp),
        ("lda", lda)
    ])
