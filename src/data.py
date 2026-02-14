from pathlib import Path
import mne
import numpy as np
mne.set_log_level("ERROR")

Path.DEFAULT_PATH = Path('/home/aen/eegenv')
def get_subject_data(
    path,
    classes,
    offset=0.5,
    duration=3.5,
    fmin=8,
    fmax=30
):
    """
    Load and concatenate EEG epochs from all EDF runs in a subject directory.

    Parameters
    ----------
    path : str or Path
        Directory containing EDF files for one subject.
    classes : tuple of str (length 2)
        Two event labels to use for binary classification Ex: (left, right).
    offset : float
        Time (in seconds) after the event onset where epoch extraction begin.
        Suggest offset be half a second after event onset to account 
        for subject reaction time to the event.
    duration : float
        Duration (in seconds) of each epoch after event onset+offset.
    fmin, fmax : float
        Bandpass filter frequency range (Hz). 8-30 for motor imagery

    Returns
    -------
    data : dict
        x : np.ndarray
            EEG data of shape (n_epochs, n_channels, n_times).
        y : np.ndarray
            Labels for each epoch of shape (n_epochs).
        groups : np.ndarray
            Numeric run identifier of shape (n_epochs). Differentiate
            epochs between runs
    """
    # Find all EDF runs for this subject
    runs = sorted(Path(path).glob("*.edf"))
    if len(runs) == 0:
        raise FileNotFoundError(f"No EDF files found in {path}")

    x_all = []  # List of epoch data arrays across runs
    y_all = []  # List of label arrays across runs
    groups = [] # Numeric run identifier for each epoch across runs
    label_map = {0: classes[0], 1: classes[1]} # Maps numeric labels to class names
    
    # Load and process each run independently
    for run_id, run_path in enumerate(runs):
        x, y = get_epochs(run_path, offset, duration, classes, fmin, fmax)
        x_all.append(x)
        y_all.append(y)
        groups.append(np.full(len(y), run_id))

    # Concatenate epochs across runs and store in a dictionary
    data = dict(
        x = np.concatenate(x_all),
        y = np.concatenate(y_all),
        groups = np.concatenate(groups),
    )

    return data


def get_epochs(
    path,
    classes,
    offset=0.5,
    duration=3.5,
    fmin=8,
    fmax=30
):
    """
    Load a single EDF file and extract bandpass-filtered EEG epochs.

    Parameters
    ----------
    path : str or Path
        Directory containing EDF files for one subject.
    classes : tuple of str (length 2)
        Two event labels to use for binary classification Ex: (left, right).
    offset : float
        Time (in seconds) after the event onset where epoch extraction begin.
        Suggest offset be half a second after event onset to account 
        for subject reaction time to the event.
    duration : float
        Duration (in seconds) of each epoch after event onset+offset.
    fmin, fmax : float
        Bandpass filter frequency range (Hz). 8-30 for motor imagery

    Returns
    -------
    X : np.ndarray
        EEG epochs of shape (n_epochs, n_channels, n_times).
    y : np.ndarray
        Labels for each epoch of shape (n_epochs).
    """
    # Load continuous EEG
    raw = mne.io.read_raw_edf(path, preload=True)

    # Bandpass filter continuous data before epoching
    raw.filter(fmin, fmax, fir_design="firwin", phase="zero")

    # Extract events from EDF annotations
    events, event_id = mne.events_from_annotations(raw)

    # Restrict to the two desired classes
    if classes is not None:
        event_id = {k: v for k, v in event_id.items() if k in classes}

        if not all(c in event_id for c in classes):
            raise ValueError(
                f"classes must be event labels in EDF files. "
                f"Available: {list(event_id.keys())}"
            )

        if len(event_id) != 2:
            raise ValueError(
                f"Exactly two classes required for classification. "
                f"Got {len(classes)}: {classes}"
            )

    # Epoch the data around each event
    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin=offset,
        tmax=offset+duration,
        baseline=None,
        preload=True
    )

    # Extract data and labels
    x = epochs.get_data()
    y = (epochs.events[..., -1] == event_id[classes[1]]).astype(int)
    y = np.array([classes[idx] for idx in y])
    
    return x, y
