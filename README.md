## EEG Motor Classification with CSP and LDA

This project presents a classical EEG motor imagery classification pipeline
using Common Spatial Patterns (CSP) and Linear Discriminant Analysis (LDA) on 
electroencephalogram (EEG) recordings stored in EDF file format with the MNE library

## Overview
Motor imagery classification is a common task in brain-computer-interface (BCI) 
research. This project showcases a signal processing and machine learning 
pipeline using CSP and LDA.

The pipeline includes:
- Bandpass filtering (8-30 Hz)
- Epoch extraction from EDF files
- CSP feature extraction
- LDA classification

## Methods
- Signal Preprocessing: MNE-Python
- Feature Extraction: MNE Common Spatial Patterns
- Classifier: scikit-learn Linear Discriminant Analysis
- Data Format: EDF EEG Recordings

## Project Structure
```text
eeg-motor-imagery-csp-lda/
├── src/
│   ├── data.py
│   ├── models.py
│   └── evaluate.py
├── notebooks/
├── data/
├── results/
└── README.md
```
## Usage
```python
from src.data import get_subject_data
from src.models import csp_lda_classifier
from src.evaluate import csp_lda_fit, csp_lda_predict

#Processes all edf files in subject folder
#get_subject_data defaults may be customized
x_train, y_train = get_subject_data("subject_001_train/")
x_valid, y_valid = get_subject_data("subject_001_test/")

#Creates classifier object
clf = csp_lda_classifier()

#Trains and predicts using given classifier
clf = csp_lda_fit(clf, x_train, y_train)
preds = csp_lda_predict(clf, x_valid, y_valid, verbose=True)
```

## Limitations
- Single-subject evaluation
- CSP and LDA may not reach the accuracy of state-of-the-art deep learning models
- CSP performance may decline when trials for a class is limited
- Performance may vary across subjects depending on BCI literacy
- No real-time BCI testing done. Only offline data used.
