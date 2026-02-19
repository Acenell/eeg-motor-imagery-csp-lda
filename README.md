## EEG Motor Classification with CSP and LDA

This project presents a classical EEG motor imagery classification pipeline
using Common Spatial Patterns (CSP) and Linear Discriminant Analysis (LDA) on 
electroencephalogram (EEG) recordings stored in EDF file format with the MNE library

# Overview
Motor imagery classification is a common task in brain-computer-interface (BCI) 
research. This project showcases a signal processing and machine learning 
pipeline using CSP and LDA.

The pipeline includes:
- Bandpass filtering (8-30 Hz)
- Epoch extraction from EDF files
- CSP feature extraction
- LDA classification

# Methods
- Signal Preprocessing: MNE-Python
- Feature Extraction: MNE Common Spatial Patterns
- Classifier: scikit-learn Linear Discriminant Analysis
- Data Format: EDF EEG Recordings

# Project Structure
```text
eeg-motor-imagery-csp-lda/
├── src/
│   ├── data.py
│   ├── evaluate.py
│   └── model.py
├── notebooks/
│   └── BCI_Literacy_Test.ipynb
├── data/
│   └── Credits_and_Data.txt
├── results/
│   └── BCI_Literacy_Test_Results.txt
├── README.md
└── requirements.txt
```

# Limitations
- Single-subject evaluation
- CSP and LDA may not reach the accuracy of state-of-the-art deep learning models
- CSP performance may decline when trials for a class is limited
- Performance may vary across subjects depending on BCI literacy
- No real-time BCI testing done. Only offline data used.
