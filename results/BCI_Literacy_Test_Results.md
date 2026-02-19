## Results

This EEG classification pipeline using CSP and LDA was evaluated using the 2009
EEG Motor Movement/Imagery Dataset from PhysioNet (see in data/Credits_and_Data) 
using only the motor imagery trials to classify subject-wise left vs. right hand
motor imagery.

# Performance
- Total subjects: 109
- Average accuracy using CSP+LDA (within subject): 60.3%
- BCI-literate threshold >= 70%
- Number of BCI-literate subjects: 27/109 (24.8%)


# Interpretation
Results show that accuracies vary heavily across different subject which is 
to be expected due to the well-known concept of BCI-literacy, the ability for 
an individual to effectively use BCI systems. With CSP + LDA, most people 
achieve ~60% accuracy without training and 70% can be achieved with training.