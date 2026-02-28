## Results

This real-time EEG classification pipeline using CSP and LDA was evaluated using 
the 2009 EEG Motor Movement/Imagery Dataset from PhysioNet (see in data/Credits_and_Data) 
using only the motor imagery trials to classify subject-wise left vs. right hand
motor imagery in a simulated real-time scenario with variating window and step 
lengths for different tests

# Performance
- Total subjects: 109
- Average accuracy using window size 3, step size 0.75 (seconds): 60.5%
- Average accuracy using window size 2, step size 0.5 (seconds): 58.1%
- Average accuracy using window size 1, step size 0.25 (seconds): 56.4%


# Interpretation
As shown above, accuracies decrease as window size and step size is decreased.
This is expected as lower window sizes means less temporal data to identify strong 
features by common spatial patterns.

Longer window and step sizes will increase stability and temporal feature strength, 
but increases latency.