# Decision Score Histogram
A new histogram-based approach for visualizing anomaly detection algorithm performance and prediction confidence.

## Background
Performance visualization of anomaly detection algorithms is an 
essential aspect of anomaly and intrusion detection systems. 
It allows analysts to highlight trends and outliers in anomaly 
detection models results to gain intuitive understanding of detection
models. This work presents a new way of visualizing anomaly 
detection algorithm results using a histogram. 

### Importance of the Visualization Approach

- provides a better understanding
of detection algorithms performance by revealing the exact 
proportions of true positives, true negatives, false positives, 
and false negatives of detection algorithms. 
- provides insights into the strengths and weaknesses of detection
algorithms performance on different aspects of a datasets unlike 
previous approaches that rely on only positive and negative 
decision scores. 
- can be applied to performance visualization and analysis of 
supervised machine learning techniques involving 
binary classification of imbalanced datasets.


## Usage
Below is an example of how to use this histogram.

**Input:**
- Decision/Anomaly scores of detection or binary 
classification algorithm.
- Ground truth: binary class labels of normal (+1) and anomalous (-1)
data instances.
- **Input type**: ```list, numpy array, pandas DataFrame, pandas Series```

**Output:**
- Histogram visualization of true positives, true negatives, false
positives and false negatives with their prediction confidences.

**General use case**
``` python
from hist_score import AnomalyScoreHist

fig = AnomalyScoreHist()
fig.plot_hist(decision_scores, ground_truth)
```

### Example

```python
from hist_score import AnomalyScoreHist

fig = AnomalyScoreHist()
fig.plot_hist(test_dec, test['class'])
```
where ```test_dec``` is the decision score output of isolation forest 
detection algorithm on test set 2 of [TLIGHT dataset](https://github.com/emmanuelaboah/TLIGHT-SYSTEM/tree/main/Dataset)
and ```test['class']```
is the class label (1, -1).

![hist visualization](images/img1.png)


## References
Users can refer to our papers below for further insight and 
examples:

