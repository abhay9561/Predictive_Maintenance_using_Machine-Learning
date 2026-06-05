# Technical Memo

## Evaluation Metric Selection

This project addresses a highly imbalanced predictive maintenance problem where only 0.5% of observations represent machine failures. In such scenarios, Accuracy is not an appropriate evaluation metric because a model predicting all observations as non-failures would still achieve approximately 99.5% Accuracy.

Therefore, Precision-Recall AUC (PR-AUC) was selected as the primary evaluation metric. PR-AUC focuses directly on the minority class and measures how effectively the model identifies machine failures while balancing Precision and Recall.

Although ROC-AUC was also evaluated, it can appear overly optimistic in highly imbalanced datasets because it considers performance across both classes. The Random Forest model achieved a PR-AUC of 0.6224 and ROC-AUC of 0.9932, outperforming Logistic Regression and demonstrating superior failure detection capability.

## Cost-Sensitive Threshold Adjustment

The business requirement states that the cost of a False Negative is 100 times higher than the cost of a False Positive. To reduce the likelihood of missing machine failures, the decision threshold can be adjusted using:

Threshold = Cost(FP) / (Cost(FP) + Cost(FN))

For Cost(FP)=1 and Cost(FN)=100:

Threshold = 1 / (1 + 100) = 0.0099

Python implementation:

fn_cost = 100

fp_cost = 1

threshold = fp_cost / (fp_cost + fn_cost)

predictions = (probabilities >= threshold).astype(int)

This threshold prioritizes failure detection by increasing Recall and minimizing costly missed failures.
