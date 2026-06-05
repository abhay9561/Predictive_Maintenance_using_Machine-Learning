# Predictive Maintenance using Machine Learning

## Project Overview

This project develops a predictive maintenance model for a manufacturing environment using highly imbalanced sensor data. The objective is to identify potential machine failures before they occur.

The dataset is synthetically generated using Scikit-Learn's `make_classification()` and contains 100,000 sensor readings. Only 0.5% of observations represent machine failures, making this a highly imbalanced classification problem.

---
## Video Walkthrough

Watch the project walkthrough here:
https://drive.google.com/file/d/1hcS7-cU3oqNF-ivIwLNuidk3u-jNxdZX/view?usp=sharing

## Problem Statement

Build an end-to-end machine learning pipeline that:

* Handles highly imbalanced data
* Performs feature engineering
* Applies outlier treatment
* Trains multiple classification models
* Tunes model hyperparameters
* Evaluates models using imbalance-aware metrics
* Implements business-aware threshold optimization

---

## Dataset Characteristics

* Total Samples: 100,000
* Features: 20 Sensor Features
* Engineered Features: 2
* Target Variable: Machine Failure
* Failure Rate: ~0.5%

---

## Project Workflow

1. Synthetic Data Generation
2. Exploratory Data Analysis
3. Feature Engineering
4. Train-Test Split
5. Outlier Treatment
6. Feature Scaling
7. Class Imbalance Handling using SMOTE
8. Logistic Regression Pipeline
9. Random Forest Pipeline
10. Hyperparameter Tuning using GridSearchCV
11. Model Evaluation
12. Threshold Optimization
13. Model Persistence

---

## Models Evaluated

### Logistic Regression

* Precision: 0.0217
* Recall: 0.81
* F1 Score: 0.0422
* ROC-AUC: 0.8980
* PR-AUC: 0.1179

### Random Forest (Selected Model)

* Precision: 0.7321
* Recall: 0.41
* F1 Score: 0.5256
* ROC-AUC: 0.9932
* PR-AUC: 0.6224

---

## Best Model Configuration

RandomForestClassifier

* n_estimators = 200
* max_depth = None
* class_weight = balanced

Best Cross Validation Score: 0.5557

---

## Business Threshold Optimization

Business Requirement:

* False Negative Cost = 100
* False Positive Cost = 1

Threshold Formula:

Threshold = Cost(FP) / (Cost(FP) + Cost(FN))

Calculated Threshold:

0.0099

This configuration achieved 100% Recall but introduced additional false positives, demonstrating the trade-off between operational cost and failure detection.

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn
* Imbalanced-Learn
* Joblib
* Jupyter Notebook

---

## Project Structure

Predictive-Maintenance-Assignment/

├── Predictive_Maintenance.ipynb

├── README.md

├── TECHNICAL_MEMO.md

├── requirements.txt

├── predictive_maintenance_model.pkl

└── images/

---

## Installation

```bash
git clone <repository_url>

cd Predictive-Maintenance-Assignment

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
```

---

## Run Project

Open Jupyter Notebook:

```bash
jupyter notebook
```

Run:

```text
Predictive_Maintenance.ipynb
```

from top to bottom.

---

## Conclusion

The Random Forest model outperformed Logistic Regression across PR-AUC, Precision, F1 Score, and ROC-AUC. The final model achieved a ROC-AUC of 0.9932 and PR-AUC of 
0.6224, making it the preferred solution for predictive maintenance in highly imbalanced manufacturing datasets.
