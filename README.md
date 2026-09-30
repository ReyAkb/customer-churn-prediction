# Telco Customer Churn Prediction

**Author:** Oybek Akbarov  
**Tools:** Python, scikit-learn, imbalanced-learn, pandas, NumPy, matplotlib, seaborn  

---

## Overview

This project builds and evaluates machine-learning classification models to predict customer churn for a telecommunications company. The goal is to identify customers who are likely to cancel their service so the business can intervene proactively.

**Key challenge:** The dataset is imbalanced, with roughly 70% of customers not churning and 30% churning. Because a model can achieve high accuracy by favoring the majority class, this project emphasizes recall, precision, and ROC AUC in addition to accuracy.

---

## Results Summary

| Model | AUC | Recall | Notes |
|-------|-----|--------|-------|
| Logistic Regression (PCA, 22 components) | 0.85 | — | Balanced training set |
| KNN (tuned, grid search) | 0.756 | 0.59 | Balanced training set |
| Gaussian Naive Bayes | 0.85 | 0.85 | Highest reported recall |
| AdaBoost | 0.85 | 0.76 | Selected final model |
| Gradient Boosting | 0.84 | 0.73 | Strong recall and AUC |
| Decision Tree (unbalanced) | 0.84 | 0.51 | 79% accuracy, lower recall |

**Selected model:** AdaBoost was selected as the final model based on the project's business priority of identifying customers who are likely to churn rather than maximizing raw accuracy. It achieved an AUC of 0.85 and recall of 0.76, providing a strong tradeoff between identifying churners and limiting false positives.

Gaussian Naive Bayes achieved higher recall (0.85), so AdaBoost was not selected because it had the highest recall. The selection reflects the project's emphasis on considering multiple evaluation metrics and the business tradeoff between false negatives and false positives.

---

## Dataset

- ~7,000 customer records
- 20+ features including contract type, monthly charges, tenure, internet service, and support ticket history
- Target variable: `Churn` (Yes / No)
- Source: Course dataset (`telco_churn.csv`)

---

## Methods

### Preprocessing

- Dropped identifier columns (`customerID`, `SupportTicketID`) and the constant `ActiveCountryCode` column
- Cleaned the `TotalCharges` column, which was stored as text and contained blank values
- Converted `Education` and `Contract` to ordinal values
- One-hot encoded nominal categorical columns
- Applied `MinMaxScaler` to the feature set
- Used a stratified 70/30 train/test split

### Class Imbalance

- Applied SMOTE (Synthetic Minority Over-sampling Technique) to the training set only
- Produced an approximately balanced 50/50 class distribution in the training data
- Compared model performance on balanced and unbalanced training data

### Dimensionality Reduction

- Evaluated PCA from 1 through the maximum number of available components
- Selected 22 components based on the observed cross-validated accuracy plateau
- Used PCA with logistic regression and KNN to evaluate the effect of dimensionality reduction

### Models Compared

- Logistic Regression
- K-Nearest Neighbors (KNN), including GridSearchCV tuning
- Gaussian Naive Bayes
- Categorical Naive Bayes
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 score
- ROC AUC
- Confusion matrix
- Precision-recall curve

---

## Key Findings

1. **Class balancing affected the precision-recall tradeoff.** SMOTE generally increased the model's ability to identify churners, while also increasing false positives for several models.
2. **AdaBoost provided a strong combination of recall and AUC**, achieving 0.76 recall and 0.85 AUC, and was selected based on the project's emphasis on churn detection.
3. **Gaussian Naive Bayes achieved the highest reported recall (0.85)**, demonstrating the importance of evaluating multiple metrics rather than relying on a single measure.
4. **High accuracy alone can be misleading.** The unbalanced Decision Tree achieved 79% accuracy but only 0.51 recall, meaning many actual churners were not identified.
5. **PCA with 22 components provided comparable logistic regression performance**, reducing the feature space from 44 features.

---

## How to Run

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebook/telco_churn_prediction.ipynb
```

Update the CSV path in the data-loading cell if necessary:

```python
telco = pd.read_csv('../data/telco_churn.csv')
```

---

## Repository Structure

```text
customer-churn-prediction/
├── README.md
├── data/
│   └── telco_churn.csv
└── notebook/
    └── telco_churn_prediction.ipynb
```