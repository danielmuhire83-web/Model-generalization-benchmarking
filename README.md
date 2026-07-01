# Algorithmic Benchmarking: Bias-Variance Optimization and Hyperparameter Sensitivity Analysis

This repository contains a production-focused benchmarking suite that evaluates model capacity, generalization boundaries, and regularization strategies using `scikit-learn`. The codebase isolates and analyzes core machine learning challenges: mitigating overfitting in high-capacity parametric/non-parametric models, and mapping hyperparameter sensitivity curves on complex biomedical feature spaces.

---

## Technical Case Studies

### 1. Regression Analysis & Overfitting Mitigation
This case study evaluates how model architecture choices affect generalization when handling noisy data streams. Using a controlled, non-linear simulation environment with injected Gaussian noise, three distinct modeling strategies were stress-tested:

* **High-Degree Feature Expansion:** Expanding input features to an $11^{\text{th}}$-degree polynomial space exposed classic overfitting. The unregularized linear model captured high-frequency noise rather than the underlying signal, leading to excellent training performance but catastrophic validation failure. 
* **Regularization Strategy ($L_1$ Penalty):** Implementing a Lasso Regression model ($\alpha=0.01$) over the $11^{\text{th}}$-degree feature space successfully neutralized the overfitting. The $L_1$ penalty enforced coefficient sparsity, driving non-essential weights to absolute zero and restoring out-of-sample generalization.
* **Non-Parametric Baseline:** A $K$-Nearest Neighbors (KNN) Regressor was benchmarked alongside the parametric models to evaluate localized, distance-based estimation against global polynomial curve fitting.

### 2. Classification Optimization & Hyperparameter Sensitivity
This study focuses on optimizing a Support Vector Classifier (SVC) with a Radial Basis Function (RBF) kernel, utilizing the multi-dimensional physical features of the Breast Cancer Wisconsin dataset.

* **Validation Curve Mapping:** The model’s sensitivity to the kernel coefficient (`gamma`) was systematically tracked across a logarithmic scale from $10^{-7}$ to $10^{-2}$ to pinpoint the exact boundary between underfitting and overfitting.
* **Generalization Failure Modes:** 
  * **Underfitting Domain ($10^{-7}$):** Insufficient kernel flexibility resulted in high bias, causing poor classification accuracy across both training and testing datasets.
  * **Overfitting Domain ($10^{-2}$):** Excessive kernel localized flexibility allowed the model to perfectly memorize the training partition ($100\%$ accuracy), but caused validation performance to plummet due to extreme variance.
  * **Production-Ready Optimum ($10^{-4}$):** Identified the optimal peak where out-of-sample generalization is maximized before variance decay begins.

---

## 🚀 Engineering Trade-offs & Production Takeaways

* **Parametric Sparsity vs. Localized Estimation:** While KNN regression adapts flexibly to local fluctuations without feature engineering, regularized Lasso regression offers superior coefficient sparsity and interpretability. This makes Lasso highly preferred for resource-constrained production environments or compliance-heavy domains.
* **The Risk of Silent Overfitting:** The extreme sensitivity of the RBF kernel to the `gamma` parameter highlights the risk of silent overfitting in automated machine learning pipelines, emphasizing why strict validation boundaries and automated curve tracking are mandatory before deployment.

---

## Repository Structure

```text
├── README.md
├── requirements.txt
└── notebooks/
    └── regression_and_classification.ipynb
