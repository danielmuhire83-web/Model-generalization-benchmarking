# Applied Machine Learning Optimization & Benchmarking Suite

This repository serves as a centralized production portfolio demonstrating the deployment, evaluation, and optimization of supervised machine learning algorithms using `scikit-learn`. Each case study systematically isolates core engineering hurdles—including regularization design, hyperparameter sensitivity, ensemble calibration, and non-linear boundary mapping.

---

## 📂 Case Studies & Engineering Breakdowns

### Case Study 1: Foundational Frameworks & Parametric Baselines
* **Core Mechanics:** $K$-Nearest Neighbors, Baseline Linear/Logistic Models, Metric Scalability.
* **Engineering Impact:** Established foundational benchmarks for distance-based and linear modeling architectures. Focused heavily on baseline preprocessing pipelines, feature alignment, and diagnostic testing to track data leakage before expanding into higher-capacity models.

### Case Study 2: Parametric Capacity & Regularization Optimization
* **Core Mechanics:** Bias-Variance Tradeoff, Polynomial Feature Expansion, $L_1$ Regularization (Lasso), RBF Kernel Support Vector Classifiers (SVC).
* **Engineering Impact:** Stress-tested an unregularized $11^{\text{th}}$-degree polynomial space (which suffered from severe variance decay) against an $L_1$-penalized model to enforce structural coefficient sparsity. Additionally, mapped cross-validation curves across logarithmic hyperparameter spaces ($10^{-7}$ to $10^{-2}$) for RBF kernels to pinpoint exact boundaries preventing overfitting.

### Case Study 3: Tree Ensembles & Performance Calibration
* **Core Mechanics:** Decision Trees, Random Forests, Gradient Boosted Decision Trees (GBDT), ROC-AUC, Precision-Recall Calibration.
* **Engineering Impact:** Benchmarked robust non-linear ensemble architectures against baseline estimators. Emphasized the selection of cost-sensitive evaluation metrics over raw accuracy to protect model deployment against class imbalance risks, optimizing decision thresholds for production readiness.

### Case Study 4: Non-Linear Architectures & Validation Strategies
* **Core Mechanics:** Support Vector Machines (SVM), Multi-layer Perceptrons (Neural Networks), Complex Boundary Mapping, Grid/Randomized Search.
* **Engineering Impact:** Evaluated high-capacity, non-linear architectures capable of mapping intricate decision boundaries. Focused heavily on scaling features for optimization convergence and leveraging systematic hyperparameter search strategies to prevent variance issues in deep network layers.

---

## 🚀 Key Production Takeaways

* **Metric Selection Over Accuracy:** Case Study 3 highlights that accuracy is a dangerous metric in production environments. Calibrating Precision-Recall curves and monitoring ROC-AUC guarantees that structural imbalances don't cause silent model degradation.
* **Architecture Scaling and Convergence:** As demonstrated in Case Study 4, moving from tree-based ensembles to gradient-descent-driven models like Neural Networks or margin-maximizing SVMs shifts the engineering focus toward meticulous data scaling and convergence diagnostics.

---

## Repository Structure

```text
├── README.md
├── requirements.txt
└── notebooks/
    ├── 01_foundational_classifiers_and_tuning.ipynb
    ├── 02_bias_variance_and_regularization.ipynb
    ├── 03_ensemble_methods_and_evaluation_metrics.ipynb
    └── 04_non_linear_architectures_and_validation.ipynb
