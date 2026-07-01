# Applied Machine Learning Optimization & Benchmarking Suite

This repository serves as a centralized production portfolio demonstrating the deployment, evaluation, and optimization of traditional supervised and unsupervised machine learning algorithms using `scikit-learn`. Each case study systematically isolates core engineering hurdles—including regularization design, hyperparameter sensitivity, ensemble calibration, and high-dimensional space partitioning.

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

### Case Study 4: Unsupervised Architecture & Dimensionality Reduction
* **Core Mechanics:** Dimensionality Reduction (PCA / t-SNE), Clustering Frameworks ($K$-Means, Agglomerative), Mixture Models.
* **Engineering Impact:** Developed unsupervised pipelines to compress high-dimensional feature representations while preserving geometric structure. Engineered clustering validation boundaries using cluster density and separation indices to segment unlabeled data spaces.

---

## 🚀 Key Production Takeaways

* **Metric Selection Over Accuracy:** Case Study 3 and 4 highlight that accuracy is a dangerous metric in production environments. Calibrating Precision-Recall curves and monitoring ROC-AUC guarantees that structural imbalances don't cause silent model degradation.
* **Regularization vs. Ensemble Averaging:** While Case Study 2 relies on mathematical penalties ($L_1$/$L_2$) to force simplicity on individual over-parameterized estimators, Case Study 3 leverages variance-reduction techniques via bootstrapping and ensemble averaging (bagging/boosting) to achieve stability across complex boundaries.

---

## Repository Structure

```text
├── README.md
├── requirements.txt
└── notebooks/
    ├── 01_foundational_classifiers_and_tuning.ipynb
    ├── 02_bias_variance_and_regularization.ipynb
    ├── 03_ensemble_methods_and_evaluation_metrics.ipynb
    └── 04_unsupervised_learning_and_clustering.ipynb
