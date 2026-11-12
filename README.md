# ShadowScan — Phase 1
### Explainable Network Intrusion Detection

**Project:** ShadowScan  
**Phase:** 01  
**Status:** Proof of Concept Completed

## 1. Overview

Phase 1 of ShadowScan focuses on developing an explainable machine-learning-based network intrusion detection approach.

The research investigates how Explainable AI (XAI) can make network intrusion detection predictions more understandable by identifying the features influencing model decisions.

The team also developed a functional Streamlit dashboard prototype to present traffic classification results, model performance, and SHAP-based explanations.

## 2. Problem Statement

Machine-learning-based intrusion detection systems can identify suspicious network traffic, but their predictions may be difficult for responsible personnel to interpret.

Understanding why a particular traffic record is classified as malicious or benign is important when assessing cybersecurity alerts.

This phase explores the use of Explainable AI to provide feature-based explanations for network intrusion detection decisions.

## 3. Research Question

How can Explainable AI provide reliable and understandable explanations for network intrusion detection decisions, and how can these explanations support responsible personnel in interpreting and assessing cybersecurity alerts?

## 4. Objectives

- Explore machine-learning approaches for network traffic classification.
- Compare Decision Tree and Random Forest classifiers.
- Implement SHAP for explaining model predictions.
- Analyse global, class-wise and individual feature contributions.
- Examine model misclassifications and explanation behaviour.
- Develop a dashboard-based proof of concept for presenting detection results and explanations.

## 5. Methodology

The phase follows these main steps:

1. **Dataset Collection:** RT-IoT2022 dataset obtained from the UCI Machine Learning Repository.
2. **Preprocessing:** Data inspection, cleaning, categorical encoding, missing-value handling and feature preparation.
3. **Model Development:** Training and comparison of Decision Tree and Random Forest classifiers.
4. **Model Evaluation:** Evaluation using accuracy, macro F1-score, balanced accuracy and class-wise performance.
5. **Explainability:** Application of SHAP to analyse global, class-wise and individual model predictions.
6. **Prototype Development:** Integration of the selected model and explainability outputs into a Streamlit dashboard.

## 6. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- SHAP
- Matplotlib
- Streamlit
- Joblib

## 7. Prototype

The Phase 1 prototype includes:

- Overview Dashboard
- Traffic Inspector
- Model Performance
- Global XAI Insights
- Misclassification Audit

The prototype supports dataset/CSV-based traffic analysis and explanation of model predictions. It is not a live network monitoring or production deployment system.

## 8. Key Experimental Results

The selected Random Forest classifier achieved the following results on the held-out RT-IoT2022 test set:

| Metric | Result |
|---|---:|
| Accuracy | 99.8254% |
| Macro F1-score | 96.2245% |
| Balanced Accuracy | 96.5659% |
| Test Observations | 24,624 |

These results represent performance on the selected dataset and experimental setup. They do not establish generalisation to other network environments.

## 9. Scope and Limitations

- Experiments were conducted using RT-IoT2022.
- The dataset contains imbalanced traffic categories.
- Some minority classes have very limited test observations.
- SHAP explanations indicate feature contributions but do not establish causality or prediction correctness.
- Cross-dataset generalisation has not been evaluated.
- The prototype is dataset-based rather than a live network monitoring platform.

## 10. Future Direction

Phase 1 establishes the initial research and implementation foundation for ShadowScan.

Future phases will explore anomaly detection, security event correlation, risk intelligence, prioritisation and edge-assisted IoT monitoring.

## 11. Project Structure

The project implementation, experiment notebook, preprocessing files, saved model, SHAP outputs and Streamlit application are maintained in the `Phase One Code` directory.

Refer to the project files for implementation details and instructions for reproduction.

---

**ShadowScan — Observe. Detect. Explain. Protect.**
