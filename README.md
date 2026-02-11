# Biomed-ML-Framework 🩺📊

![R](https://img.shields.io/badge/R-276DC3?style=for-the-badge&logo=r&logoColor=white)
![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)

An end-to-end automated machine learning framework tailored for clinical research. Built on the `tidymodels` ecosystem, this pipeline streamlines the process of training, tuning, and interpreting high-performance models while maintaining the rigorous validation standards required for medical publications.

## 🚀 Key Features

* **Multi-Engine Support**: Seamlessly switch between LightGBM, XGBoost, Random Forest (ranger), Penalized Regression (glmnet), and SVM.
* **Automated Feature Engineering**: Includes Information Gain-based selection for classification and Distance Correlation for regression tasks.
* **Hyperparameter Racing**: Utilizes ANOVA racing methods (`finetune`) to efficiently find optimal parameters without exhaustive grid searches.
* **Clinical Readiness**: 
    * **Decision Curve Analysis (DCA)** to evaluate clinical utility and net benefit.
    * **Bootstrap Confidence Intervals** (default 500 resamples) for all performance metrics.
    * **Threshold Optimization** using Youden’s J-statistic to maximize sensitivity and specificity.
* **Explainable AI (XAI)**: Advanced interpretability via model-specific importance and SHAP visualizations (beeswarm, dependence, and waterfall plots).
* **Automated Reporting**: Generates a diagnostic dashboard, comprehensive Excel summary, and saved model objects automatically.

## 🛠 Installation

The script uses the `pacman` package to manage dependencies. Ensure it is installed:

```R
install.packages("pacman")
source("Supervised ML.R")
```

## 📖 Quick Start
```R
results <- supervised_ml(
  modeling_data = my_clinical_data,
  model_type = "lgbm",           # Options: "rf", "xgb", "lgbm", "lr", "svm"
  mode = "classification",       # Options: "classification", "regression"
  target = "Outcome_Variable",
  num_cols = 2:10,               # Indices of numeric predictors
  cat_cols = 11:20,              # Indices of categorical predictors
  output_dir = "Model_Results"
)
```

## 📂 Project Structure
* **Supervised ML.R**: The core engine containing the supervised_ml function.
* **README.md**: Project documentation and setup guide.
* **.gitignore**: Configured to exclude large .rds model files and temporary R data.

## 📊 Visual Outputs
The framework automatically exports publication-quality plots:
* **Diagnostic Dashboard**: ROC/PR curves, Calibration, and Confusion Matrices.
* **SHAP Beeswarm**: Global feature impact on model predictions.
* **Clinical Case Studies**: SHAP Waterfall plots for True Positives, False Positives, and Borderline cases.
* **Forest Plots**: Mean bootstrap Odds Ratios/Coefficients with 95% CIs (for lr models).

## ⚖️ License
Copyright 2026 Rohit Agrawal
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
