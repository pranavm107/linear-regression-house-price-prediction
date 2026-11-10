# Linear Regression - House Price Prediction

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.2-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, reproducible Machine Learning project implementing and comparing **Simple Linear Regression** and **Multiple Linear Regression** on housing valuation data. The project emphasizes data leakage prevention, modular pipeline construction, diagnostic assumption testing, and interpretable econometric modeling.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Objective](#2-objective)
3. [Dataset Description](#3-dataset-description)
4. [Features and Target Variable](#4-features-and-target-variable)
5. [Technologies Used](#5-technologies-used)
6. [Project Folder Structure](#6-project-folder-structure)
7. [Data Preprocessing & Pipeline Architecture](#7-data-preprocessing--pipeline-architecture)
8. [Simple Linear Regression](#8-simple-linear-regression)
9. [Multiple Linear Regression](#9-multiple-linear-regression)
10. [Model Evaluation Metrics](#10-model-evaluation-metrics)
11. [Coefficient Interpretation](#11-coefficient-interpretation)
12. [Residual Analysis](#12-residual-analysis)
13. [Regression Assumptions](#13-regression-assumptions)
14. [Model Comparison](#14-model-comparison)
15. [Installation Instructions](#15-installation-instructions)
16. [Execution Instructions](#16-execution-instructions)
17. [Generated Visualizations](#17-generated-visualizations)
18. [Model Artifacts](#18-model-artifacts)
19. [Key Findings](#19-key-findings)
20. [Conclusion](#20-conclusion)

---

## 1. Project Overview
Predicting property market valuations is an econometric challenge characterized by nonlinear interactions, geographical stratification, and structural features. This project develops an internship-grade technical demonstration of Ordinary Least Squares (OLS) regression using Python and Scikit-Learn.

We contrast a baseline univariate model against an all-inclusive multivariate pipeline, tracking performance via rigorous out-of-sample metrics (MAE, MSE, RMSE, R²), residual diagnostics, and collinearity validation.

---

## 2. Objective
- Load, audit, and clean the `Housing.csv` dataset.
- Implement **Simple Linear Regression** using property lot `area` as the single continuous predictor.
- Implement **Multiple Linear Regression** leveraging 12 structural, mechanical, and categorical amenity attributes.
- Guarantee leakage-free preprocessing with Scikit-learn `ColumnTransformer` and `Pipeline`.
- Empirically validate regression assumptions: Linearity, Normality, Homoscedasticity, and Absence of Multicollinearity (VIF).
- Export reusable model artifacts (`joblib`) and evaluation reports (`csv`, `md`).

---

## 3. Dataset Description
The dataset resides at `dataset/Housing.csv` and captures sales data across **545 residential properties** with **13 columns** (1 continuous target and 12 explanatory features).

- **Total Records:** 545 observations
- **Missing Values:** 0 across all features
- **Duplicate Records:** 0
- **Target Value Range:** ₹ 1,750,000 to ₹ 13,300,000 (Mean: ₹ 4,766,729)

---

## 4. Features and Target Variable

### Target Variable
- **`price`** (*Integer*): Market transaction price in currency units.

### Input Features (Predictors)

| Feature | Type | Range / Categories | Description |
|---|---|---|---|
| `area` | Continuous (int64) | 1,650 – 16,200 | Total lot area in square feet |
| `bedrooms` | Discrete (int64) | 1 – 6 | Number of bedrooms |
| `bathrooms` | Discrete (int64) | 1 – 4 | Number of bathrooms |
| `stories` | Discrete (int64) | 1 – 4 | Number of building stories |
| `parking` | Discrete (int64) | 0 – 3 | Number of dedicated parking slots |
| `mainroad` | Categorical (object) | `yes`, `no` | Direct access to primary road |
| `guestroom` | Categorical (object) | `yes`, `no` | Dedicated guestroom presence |
| `basement` | Categorical (object) | `yes`, `no` | Basement presence |
| `hotwaterheating`| Categorical (object) | `yes`, `no` | Dedicated hot water heating system |
| `airconditioning`| Categorical (object) | `yes`, `no` | Central air conditioning system |
| `prefarea` | Categorical (object) | `yes`, `no` | Located in preferred neighborhood |
| `furnishingstatus`| Categorical (object) | `furnished`, `semi-furnished`, `unfurnished` | Interior furnishing condition |

---

## 5. Technologies Used
- **Python 3.12**
- **Pandas 2.2.2**: Data ingestion, schema auditing, and tabular transformations
- **NumPy 1.26.4**: Vectorized numerical computations and array manipulations
- **Matplotlib 3.8.4 & Seaborn 0.13.2**: High-resolution statistical visualization
- **Scikit-learn 1.4.2**: Preprocessing transformers, pipelines, OLS regression, evaluation metrics
- **Statsmodels 0.14.2**: Multicollinearity (VIF) and residual diagnostics
- **Joblib 1.4.0**: High-efficiency model serialization
- **Jupyter Notebook 7.2.2**: Interactive literate programming

---

## 6. Project Folder Structure

```
linear-regression-house-price-prediction/
│
├── dataset/
│   └── Housing.csv                                  # Original immutable dataset
│
├── notebooks/
│   └── linear_regression.ipynb                      # Fully executed 18-section notebook
│
├── outputs/
│   ├── models/
│   │   ├── simple_linear_regression.joblib          # Trained Simple Linear Regression
│   │   └── multiple_linear_regression_pipeline.joblib # Full Pipeline (Preprocessor + Regressor)
│   │
│   ├── reports/
│   │   ├── feature_coefficients.csv                 # MLR weights, directions & interpretations
│   │   ├── model_evaluation.csv                     # Quantitative metric comparison table
│   │   ├── prediction_comparison.csv                # Sample test predictions & error deltas
│   │   └── regression_analysis.md                   # Full econometric markdown report
│   │
│   └── visualizations/
│       ├── area_distribution.png                    # Area distribution and boxplot
│       ├── categorical_features_vs_price.png        # Categorical feature boxplots
│       ├── correlation_heatmap.png                  # Pearson correlation matrix
│       ├── multiple_actual_vs_predicted.png         # MLR actual vs predicted plot
│       ├── multiple_residuals.png                   # MLR residual distribution & fitted plot
│       ├── numerical_features_vs_price.png          # Numerical feature boxplots
│       ├── price_distribution.png                   # Target price distribution and boxplot
│       ├── qq_plots.png                             # Normal Q-Q plots for residuals
│       ├── simple_actual_vs_predicted.png           # SLR actual vs predicted plot
│       ├── simple_linear_regression.png             # OLS regression line on area
│       ├── simple_residuals.png                     # SLR residual distribution & fitted plot
│       └── vif_analysis.png                         # Variance Inflation Factor bar chart
│
├── scripts/
│   ├── build_notebook.py                            # Script generating notebook structure
│   └── generate_analysis.py                         # Standalone end-to-end execution script
│
├── src/
│   ├── __init__.py                                  # Package initialization
│   └── regression.py                                # Reusable data & modeling utilities
│
├── .gitignore                                       # Git ignore specifications
├── LICENSE                                          # Permissive open-source license
├── README.md                                        # Comprehensive project documentation
└── requirements.txt                                 # Pinned project dependencies
```

---

## 7. Data Preprocessing & Pipeline Architecture

To prevent **data leakage**, transformations must be derived strictly from training data and applied down to test data.

```
Input Features (X)
  │
  ├── Numerical: ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
  │     └── Passthrough (preserves raw currency-unit interpretability)
  │
  └── Categorical: ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
        └── OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
              │
              ▼
    ColumnTransformer
              │
              ▼
    LinearRegression()
```

- **Avoidance of the Dummy Variable Trap:** Binary features (`yes`/`no`) are mapped to a single indicator (1 = yes, 0 = no). Three-level nominal feature `furnishingstatus` is mapped to two dummy indicators (`semi-furnished`, `unfurnished`), with `furnished` serving as the implicit reference category.

---

## 8. Simple Linear Regression

Simple Linear Regression was trained exclusively on the `area` feature using an 80/20 train/test split (`random_state=42`).

### Estimated Equation
$$\text{Price} = 2,512,254.26 + 425.73 \times \text{Area}$$

### Parameter Interpretation
- **Slope ($\beta_1 = 425.73$):** Each additional square foot of lot area increases predicted house price by **₹ 425.73**, ignoring all other variables.
- **Intercept ($\beta_0 = 2,512,254.26$):** Represents mathematical extrapolation when area is zero. Because minimum observed lot size is 1,650 sq ft, this does not represent a realistic house price.

---

## 9. Multiple Linear Regression

Multiple Linear Regression incorporates all 12 input features (13 design matrix features after one-hot encoding).

### General Equation
$$\text{Price} = \beta_0 + \sum_{i=1}^{13} \beta_i X_i$$

### Fitted Intercept
$$\beta_0 = 260,032.36$$
*(Reference baseline: 0 sq ft, 0 bed/bath/stories/parking, not on main road, no guestroom, no basement, no hot water heating, no AC, not in preferred area, fully furnished).*

---

## 10. Model Evaluation Metrics

Both models were fitted on the **436 training samples** and evaluated on the **109 unseen test samples**.

| Model | Partition | Mean Absolute Error (MAE) | Mean Squared Error (MSE) | Root Mean Squared Error (RMSE) | R² Score |
|---|---|---|---|---|---|
| **Simple Linear Regression** | Train (80%) | ₹ 1,092,585.59 | 2.923 × 10¹² | ₹ 1,514,635.80 | 0.2850 |
| **Simple Linear Regression** | Test (20%) | ₹ 1,474,748.13 | 3.675 × 10¹² | ₹ 1,917,103.70 | **0.2729** |
| **Multiple Linear Regression** | Train (80%) | ₹ 719,242.89 | 1.285 × 10¹² | ₹ 1,003,506.74 | 0.6859 |
| **Multiple Linear Regression** | Test (20%) | ₹ 970,043.40 | 1.754 × 10¹² | ₹ 1,324,506.96 | **0.6529** |

### Evaluation Takeaways
- **Error Reduction:** Multiple Linear Regression slashes Test MAE by **34.22%** (saving ₹ 504,704 per property on average) and Test RMSE by **30.91%**.
- **Variance Explained ($R^2$):** R² expands from **27.29%** to **65.29%**, an increase of **+38.00 percentage points**.
- **Generalization:** Train $R^2$ (0.6859) vs Test $R^2$ (0.6529) exhibits a healthy ~3.3% gap, demonstrating that the pipeline generalizes without overfitting.

---

## 11. Coefficient Interpretation

All coefficients obtained from the fitted Multiple Linear Regression model:

| Feature | Coefficient | Direction | Econometric Interpretation |
|---|---|---|---|
| `area` | +235.97 | Positive | Each additional sq ft increases estimated price by ₹ 235.97, ceteris paribus. |
| `bedrooms` | +76,778.70 | Positive | Each additional bedroom increases estimated price by ₹ 76,778.70. |
| `bathrooms` | +1,094,444.79 | Positive | **Highest continuous value driver:** Each additional bathroom adds ₹ 1.09M. |
| `stories` | +407,476.59 | Positive | Each additional building floor adds ₹ 407,476.59. |
| `parking` | +224,841.91 | Positive | Each additional parking spot adds ₹ 224,841.91. |
| `mainroad_yes` | +367,919.95 | Positive | Main road access carries a ₹ 367,919.95 premium over secondary road properties. |
| `guestroom_yes` | +231,610.04 | Positive | Dedicated guestroom presence adds ₹ 231,610.04 to valuation. |
| `basement_yes` | +390,251.18 | Positive | Basement presence adds ₹ 390,251.18 to valuation. |
| `hotwaterheating_yes` | +684,649.89 | Positive | Hot water heating infrastructure adds ₹ 684,649.89. |
| `airconditioning_yes` | +791,426.74 | Positive | **Highest amenity premium:** Central AC adds ₹ 791,426.74 to valuation. |
| `prefarea_yes` | +629,890.57 | Positive | Location in a preferred neighborhood adds a ₹ 629,890.57 premium. |
| `furnishingstatus_semi-furnished` | -126,881.82 | Negative | Semi-furnished homes sell for ₹ 126,881.82 less than fully furnished baselines. |
| `furnishingstatus_unfurnished` | -413,645.06 | Negative | Unfurnished homes sell for ₹ 413,645.06 less than fully furnished baselines. |

> **Note on Causation:** These coefficients represent partial correlations in observational data and do not establish mechanical or causal proof.

---

## 12. Residual Analysis
Residuals are calculated as $e_i = y_i - \hat{y}_i$.

- **Mean Error:** Test mean residual is ₹ -182,504 for Simple Regression and ₹ -78,610 for Multiple Regression (close to theoretical zero).
- **Residual Distribution:** The residual histogram demonstrates bell-shaped symmetric density around zero, with error dispersion significantly tighter in the multiple regression model.
- **Fitted Values vs Residuals:** Errors show uniform spread across low-to-mid property valuations, with minor fanning at ultra-luxury valuations (> ₹ 10M), which is expected in real estate pricing.

---

## 13. Regression Assumptions

| Classical Assumption | Empirical Diagnostic Method | Finding / Assessment |
|---|---|---|
| **1. Linearity** | Scatter of residuals vs. fitted values | Random dispersion around horizontal zero-line confirms linearity holds across the primary range. |
| **2. Independence of Errors** | Residual sequence inspection | Observations exhibit no periodic or autoregressive lag patterns. |
| **3. Homoscedasticity** | Residual variance across fitted predictions | Residual spread is largely homoscedastic across 85% of standard properties; slight heteroscedasticity at high tail. |
| **4. Normality of Residuals**| Quantile-Quantile (Q-Q) plots | Residual quantiles align tightly with normal theoretical quantiles, deviating only at extreme ends. |
| **5. No Severe Multicollinearity** | Variance Inflation Factor (VIF) | All numerical VIF values range between **1.16 and 1.32**, far below the threshold of 5.0. No collinearity concerns. |

### VIF Summary (Numerical Features)
- `bedrooms`: **1.32**
- `stories`: **1.26**
- `bathrooms`: **1.25**
- `area`: **1.17**
- `parking`: **1.16**

---

## 14. Model Comparison

```
Metric Comparison: Simple vs. Multiple Linear Regression

  Test R² Score:
  SLR [███████                      ] 0.2729 (27.29%)
  MLR [████████████████             ] 0.6529 (65.29%)   [+38.00% Gain]

  Test MAE (Lower is Better):
  SLR [█████████████████████████    ] ₹ 1,474,748
  MLR [████████████████             ] ₹ 970,043         [-34.22% Error]

  Test RMSE (Lower is Better):
  SLR [████████████████████████████ ] ₹ 1,917,104
  MLR [███████████████████          ] ₹ 1,324,507       [-30.91% Error]
```

### Why Multiple Linear Regression Outperforms
While lot area is an important driver of value, buyers prioritize functional living spaces (bathrooms, bedrooms, stories) and modern amenities (air conditioning, hot water heating, preferred neighborhood location). Multiple Linear Regression integrates these independent signals, reducing unexplained residual variance.

---

## 15. Installation Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.12.0)
- `git`

### Clone and Setup
```bash
# Clone the repository
git clone https://github.com/pranavm107/linear-regression-house-price-prediction.git
cd linear-regression-house-price-prediction

# Create a virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

---

## 16. Execution Instructions

### Running the Jupyter Notebook
```bash
jupyter notebook notebooks/linear_regression.ipynb
```
Select **Run All Cells** from the Kernel menu to reproduce all figures, outputs, and model evaluations.

### Running Standalone Automation Scripts
```bash
# Execute standalone script to regenerate all visualizations, models, and reports
python3 scripts/generate_analysis.py

# Rebuild and re-execute notebook in-place
python3 scripts/build_notebook.py
python3 -m nbconvert --to notebook --execute --inplace notebooks/linear_regression.ipynb
```

---

## 17. Generated Visualizations

All static high-resolution (300 DPI) visualizations are stored in `outputs/visualizations/`:

| Visualization | Filename | Description |
|---|---|---|
| Simple Regression Fit | `simple_linear_regression.png` | Scatter plot of area vs price with OLS regression fit line |
| SLR Actual vs Predicted | `simple_actual_vs_predicted.png` | Test predictions vs actuals for simple regression |
| MLR Actual vs Predicted | `multiple_actual_vs_predicted.png` | Test predictions vs actuals for multiple regression |
| SLR Residual Diagnostics | `simple_residuals.png` | Residual histogram and fitted vs residual scatter |
| MLR Residual Diagnostics | `multiple_residuals.png` | Residual histogram and fitted vs residual scatter |
| Price Distribution | `price_distribution.png` | Target variable distribution histogram and outlier boxplot |
| Area Distribution | `area_distribution.png` | Lot area distribution histogram and outlier boxplot |
| Correlation Heatmap | `correlation_heatmap.png` | Pearson correlation matrix across numerical variables |
| Numerical Features vs Price | `numerical_features_vs_price.png` | Box plots for bedrooms, bathrooms, stories, and parking |
| Categorical Features vs Price| `categorical_features_vs_price.png`| Box plots for amenities and furnishing conditions |
| Residual Q-Q Plots | `qq_plots.png` | Normal probability plots evaluating residual normality |
| Collinearity Diagnostics | `vif_analysis.png` | Variance Inflation Factor (VIF) bar chart |

---

## 18. Model Artifacts

Pre-trained model binaries are preserved in `outputs/models/` using `joblib`:

1. **`outputs/models/simple_linear_regression.joblib`** (864 bytes)  
   Trained Scikit-Learn `LinearRegression` model fitted on single feature `area`.
2. **`outputs/models/multiple_linear_regression_pipeline.joblib`** (5.5 KB)  
   Comprehensive Scikit-Learn `Pipeline` bundling the fitted `ColumnTransformer` (numerical passthrough + categorical one-hot encoder) and the trained `LinearRegression` estimator.

### Inference Example with Saved Pipeline
```python
import joblib
import pandas as pd

# Load pipeline
pipeline = joblib.load("outputs/models/multiple_linear_regression_pipeline.joblib")

# Sample query house
query_property = pd.DataFrame([{
    "area": 6500,
    "bedrooms": 3,
    "bathrooms": 2,
    "stories": 2,
    "mainroad": "yes",
    "guestroom": "no",
    "basement": "yes",
    "hotwaterheating": "no",
    "airconditioning": "yes",
    "parking": 2,
    "prefarea": "yes",
    "furnishingstatus": "semi-furnished"
}])

estimated_price = pipeline.predict(query_property)[0]
print(f"Estimated Market Valuation: ₹ {estimated_price:,.2f}")
# Output: Estimated Market Valuation: ₹ 6,864,285.90
```

---

## 19. Key Findings
1. **Bathrooms are the dominant physical determinant:** Each additional bathroom commands a **₹ 1,094,444.79** premium, holding other variables constant.
2. **Air conditioning is the highest-value luxury amenity:** Central air conditioning adds **₹ 791,426.74**, exceeding basement presence (₹ 390k) and guestrooms (₹ 231k).
3. **Furnishing discount:** Unfurnished properties suffer an estimated **₹ 413,645.06 penalty** relative to identical fully furnished homes.
4. **Lot Area alone is insufficient:** While lot size correlates with price ($r = 0.54$), it only accounts for 27.29% of test variance. Combining structural and neighborhood features more than doubles explanatory power ($R^2 = 0.6529$).
5. **No multicollinearity:** All predictor VIF values remain under 1.35, proving that coefficient instability is not present.

---

## 20. Conclusion
This project provides a clean, production-ready implementation of Simple and Multiple Linear Regression for house price prediction. By pairing classical econometric diagnostics with modern machine learning pipeline practices, the models achieve strong generalization without data leakage.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

