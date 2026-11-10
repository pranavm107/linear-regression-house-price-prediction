# House Price Prediction - Regression Analysis Report

## Executive Summary
This report summarizes the comparative empirical results between **Simple Linear Regression** (using lot area alone) and **Multiple Linear Regression** (using all 12 structural and amenity features) on the benchmark `Housing.csv` dataset.

---

## 1. Dataset & Partitioning
- **Total Records:** 545 observations across 13 columns (1 target + 12 explanatory variables).
- **Partitioning:** 80% training (436 samples), 20% testing (109 samples), partitioned reproducibly with `random_state=42`.
- **Target Variable:** `price` (Continuous, range: 1,750,000 to 13,300,000, Mean: 4,766,729).

---

## 2. Quantitative Model Performance (Test Set)

| Metric | Simple Linear Regression (Area) | Multiple Linear Regression (All Features) | Performance Differential |
|---|---|---|---|
| **Mean Absolute Error (MAE)** | 1,474,748.13 | 970,043.40 | **-34.22%** (Error reduced by 504,704.73) |
| **Mean Squared Error (MSE)** | 3,675,286,604,768.19 | 1,754,318,687,330.66 | **-52.27%** |
| **Root Mean Squared Error (RMSE)** | 1,917,103.70 | 1,324,506.96 | **-30.91%** (Error reduced by 592,596.74) |
| **Coefficient of Determination (R²)** | 0.2729 (27.29%) | 0.6529 (65.29%) | **+38.00% absolute increase** |

---

## 3. Estimated Model Equations

### Simple Linear Regression
$$\text{Price} = 2,512,254.26 + 425.73 \times \text{Area}$$

- **Slope Interpretation:** For every additional square foot of land area, the expected house price increases by **425.73 currency units**, holding no other factors constant.
- **Intercept Caveat:** Zero square feet falls outside the observed empirical domain (minimum observed area = 1,650 sq ft). The intercept represents the mathematical extrapolation at $X = 0$, not a realistic vacant lot or baseline structure value.

### Multiple Linear Regression
$$\text{Price} = \beta_0 + \sum_{i=1}^{13} \beta_i X_i$$

- **Intercept (\beta_0):** 260,032.36 (reference home: 0 sq ft, 0 bed/bath/stories/parking, not on main road, no guestroom, no basement, no hot water heating, no AC, not in preferred area, fully furnished).

#### Fitted Coefficients:
| Feature | Coefficient | Direction | Interpretation |
|---|---|---|---|
| area | 235.97 | Positive | Each additional square foot of lot area increases estimated price by ~235.97, all else equal. |
| bedrooms | 76,778.70 | Positive | Each additional bedroom is associated with a ~76,778.70 increase in estimated price. |
| bathrooms | 1,094,444.79 | Positive | Each additional bathroom increases estimated price substantially by ~1,094,444.79. |
| stories | 407,476.59 | Positive | Each additional story increases estimated price by ~407,476.59. |
| parking | 224,841.91 | Positive | Each additional dedicated parking space increases estimated price by ~224,841.91. |
| mainroad_yes | 367,919.95 | Positive | Houses connected to a main road command a ~367,919.95 premium over houses without direct main road access. |
| guestroom_yes | 231,610.04 | Positive | Homes with a dedicated guestroom add ~231,610.04 to estimated price. |
| basement_yes | 390,251.18 | Positive | Presence of a basement adds ~390,251.18 to estimated price. |
| hotwaterheating_yes | 684,649.89 | Positive | Inclusion of hot water heating adds ~684,649.89 to estimated price. |
| airconditioning_yes | 791,426.74 | Positive | Presence of air conditioning adds ~791,426.74 to estimated price. |
| prefarea_yes | 629,890.57 | Positive | Houses located in a preferred neighborhood area command a ~629,890.57 premium. |
| furnishingstatus_semi-furnished | -126,881.82 | Negative | Semi-furnished homes sell for ~126,881.82 less than fully furnished baseline homes. |
| furnishingstatus_unfurnished | -413,645.06 | Negative | Unfurnished homes sell for ~413,645.06 less than fully furnished baseline homes. |

---

## 4. Key Engineering & Econometric Insights
1. **Bathrooms and Amenities Dominate Marginal Value:** An additional bathroom increases predicted market value by **1,094,444.79**, making it the single largest positive structural contributor. Air conditioning (+791,426.74), hot water heating (+684,649.89), and preferred neighborhood area (+629,890.57) provide massive premiums.
2. **Furnishing Status Discount:** Relative to fully furnished properties, semi-furnished properties experience a 126,881.82 discount, while unfurnished properties experience a 413,645.06 discount.
3. **Multicollinearity Diagnostics:** All numerical features exhibit Variance Inflation Factors (VIF) between **1.16 and 1.32**, far below the rule-of-thumb threshold of 5.0, confirming that multicollinearity is not destabilizing the coefficient estimates.
4. **Generalization:** Multiple Linear Regression achieves a train R² of 0.6859 and a test R² of 0.6529. This modest drop demonstrates that the model generalizes robustly without severe overfitting.

---

## 5. Visual Artifacts Produced
- `simple_linear_regression.png`: Scatterplot with Ordinary Least Squares regression fit line.
- `simple_actual_vs_predicted.png`: Actual vs. predicted prices for simple regression.
- `multiple_actual_vs_predicted.png`: Actual vs. predicted prices for multiple regression.
- `simple_residuals.png`: Distribution and fitted vs. residual plots for simple regression.
- `multiple_residuals.png`: Distribution and fitted vs. residual plots for multiple regression.
- `price_distribution.png`: Histogram and boxplot of target variable.
- `area_distribution.png`: Histogram and boxplot of primary continuous feature.
- `correlation_heatmap.png`: Pearson correlation matrix across numerical variables.
- `numerical_features_vs_price.png`: Feature distributions across price tiers.
- `categorical_features_vs_price.png`: Boxplots of categorical amenities vs. price.
- `qq_plots.png`: Normal Q-Q plots evaluating residual normality.
- `vif_analysis.png`: Multicollinearity diagnostic chart.
