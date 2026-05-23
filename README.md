# Ridge vs Lasso vs Elastic Net Regression

A machine learning project comparing three regularized regression methods:

* Ridge Regression
* Lasso Regression
* Elastic Net Regression

The project uses synthetic high-dimensional data to evaluate:

* Predictive performance
* Feature selection behavior
* Model interpretability
* Regularization effects

---

# Project Objective

The goal of this project is to compare how Ridge, Lasso, and Elastic Net regression perform on continuous-response prediction problems involving many predictors.

The project investigates:

* Which model achieves the lowest prediction error
* Which model produces the highest explanatory power (R²)
* How regularization impacts coefficient shrinkage
* Which models perform feature selection
* The tradeoff between prediction accuracy and interpretability

---

# Dataset

Synthetic regression data was generated using Scikit-learn.

Dataset characteristics:

| Property             | Value     |
| -------------------- | --------- |
| Samples              | 500       |
| Features             | 50        |
| Informative Features | 8         |
| Noise Level          | 20        |
| Train/Test Split     | 75% / 25% |

The dataset intentionally includes many non-informative predictors to demonstrate the effects of regularization and feature selection.

---

# Models Compared

## Ridge Regression

Ridge regression applies an L2 penalty to shrink coefficient values.

### Characteristics

* Handles multicollinearity well
* Reduces overfitting
* Keeps all predictors in the model

---

## Lasso Regression

Lasso regression applies an L1 penalty.

### Characteristics

* Performs automatic feature selection
* Shrinks some coefficients exactly to zero
* Produces simpler and more interpretable models

---

## Elastic Net Regression

Elastic Net combines both L1 and L2 penalties.

### Characteristics

* Balances Ridge and Lasso behavior
* Useful when predictors are highly correlated
* Can both shrink and select variables

---

# Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

# Machine Learning Workflow

The project follows these steps:

1. Generate synthetic regression data
2. Split data into training and testing sets
3. Standardize predictors
4. Train models using pipelines
5. Tune hyperparameters using GridSearchCV
6. Evaluate models using:

   * RMSE
   * MAE
   * R²
7. Compare coefficient behavior
8. Visualize results

---

# Evaluation Metrics

## RMSE (Root Mean Squared Error)

Measures prediction error magnitude.

RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}

Lower RMSE indicates better predictive accuracy.

---

## R² Score

Measures the proportion of variance explained by the model.

R^2=1-\frac{\sum (y_i-\hat{y}_i)^2}{\sum (y_i-\bar{y})^2}

Higher R² indicates better model fit.

---

# Example Findings

Typical results from the project:

* Ridge retained all variables
* Lasso selected a smaller subset of predictors
* Elastic Net behaved similarly to Ridge depending on tuning parameters
* Lasso achieved the best balance between predictive performance and interpretability

---

# Visualizations Included

The project generates:

* RMSE comparison chart
* R² comparison chart
* Number of selected variables
* Coefficient comparison plot

These visualizations help demonstrate the impact of regularization on model complexity and prediction performance.

---

# Example Output Questions

The code automatically answers:

* Which model had the lowest RMSE?
* Which model had the highest R²?
* How many variables did Lasso keep?
* Did Elastic Net select fewer variables than Ridge?
* Did Ridge keep all variables?
* Which model provides the best balance between accuracy and interpretability?

---

# How to Run the Project

## Clone the repository

```bash
git clone https://github.com/yourusername/ridge-lasso-elasticnet-comparison.git
cd ridge-lasso-elasticnet-comparison
```

---

## Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## Run the project

```bash
python PenaltySelection.py
```

---

# Project Structure

```bash
├── PenaltySelection.py
├── README.md
│── rmse_comparison.png
│── r2_comparison.png
│── variable_selection.png
│── coefficient_comparison.png
```

---

# Statistical Learning Concepts Demonstrated

This project demonstrates important statistical learning concepts including:

* Regularization
* Bias-variance tradeoff
* Feature selection
* Cross-validation
* High-dimensional regression
* Penalized regression methods
* Model interpretability
* Predictive analytics

---

# Future Improvements

Possible extensions include:

* Adding correlated predictors
* Increasing dimensionality
* Comparing against Linear Regression
* Using real-world datasets
* Adding Bootstrap analysis
* Testing additional regularization strengths
* Comparing against XGBoost or Random Forests

---

# Author

Feda Bashbishi
MBA, M.Sc. Eng., MDSAI
University of Waterloo

---

# License

This project is for educational and research purposes.
