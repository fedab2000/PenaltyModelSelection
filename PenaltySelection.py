import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Create synthetic data
X, y, true_coef = make_regression(
    n_samples=500,
    n_features=50,
    n_informative=8,
    noise=20,
    coef=True,
    random_state=42
)

feature_names = [f"X{i+1}" for i in range(X.shape[1])]
X = pd.DataFrame(X, columns=feature_names)

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 3. Define models
models = {
    "Ridge": {
        "model": Ridge(),
        "params": {"model__alpha": [0.01, 0.1, 1, 10, 100]}
    },
    "Lasso": {
        "model": Lasso(max_iter=10000),
        "params": {"model__alpha": [0.001, 0.01, 0.1, 1, 10]}
    },
    "Elastic Net": {
        "model": ElasticNet(max_iter=10000),
        "params": {
            "model__alpha": [0.001, 0.01, 0.1, 1, 10],
            "model__l1_ratio": [0.2, 0.5, 0.8]
        }
    }
}

# 4. Train and evaluate
results = []
coefs = {}

for name, item in models.items():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", item["model"])
    ])

    grid = GridSearchCV(
        pipe,
        item["params"],
        cv=5,
        scoring="neg_mean_squared_error"
    )

    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)

    coef_values = grid.best_estimator_.named_steps["model"].coef_
    nonzero_count = np.sum(coef_values != 0)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Best Parameters": grid.best_params_,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Nonzero Coefficients": nonzero_count
    })

    coefs[name] = coef_values

results_df = pd.DataFrame(results)

print("\nModel Comparison Results:")
print(results_df)

# 5. Answer project questions automatically
lowest_rmse_model = results_df.loc[results_df["RMSE"].idxmin()]
highest_r2_model = results_df.loc[results_df["R2"].idxmax()]

ridge_vars = results_df.loc[results_df["Model"] == "Ridge", "Nonzero Coefficients"].values[0]
lasso_vars = results_df.loc[results_df["Model"] == "Lasso", "Nonzero Coefficients"].values[0]
elastic_vars = results_df.loc[results_df["Model"] == "Elastic Net", "Nonzero Coefficients"].values[0]

print("\nInterpretation Questions:")
print(f"1. The model with the lowest RMSE is {lowest_rmse_model['Model']} with RMSE = {lowest_rmse_model['RMSE']:.2f}.")
print(f"2. The model with the highest R² is {highest_r2_model['Model']} with R² = {highest_r2_model['R2']:.3f}.")
print(f"3. Lasso kept {lasso_vars} variables.")
print(f"4. Elastic Net kept {elastic_vars} variables, while Ridge kept {ridge_vars} variables.")

if elastic_vars < ridge_vars:
    print("   Yes, Elastic Net selected fewer variables than Ridge.")
else:
    print("   No, Elastic Net did not select fewer variables than Ridge.")

if ridge_vars == X.shape[1]:
    print(f"5. Yes, Ridge kept all {X.shape[1]} variables.")
else:
    print(f"5. No, Ridge kept {ridge_vars} out of {X.shape[1]} variables.")

# 6. Best balance between accuracy and interpretability
best_rmse = results_df["RMSE"].min()

results_df["RMSE Gap From Best"] = results_df["RMSE"] - best_rmse

balanced_candidates = results_df[
    results_df["RMSE Gap From Best"] <= 0.05 * best_rmse
]

best_balance = balanced_candidates.sort_values(
    by=["Nonzero Coefficients", "RMSE"]
).iloc[0]

print(f"6. The best balance between accuracy and interpretability is {best_balance['Model']}.")
print(
    f"   It has RMSE = {best_balance['RMSE']:.2f}, "
    f"R² = {best_balance['R2']:.3f}, and keeps "
    f"{best_balance['Nonzero Coefficients']} variables."
)

# 7. Plot RMSE comparison
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["RMSE"])
plt.ylabel("Test RMSE")
plt.title("Model Comparison: Ridge vs Lasso vs Elastic Net")
plt.show()

# 8. Plot R² comparison
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["R2"])
plt.ylabel("Test R²")
plt.title("R² Comparison: Ridge vs Lasso vs Elastic Net")
plt.show()

# 9. Plot number of selected variables
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["Nonzero Coefficients"])
plt.ylabel("Number of Nonzero Coefficients")
plt.title("Model Interpretability: Number of Variables Kept")
plt.show()

# 10. Coefficient comparison
coef_df = pd.DataFrame(coefs, index=feature_names)

coef_df.plot(kind="bar", figsize=(14, 6))
plt.title("Coefficient Comparison")
plt.ylabel("Coefficient Value")
plt.xlabel("Features")
plt.tight_layout()
plt.show()