import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# 1. Create synthetic data
# -----------------------------
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

# -----------------------------
# 2. Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -----------------------------
# 3. Define models and tuning grids
# -----------------------------
models = {
    "Ridge": {
        "model": Ridge(),
        "params": {
            "model__alpha": [0.01, 0.1, 1, 10, 100]
        }
    },
    "Lasso": {
        "model": Lasso(max_iter=10000),
        "params": {
            "model__alpha": [0.001, 0.01, 0.1, 1, 10]
        }
    },
    "Elastic Net": {
        "model": ElasticNet(max_iter=10000),
        "params": {
            "model__alpha": [0.001, 0.01, 0.1, 1, 10],
            "model__l1_ratio": [0.2, 0.5, 0.8]
        }
    }
}

# -----------------------------
# 4. Train and evaluate
# -----------------------------
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
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        "Model": name,
        "Best Parameters": grid.best_params_,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Nonzero Coefficients": np.sum(grid.best_estimator_.named_steps["model"].coef_ != 0)
    })
    
    coefs[name] = grid.best_estimator_.named_steps["model"].coef_

results_df = pd.DataFrame(results)
print(results_df)

# -----------------------------
# 5. Plot model comparison
# -----------------------------
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["RMSE"])
plt.ylabel("Test RMSE")
plt.title("Model Comparison: Ridge vs Lasso vs Elastic Net")
plt.show()

# -----------------------------
# 6. Compare coefficient shrinkage
# -----------------------------
coef_df = pd.DataFrame(coefs, index=feature_names)

plt.figure(figsize=(12, 6))
coef_df.plot(kind="bar", figsize=(14, 6))
plt.title("Coefficient Comparison")
plt.ylabel("Coefficient Value")
plt.xlabel("Features")
plt.tight_layout()
plt.show()