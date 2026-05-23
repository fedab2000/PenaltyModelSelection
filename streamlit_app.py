import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Ridge vs Lasso vs Elastic Net",
    page_icon="📈",
    layout="wide"
)

st.title("Ridge vs Lasso vs Elastic Net Regression")

st.write("""
This Streamlit app compares Ridge, Lasso, and Elastic Net regression models
using synthetic regression data.
""")

# ---------------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------------

st.sidebar.header("Synthetic Data Settings")

n_samples = st.sidebar.slider(
    "Number of samples",
    100,
    2000,
    500,
    100
)

n_features = st.sidebar.slider(
    "Number of features",
    10,
    200,
    50,
    10
)

n_informative = st.sidebar.slider(
    "Number of informative features",
    2,
    min(30, n_features),
    8,
    1
)

noise = st.sidebar.slider(
    "Noise level",
    0,
    150,
    20,
    5
)

test_size = st.sidebar.slider(
    "Test size",
    0.10,
    0.50,
    0.25,
    0.05
)

random_state = st.sidebar.number_input(
    "Random state",
    value=42
)

# ---------------------------------------------------
# MODEL PARAMETERS
# ---------------------------------------------------

st.sidebar.header("Model Tuning")

ridge_alphas = st.sidebar.multiselect(
    "Ridge alpha values",
    [0.01, 0.1, 1, 10, 100],
    default=[0.01, 0.1, 1, 10, 100]
)

lasso_alphas = st.sidebar.multiselect(
    "Lasso alpha values",
    [0.001, 0.01, 0.1, 1, 10],
    default=[0.001, 0.01, 0.1, 1, 10]
)

elastic_alphas = st.sidebar.multiselect(
    "Elastic Net alpha values",
    [0.001, 0.01, 0.1, 1, 10],
    default=[0.001, 0.01, 0.1, 1, 10]
)

elastic_l1_ratios = st.sidebar.multiselect(
    "Elastic Net l1_ratio values",
    [0.2, 0.5, 0.8],
    default=[0.2, 0.5, 0.8]
)

# ---------------------------------------------------
# DATA GENERATION
# ---------------------------------------------------

@st.cache_data
def generate_data():

    X, y, true_coef = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        coef=True,
        random_state=int(random_state)
    )

    feature_names = [f"X{i+1}" for i in range(n_features)]

    X = pd.DataFrame(X, columns=feature_names)

    return X, y, true_coef, feature_names

# ---------------------------------------------------
# TRAIN MODELS
# ---------------------------------------------------

def train_models(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=int(random_state)
    )

    models = {

        "Ridge": {
            "model": Ridge(),
            "params": {
                "model__alpha": ridge_alphas
            }
        },

        "Lasso": {
            "model": Lasso(max_iter=10000),
            "params": {
                "model__alpha": lasso_alphas
            }
        },

        "Elastic Net": {
            "model": ElasticNet(max_iter=10000),
            "params": {
                "model__alpha": elastic_alphas,
                "model__l1_ratio": elastic_l1_ratios
            }
        }
    }

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

        nonzero_count = np.sum(np.abs(coef_values) > 1e-8)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        mae = mean_absolute_error(y_test, y_pred)

        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "Best Parameters": str(grid.best_params_),
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Nonzero Coefficients": nonzero_count
        })

        coefs[name] = coef_values

    results_df = pd.DataFrame(results)

    coef_df = pd.DataFrame(
        coefs,
        index=X.columns
    )

    return results_df, coef_df

# ---------------------------------------------------
# RUN PIPELINE
# ---------------------------------------------------

X, y, true_coef, feature_names = generate_data()

results_df, coef_df = train_models(X, y)

# ---------------------------------------------------
# DATA SUMMARY
# ---------------------------------------------------

st.subheader("Dataset Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Samples", n_samples)
col2.metric("Features", n_features)
col3.metric("Informative Features", n_informative)
col4.metric("Noise", noise)

# ---------------------------------------------------
# RESULTS TABLE
# ---------------------------------------------------

st.subheader("Model Results")

st.dataframe(results_df)

# ---------------------------------------------------
# AUTOMATIC INTERPRETATION
# ---------------------------------------------------

lowest_rmse_model = results_df.loc[
    results_df["RMSE"].idxmin()
]

highest_r2_model = results_df.loc[
    results_df["R2"].idxmax()
]

ridge_vars = results_df.loc[
    results_df["Model"] == "Ridge",
    "Nonzero Coefficients"
].values[0]

lasso_vars = results_df.loc[
    results_df["Model"] == "Lasso",
    "Nonzero Coefficients"
].values[0]

elastic_vars = results_df.loc[
    results_df["Model"] == "Elastic Net",
    "Nonzero Coefficients"
].values[0]

st.subheader("Automatic Interpretation")

st.write(
    f"Lowest RMSE: {lowest_rmse_model['Model']} "
    f"({lowest_rmse_model['RMSE']:.2f})"
)

st.write(
    f"Highest R²: {highest_r2_model['Model']} "
    f"({highest_r2_model['R2']:.3f})"
)

st.write(
    f"Lasso kept {lasso_vars} variables."
)

st.write(
    f"Elastic Net kept {elastic_vars} variables."
)

if elastic_vars < ridge_vars:
    st.write(
        "Elastic Net selected fewer variables than Ridge."
    )
else:
    st.write(
        "Elastic Net did NOT select fewer variables than Ridge."
    )

if ridge_vars == n_features:
    st.write(
        f"Ridge kept all {n_features} variables."
    )
else:
    st.write(
        f"Ridge kept {ridge_vars} variables."
    )

# ---------------------------------------------------
# RMSE PLOT
# ---------------------------------------------------

st.subheader("RMSE Comparison")

fig1, ax1 = plt.subplots(figsize=(7, 4))

ax1.bar(
    results_df["Model"],
    results_df["RMSE"]
)

ax1.set_ylabel("Test RMSE")

st.pyplot(fig1)

# ---------------------------------------------------
# R2 PLOT
# ---------------------------------------------------

st.subheader("R² Comparison")

fig2, ax2 = plt.subplots(figsize=(7, 4))

ax2.bar(
    results_df["Model"],
    results_df["R2"]
)

ax2.set_ylabel("Test R²")

st.pyplot(fig2)

# ---------------------------------------------------
# VARIABLE SELECTION PLOT
# ---------------------------------------------------

st.subheader("Variables Kept")

fig3, ax3 = plt.subplots(figsize=(7, 4))

ax3.bar(
    results_df["Model"],
    results_df["Nonzero Coefficients"]
)

ax3.set_ylabel("Number of Variables")

st.pyplot(fig3)

# ---------------------------------------------------
# COEFFICIENT PLOT
# ---------------------------------------------------

st.subheader("Coefficient Comparison")

top_features = coef_df.abs().max(axis=1)\
    .sort_values(ascending=False)\
    .head(15)\
    .index

fig4, ax4 = plt.subplots(figsize=(12, 5))

coef_df.loc[top_features].plot(
    kind="bar",
    ax=ax4
)

ax4.set_ylabel("Coefficient Value")

plt.tight_layout()

st.pyplot(fig4)

# ---------------------------------------------------
# COEFFICIENT TABLE
# ---------------------------------------------------

st.subheader("Coefficient Table")

st.dataframe(coef_df)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.caption(
    "Author: Feda Bashbishi fbashbis@uwaterloo.ca"
)