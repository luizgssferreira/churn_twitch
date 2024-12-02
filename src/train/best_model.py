# %% Import libraries
import os
import pandas as pd
import mlflow
import mlflow.sklearn
import sqlalchemy as sa
from sqlalchemy import text
import scikitplot as skplt
import matplotlib.pyplot as plt

# Create a directory to save the plots
os.makedirs("plots", exist_ok=True)

# %% Load SQL query and data
def import_query(path):
    """Load SQL query from a file."""
    with open(path, 'r') as file:
        return file.read()

query = import_query("abt.sql")
engine = sa.create_engine("sqlite:///../../data/feature_store.db")

# Execute SQL query and load data into a DataFrame
with engine.connect() as connection:
    df = pd.read_sql(text(query), connection)

# Prepare training and out-of-time (OOT) data
df_OOT = df[df['dtRef'] == df['dtRef'].max()]
df_train = df[df['dtRef'] < df['dtRef'].max()]

target = 'flChurn'  # Target variable: 1 = Churn, 0 = Non-Churn
exclude_columns = ['flChurn', 'dtRef', 'idCustomer']  # Exclude these columns
features = [col for col in df_train.columns if col not in exclude_columns]

# %% Load pre-trained model from MLflow
model = mlflow.sklearn.load_model("models:/Churn-Teo-Me-Why/production")
model_info = mlflow.models.get_model_info("models:/Churn-Teo-Me-Why/production")
print(f"Loaded model info: {model_info}")

# Predict probabilities for evaluation
y_oot_proba = model.predict_proba(df_OOT[features])


#%% Checking Model Metrics

#%%
# %% Plot evaluation curves for OOT dataset
# Precision-Recall Curve
plt.figure(figsize=(8, 6))
skplt.metrics.plot_precision_recall_curve(df_OOT[target], y_oot_proba, title="Precision-Recall Curve (Class 1 = Churn) - OOT")
plt.savefig("plots/precision_recall_curve_oot.png")
plt.show()

# ROC Curve
plt.figure(figsize=(8, 6))
skplt.metrics.plot_roc_curve(df_OOT[target], y_oot_proba, title="ROC Curve (Class 1 = Churn) - OOT")
plt.savefig("plots/roc_curve_oot.png")
plt.show()

# Cumulative Gain Curve
plt.figure(figsize=(8, 6))
skplt.metrics.plot_cumulative_gain(df_OOT[target], y_oot_proba, title="Cumulative Gain Curve (Class 1 = Churn) - OOT")
plt.savefig("plots/cumulative_gain_curve_oot.png")
plt.show()

# Prepare OOT probabilities for lift and additional analysis
oot_proba = pd.DataFrame({
    "true": df_OOT[target],
    "proba": y_oot_proba[:, 1]
}).sort_values("proba", ascending=False)
oot_proba["sum_true"] = oot_proba["true"].cumsum()

# Lift Curve
plt.figure(figsize=(8, 6))
skplt.metrics.plot_lift_curve(df_OOT[target], y_oot_proba, title="Lift Curve (Class 1 = Churn) - OOT")
plt.savefig("plots/lift_curve_oot.png")
plt.show()

# Analyze top 100 OOT users by churn probability
top_100_churn_rate = oot_proba.head(100)["true"].mean()
print(f"Top 100 churn rate in OOT: {top_100_churn_rate:.2f}")

# Compare to baseline churn rate
baseline_churn_rate = oot_proba["true"].mean()
print(f"Baseline churn rate in OOT: {baseline_churn_rate:.2f}")

# KS Statistic
plt.figure(figsize=(8, 6))
skplt.metrics.plot_ks_statistic(df_OOT[target], y_oot_proba, title="KS Statistic (Class 1 = Churn) - OOT")
plt.savefig("plots/ks_statistic_oot.png")
plt.show()

# %%
