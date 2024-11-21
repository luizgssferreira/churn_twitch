#%% Step 1: Import Necessary Libraries and Load Pre-Trained Models

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
import datetime

# Load pre-trained models for RF and FV clustering and churn prediction
cluster_RF = pd.read_pickle("../../models/cluster_rf_01.pkl")
cluster_FV = pd.read_pickle("../../models/cluster_fv_01.pkl")
model_churn = pd.read_pickle("../../models/rf_01.pkl")

#%% Step 2: Function to Import SQL Queries from File

def import_query(path):
    """
    Reads and returns the content of a SQL file.
    
    Parameters:
        path (str): Path to the SQL file.
        
    Returns:
        str: The content of the SQL file as a string.
    """
    with open(path, 'r') as open_file:
        return open_file.read()

# Import the SQL query from the specified path
query = import_query("../predict/etl.sql")

# Display the imported query for verification
print("Imported SQL Query:")
print(query)

#%% Step 3: Load Data from Database

print("Loading the engine...")
# Create a connection to the SQLite database
engine = sa.create_engine("sqlite:///../../data/feature_store.db")

# Execute the imported SQL query and load the result into a pandas DataFrame
with engine.connect() as connection:
    df = pd.read_sql(text(query), connection)

# Display the resulting DataFrame to verify successful query execution
print("Preview of loaded data:")
print(df.head())
print(f"Total active users in database: {df.shape[0]}")  # Number of rows indicates active users

#%% Step 4: Generate Predictions Using Pre-Trained Models

# Predict churn probability using the pre-trained churn model
df['probaChurn'] = model_churn['model'].predict_proba(df[model_churn['features']])[:, 1]

# Predict RF cluster using the pre-trained RF clustering model
df['clusterRF'] = cluster_RF['model'].predict(df[cluster_RF['features']])

# Predict FV cluster using the pre-trained FV clustering model
df['clusterFV'] = cluster_FV['model'].predict(df[cluster_FV['features']])

# Map clusterFV short codes to extensive names
def map_fv_cluster(cluster):
    """
    Maps short cluster codes for FV clusters to meaningful names.
    
    Parameters:
        cluster (str): Short cluster code (e.g., 'LL', 'LM', etc.).
        
    Returns:
        str: Cluster name based on value and frequency.
    """
    cluster_mapping = {
        "LL": "Low Value, Low Frequency",
        "LM": "Low Value, Medium Frequency",
        "LH": "Low Value, High Frequency",
        "LV": "Low Value, Very High Frequency",
        "ML": "Medium Value, Low Frequency",
        "MM": "Medium Value, Medium Frequency",
        "MH": "Medium Value, High Frequency",
        "MV": "Medium Value, Very High Frequency",
        "HL": "High Value, Low Frequency",
        "HM": "High Value, Medium Frequency",
        "HH": "High Value, High Frequency",
        "HV": "High Value, Very High Frequency"
    }
    return cluster_mapping.get(cluster, "Unknown Cluster")

# Apply the mapping to the clusterFV column
df['clusterFV'] = df['clusterFV'].apply(map_fv_cluster)

#%% Step 5: Save Results to a New Table in the Database

# Reorder the columns as requested: dtRef, idCustomer, RF, FV, probaChurn
columns = ['dtRef', 'idCustomer', 'clusterRF', 'clusterFV', 'probaChurn']

df_final = df[columns].copy()
df_final['dtUpdate'] = datetime.datetime.now()

# Write the DataFrame with selected columns to the `customer_profile` table in the database
# Use 'replace' mode to overwrite the table if it already exists
df_final.to_sql(
    name='customer_profile',
    con=engine,
    index=False,
    if_exists='replace'
)

print("Customer profiles saved to the 'customer_profile' table in the database in the specified order.")

# End of script

# %%
