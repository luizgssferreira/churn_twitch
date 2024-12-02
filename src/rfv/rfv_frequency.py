#%% Imports and Initial Setup
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import tree
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

#%% Step 1: Load Data from the Database

# Create a connection to the SQLite database
engine = sa.create_engine("sqlite:///../../data/feature_store.db")

# SQL query to fetch the most recent data from the 'fs_general' table
query = '''
    SELECT *
    FROM fs_general
    WHERE dtRef = (SELECT MAX(dtRef) FROM fs_general)
    '''

# Load the result into a pandas DataFrame
with engine.connect() as connection:
    df = pd.read_sql(text(query), connection)

# Display the first few rows of the DataFrame to verify the data
print("Preview of loaded data:")
print(df.head())

#%% Step 2: Scatter Plot of Points vs Frequency

plt.figure(dpi=400)  # High DPI for better resolution
sns.set_theme(style='white')  # Use a clean theme

# Scatter plot to visualize the relationship between points and frequency
sns.scatterplot(
    data=df,
    x="pointsValue",
    y="frequencyDays",
)

plt.title("RFV: Points (Value) vs. Frequency", fontsize=18, fontweight='bold')
plt.xlabel("Points Value")
plt.ylabel("Frequency Days")
plt.show()

#%% Step 3: Data Standardization

# Normalize 'pointsValue' and 'frequencyDays' using MinMaxScaler
scaler = MinMaxScaler()
X_trans = scaler.fit_transform(df[['pointsValue', 'frequencyDays']])

#%% Step 4: Apply Clustering

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(X_trans)

# Apply Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=4, linkage='ward')
agg_labels = agg_clustering.fit_predict(X_trans)

# Add cluster labels to the original DataFrame
df['kmeans_cluster'] = kmeans_labels
df['agg_cluster'] = agg_labels

#%% Step 5: Visualize KMeans Clustering Results

plt.figure(figsize=(10, 6), dpi=400)
sns.set_theme(style='whitegrid')

# Plot each KMeans cluster separately
for i in df['kmeans_cluster'].unique():
    data = df[df['kmeans_cluster'] == i]
    sns.scatterplot(
        data=data,
        x="pointsValue",
        y="frequencyDays",
        label=f'KMeans Cluster {i}'
    )

plt.title("KMeans Clustering of Points and Frequency", fontsize=16)
plt.xlabel("Points Value", fontsize=12)
plt.ylabel("Frequency Days", fontsize=12)
plt.legend(title="Clusters")
plt.show()

#%% Step 6: Visualize Agglomerative Clustering Results

# Plot Agglomerative clusters
for i in df['agg_cluster'].unique():
    data = df[df['agg_cluster'] == i]
    sns.scatterplot(
        data=data,
        x="pointsValue",
        y="frequencyDays",
        alpha=0.8,  # Slight transparency
    )

# Add reference lines for segmentation
line_color = "purple"
plt.hlines(8.5, xmin=0, xmax=4000, colors=line_color, linestyles='dashed', linewidth=1.5)
plt.hlines(2.5, xmin=0, xmax=4000, colors=line_color, linestyles='dashed', linewidth=1.5)
plt.hlines(13.5, xmin=0, xmax=4000, colors=line_color, linestyles='dashed', linewidth=1.5)
plt.vlines(500, ymin=0, ymax=23, colors=line_color, linestyles='dashed', linewidth=1.5)
plt.vlines(1400, ymin=0, ymax=23, colors=line_color, linestyles='dashed', linewidth=1.5)

# Customize title and labels
plt.title("RFV Segmentation: Points vs Frequency", fontsize=18, fontweight='bold')
plt.xlabel("Points Value", fontsize=14)
plt.ylabel("Frequency Days", fontsize=14)
plt.legend([], [], frameon=False)  # Remove legend
plt.tight_layout()
plt.show()

#%% Step 7: Analyze Cluster Volumes

# Count customers per cluster for both methods
print("KMeans Cluster Counts:")
print(df.groupby("kmeans_cluster")["idCustomer"].count())

print("\nAgglomerative Cluster Counts:")
print(df.groupby("agg_cluster")["idCustomer"].count())

#%% Step 8: Define RF Segmentation Rules

def rf_cluster(row):
    """
    Assign RF clusters based on pointsValue and frequencyDays using predefined thresholds.
    """
    if row['pointsValue'] < 500:
        if row['frequencyDays'] < 2.5:
            return "LL"  # Low Value, Low Frequency
        elif row['frequencyDays'] < 8.5:
            return "LM"  # Low Value, Medium Frequency
        elif row['frequencyDays'] < 13.5:
            return "LH"  # Low Value, High Frequency
        else:
            return "LV"  # Low Value, Very High Frequency

    elif row['pointsValue'] < 1400:
        if row['frequencyDays'] < 2.5:
            return "ML"  # Medium Value, Low Frequency
        elif row['frequencyDays'] < 8.5:
            return "MM"  # Medium Value, Medium Frequency
        elif row['frequencyDays'] < 13.5:
            return "MH"  # Medium Value, High Frequency
        else:
            return "MV"  # Medium Value, Very High Frequency

    else:
        if row['frequencyDays'] < 2.5:
            return "HL"  # High Value, Low Frequency
        elif row['frequencyDays'] < 8.5:
            return "HM"  # High Value, Medium Frequency
        elif row['frequencyDays'] < 13.5:
            return "HH"  # High Value, High Frequency
        else:
            return "HV"  # High Value, Very High Frequency

# Apply RF segmentation rules
df['rf_cluster'] = df.apply(rf_cluster, axis=1)

#%% Step 9: Visualize RF Clusters

plt.figure(dpi=400)  # High DPI for clarity

# Scatter plot for each RF cluster
for i in df['rf_cluster'].unique():
    data = df[df['rf_cluster'] == i]
    sns.scatterplot(
        data=data,
        x="pointsValue",
        y="frequencyDays",
        alpha=0.8,
        label=i,
    )

# Customize plot
plt.title("Cluster Analysis: Frequency vs Value", fontsize=16, fontweight="bold")
plt.xlabel("Points Value", fontsize=14)
plt.ylabel("Frequency Days", fontsize=14)

# Legend customization
plt.legend(
    title="RF Clusters",
    title_fontsize=12,
    fontsize=10,
    loc='lower right',
    ncol=2,
    frameon=True,
)
plt.show()

#%% Step 10: Train Decision Tree Model

# Train a Decision Tree Classifier to predict RF clusters
clf = tree.DecisionTreeClassifier(random_state=42, min_samples_leaf=1, max_depth=None)
clf.fit(df[['frequencyDays', 'pointsValue']], df['rf_cluster'])

# Save the model and features for future use
model_freq_value = pd.Series(
    {
        "model": clf,
        "features": ['frequencyDays', 'pointsValue']
    }
)

model_freq_value.to_pickle('../../models/cluster_fv_01.pkl')

# End of script

# %%
cluster_counts = df['rf_cluster'].value_counts()

# Display the results
print(cluster_counts)
#%%
