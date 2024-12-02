#%% Import libraries and define necessary functions
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
from sklearn import tree
import matplotlib.pyplot as plt

# Define a function to assign lifecycle stages to users
def life_cycle(row):
    """
    Assigns a lifecycle stage to a user based on their recency and age in days.
    - row: A row from a DataFrame, expected to contain 'baseAgeDays' and 'recencyDays' columns.
    
    Returns:
        A string representing the lifecycle stage of the user.
    """
    if row['baseAgeDays'] <= 7:
        return 'New User'
    elif row['recencyDays'] <= 2:
        return 'Super Active User'
    elif row['recencyDays'] <= 6:
        return 'Active User'
    elif row['recencyDays'] <= 12:
        return 'Cold Active'
    elif row['recencyDays'] <= 18:
        return 'Unengaged'
    else:
        return 'Pre-Churn'

#%% Step 1: Connect to SQLite database and query data

# Create a connection to the SQLite database
engine = sa.create_engine("sqlite:///../../data/feature_store.db")

# SQL query to fetch the most recent data from the 'fs_general' table
query = '''
SELECT *
FROM fs_general
WHERE dtRef = (SELECT MAX(dtRef) FROM fs_general)
'''

# Execute the query and load the result into a DataFrame
with engine.connect() as connection:
    df = pd.read_sql(text(query), connection)

# Display the first few rows of the DataFrame
print("Preview of queried data:")
print(df.head())

#%% Step 2: Visualize the distribution of recency days

# Plot a histogram of the 'recencyDays' column to understand the distribution of user recency
df["recencyDays"].hist()
plt.xlabel("Recency (Days)")
plt.ylabel("Frequency")
plt.title("Histogram of Recency Days")
plt.show()

#%% Step 3: Analyze cumulative recency and assign lifecycle stages

# Prepare a DataFrame for cumulative recency analysis by selecting and sorting relevant columns
df_recency = df[["recencyDays", "baseAgeDays"]].sort_values(by="recencyDays")

# Add a column for counting each user as a unit
df_recency['unit'] = 1

# Calculate cumulative count of users by recency
df_recency['Cumulative'] = df_recency['unit'].cumsum()

# Compute cumulative percentage of users
df_recency["Cumulative Percentage"] = df_recency['Cumulative'] / df_recency['Cumulative'].max()

plt.plot(df_recency["recencyDays"], df_recency["Cumulative Percentage"], label="Cumulative Percentage")

# Define the recency thresholds and their corresponding lifecycle stages
lifecycle_stages = {
    2: "Super Active User",
    6: "Active User",
    12: "Cold Active",
    18: "Unengaged",  # 'Unengaged' will be on the left of the last line
}

# Add vertical lines at specific recencyDays thresholds and label with the lifecycle stages
for i, (recency, stage) in enumerate(lifecycle_stages.items()):
    # Place the text to the left of the line for the first 3 stages
    if i < len(lifecycle_stages) - 1:
        plt.axvline(x=recency, color='red', linestyle='--')
        plt.text(recency - 0.2, 0.9, stage, rotation=0, color='red', ha='right', va='top', fontsize=10)
    # For the last stage (Unengaged and Pre-Churn), place Unengaged to the left and Pre-Churn to the right
    else:
        plt.axvline(x=recency, color='red', linestyle='--')
        plt.text(recency - 0.2, 0.9, "Unengaged", rotation=0, color='red', ha='right', va='top', fontsize=10)
        plt.text(recency + 0.2, 0.9, "Pre-Churn", rotation=0, color='red', ha='left', va='top', fontsize=10)

# Adding labels and title
plt.xlabel("Recency (Days)")
plt.ylabel("Cumulative Percentage")
plt.title("Recency segmentation")
plt.grid(True)

# Show the plot
plt.show()
# Assign lifecycle stages to users based on their recency and age
df_recency['LifeCycle'] = df_recency.apply(life_cycle, axis=1)

# Group the data by lifecycle stages and calculate summary statistics
result = df_recency.groupby('LifeCycle').agg({
    'recencyDays': ['mean', 'count'],  # Calculate mean and count of 'recencyDays'
    'baseAgeDays': ['mean'],          # Calculate mean of 'baseAgeDays'
})

# Display the grouped summary statistics
print("Summary statistics by lifecycle stages:")
print(result)

 #%% Step 4: Build and save a Decision Tree Classifier 

# Prepare the data for training
# Extract the means of 'recencyDays' and 'baseAgeDays' as features (X) and lifecycle stages as labels (y)


# Train a Decision Tree Classifier
clf = tree.DecisionTreeClassifier(min_samples_leaf=1, max_depth=None, random_state=42)
clf.fit(df_recency[['recencyDays', 'baseAgeDays']], df_recency['LifeCycle'])
# Save the trained model and feature names for future use
model = {
    "model": clf,
    "features": ['recencyDays', 'baseAgeDays']  # Use the column used in training
}

# Save to pickle
pd.to_pickle(model, '../../models/cluster_rf_01.pkl')

#%% Final: Verify results
# Display the first few rows of the grouped summary to ensure everything looks correct
print("First few rows of grouped lifecycle statistics:")
print(result.head())

# %%
