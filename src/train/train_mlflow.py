#%%
import pandas as pd
import numpy as np
import sqlalchemy as sa
from sqlalchemy import text
import datetime
import mlflow
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble, model_selection, pipeline, preprocessing, metrics
from feature_engine import encoding
#%%

def import_query(path):
    with open(path, 'r') as open_file:
        return open_file.read()
    
query = import_query("abt.sql")

print(query)
#%%
# Create the engine
engine = sa.create_engine("sqlite:///../../data/feature_store.db")

# Use the engine connection to execute the query and load the result into a DataFrame
with engine.connect() as connection:
        df = pd.read_sql(text(query), connection)

# Show the first few rows of the DataFrame

# %%
## doing our splits
## out of time will be our model delocated by time, while train and test will be random samples
## because we want to apply this model to the future, its good to see how our model perform in a temporal space different to where it was trained
# its similar way to think like temporal series split , this will help dealing with the classification problem inside a time series evenvents.  


df_OOT = df[df['dtRef']==df['dtRef'].max()]
df_OOT.shape # we will have 3096 entries out of time entries, this is the last photo of our database
#%%
df_train = df[df['dtRef']<df['dtRef'].max()] # our train will be capped ultil the oot day
df_train
# %%
df_train['dtRef'].unique()
# %%
target = 'flChurn'
features = df_train.columns[3:].tolist()
features
#%%
X_train, X_test, y_train, y_test = model_selection.train_test_split(df_train[features],df_train[target],
                                                                    random_state=42,
                                                                    train_size=0.8, stratify=df_train[target])
# %%
print("Response rate in Train base: ", y_train.mean())
print("Response rate in Test base: ", y_test.mean())
# %% 
cat_features = X_train.dtypes[X_train.dtypes == 'object'].index.tolist()
num_features = list(set(features) - set(cat_features))
# %%
X_train[cat_features].describe()
#%%
X_train[num_features].describe().T
# %%

## we have zero features with null values, so we dont have to care about
## the null handling, if this means something, its means we did a great work
## when constructing our database :)
X_train[num_features].isna().sum().max()
#%%
## handling maxQuantityProduct categorical variable
## we have 3 levels, so one hot should work without increasing a lot the size of our dataset
## in this sense will go from 61 to 63 features

## Setting the experiment in mlflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080/")
mlflow.set_experiment(experiment_id=459048156521582659)
mlflow.autolog()

#%%
# Define OneHotEncoder

with mlflow.start_run():  
    onehot = preprocessing.OneHotEncoder(handle_unknown='ignore', drop='first')
    

    base_estimator = DecisionTreeClassifier(random_state=42)
    # Define the model
    model = ensemble.BaggingClassifier(base_estimator, random_state=42)

    # Define hyperparameter grid
    params = {
    "n_estimators": [10, 50, 100, 200, 500, 1000],  # Larger range of base estimators
    "max_samples": [0.2, 0.5, 0.7, 0.9, 1.0],       # Broader fractions of samples to draw
    "max_features": [0.2, 0.5, 0.7, 0.9, 1.0],      # Broader fractions of features to draw
    "bootstrap": [True, False],                     # Keep both options
    "bootstrap_features": [True, False]             # Keep both options
    }

    # Initialize the tqdm progress bar
    param_grid = list(model_selection.ParameterGrid(params))
    n_iterations = len(param_grid)
    best_score = -np.inf
    best_params = None

    # Initialize progress bar
    with tqdm(total=n_iterations, desc="Grid Search Progress", ncols=100) as pbar:
        # Iterate over each combination of parameters
        for param_set in param_grid:
            # Set the model parameters
            model.set_params(**param_set)

            # Define the pipeline
            model_pipeline = pipeline.Pipeline([
                ('OneHotEncoder', onehot),
                ('Model', model)
            ])

            # Perform cross-validation
            scores = model_selection.cross_val_score(
                model_pipeline, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1
            )
            mean_score = np.mean(scores)

            # Update the best score and parameters if necessary
            if mean_score > best_score:
                best_score = mean_score
                best_params = param_set

            # Update the progress bar
            pbar.update(1)

    # Print the best parameters and score
    print("Best Hyperparameters:", best_params)
    print("Best Cross-Validation ROC AUC Score:", best_score)

    # Refit the model with the best parameters
    model.set_params(**best_params)
    model_pipeline = pipeline.Pipeline([
        ('OneHotEncoder', onehot),
        ('Model', model)
    ])
    model_pipeline.fit(X_train, y_train)
# Make predictions
    y_train_proba = model_pipeline.predict_proba(X_train)
    y_test_proba = model_pipeline.predict_proba(X_test)
    y_oot_proba = model_pipeline.predict_proba(df_OOT[features])

    # Define a function to report metrics
    def report_metrics(y_true, y_proba, base, threshold=0.5):
        y_pred = (y_proba[:, 1] > threshold).astype(int)
        acc = metrics.accuracy_score(y_true, y_pred)
        auc = metrics.roc_auc_score(y_true, y_proba[:, 1])
        precision = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred)

        return {
            f'{base}Accuracy': acc,
            f'{base}ROC AUC': auc,
            f'{base}Precision': precision,
            f'{base}Recall': recall
        }

    report = {}
    report.update(report_metrics(y_train, y_train_proba, 'Train'))
    report.update(report_metrics(y_test, y_test_proba, 'Test'))
    report.update(report_metrics(df_OOT[target], y_oot_proba, 'OTT'))
    
    mlflow.log_metrics(report)
#%%

model_series = pd.Series({"model" : model_pipeline,
                          "features": features, 
                          "metrics" : report,
                          "dt_train" : datetime.datetime.now()})

model_series.to_pickle("../../models/rf_01.pkl")
## Performance enhancement:
## Best Etl
## Add or Remove variables
## Use mean encoding
## Test other models
# %%