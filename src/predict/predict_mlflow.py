#%%

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy import exc

import mlflow 
import mlflow.sklearn

import json
# %%

print("Loading the model...")

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080/")
mlflow.set_registry_uri(uri="http://127.0.0.1:8080/")

model = mlflow.sklearn.load_model("models:/Churn-Teo-Me-Why/production")
model_info = mlflow.models.get_model_info("models:/Churn-Teo-Me-Why/production")
# %%
print("Loading features...")
features = [i['name'] for i in json.loads(model_info._signature_dict['inputs'])]

#%%

print("Loading database for score...")
def import_query(path):
    with open(path, 'r') as open_file:
        return open_file.read()
    
query = import_query("../predict/etl.sql")

print(query)
#%%

print("Loading the engine...")
engine = sa.create_engine("sqlite:///../../data/feature_store.db")

# Use the engine connection to execute the query and load the result into a DataFrame
with engine.connect() as connection:
        df = pd.read_sql(text(query), connection)
# %%
df # 413 active users in our database
# %%

print("Making predictions...")
pred = model.predict_proba(df[features])# %%
proba_churn = pred[:,1]

print("Persisting data...")
df_predict = df[['dtRef', 'idCustomer']].copy()
df_predict['probaChurn'] = proba_churn.copy()

df_predict = (df_predict.sort_values("probaChurn", ascending = False)
                        .reset_index(drop=True))

df_predict
# %%
with engine.connect() as con:
    state = f"DELETE FROM tb_churn WHERE dtRef = '{df_predict['dtRef'].min()}';"
    try:
        state = sa.text(state)
        con.execute(state)
        con.commit()
    except exc.OperationalError as err:
        print("Inexisting table... Creating...")

df_predict.to_sql("tb_churn", engine, if_exists='append', index=False)

print("Table was created!")
# %%
print("All tasks done!")
#%%