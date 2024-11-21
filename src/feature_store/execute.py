#%% 
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy import exc
import datetime
from tqdm import tqdm
import argparse
#%%
ORIGIN_ENGINE = sa.create_engine("sqlite:///../../data/database.db")
TARGET_ENGINE = sa.create_engine("sqlite:///../../data/feature_store.db")
# %%
def import_query(path):
    with open(path, 'r') as open_file:
        return open_file.read()
    
def date_range(start, stop):
    dt_start = datetime.datetime.strptime(start, '%Y-%m-%d')
    dt_stop = datetime.datetime.strptime(stop, '%Y-%m-%d')
    dates = []
    while dt_start <= dt_stop:
        dates.append(dt_start.strftime("%Y-%m-%d"))
        dt_start += datetime.timedelta(days=1)
    return dates

def ingest_date(query, table, dt):
    query_fmt = query.format(date=dt)

    # Fetch data from the origin engine
    with ORIGIN_ENGINE.connect() as connection:
        df = pd.read_sql(text(query_fmt), connection)

    # Delete existing records and insert new data into target engine
    with TARGET_ENGINE.connect() as connection:
        try:
            delete_query = f"DELETE FROM {table} WHERE dtRef = :date"
            connection.execute(sa.text(delete_query), {"date": dt})
        except exc.OperationalError as err:
            print("Inexisting table, creating new table...")
        
    # Insert data into the target table
    df.to_sql(table, TARGET_ENGINE, index=False, if_exists='append')
# %%
now = datetime.datetime.now().strftime("%Y-%m-%d")

parser = argparse.ArgumentParser()
parser.add_argument("--feature_store", "-f", help="Nome da feature Store", type=str)
parser.add_argument("--start", "-s", help="Data de início", default=now, type=str)
parser.add_argument("--stop", "-p", help="Data de fim", default=now, type=str)
args = parser.parse_args()
# %%
query = import_query(f"{args.feature_store}.sql")
dates = date_range(args.start, args.stop)

#%%
for i in tqdm(dates):
    ingest_date(query, args.feature_store, i)
#%%
#%%
with TARGET_ENGINE.connect() as connection:
    result = connection.execute(sa.text("SELECT name FROM sqlite_master WHERE type='table';"))
    tables = [row[0] for row in result]
    print("Existing tables:", tables)
# %%
