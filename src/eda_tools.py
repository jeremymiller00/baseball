'''
A suite of functions to make eda plots with baseball data
'''
import matplotlib.pyplot as plt
import pandas as pd
from pymongo import MongoClient


def plot_stats_over_time(year_index, df, stats_to_plot, games_per_year):
    '''
    Parameters:
    ----------
    Input: 
    year_index {list-like}: list of years covered by plot, generally 1871-2018
    df {dataframe}: Pandas dataframe with data to plot
    stats_to_plot {list}: list of stats (columns) to plot
    games_per_year {list-like}: list of the total number of mlb games played for each year described in plot

    Output:
    Plot
    '''


    fig, ax = plt.subplots(figsize=(20,10))
    for stat in stats_to_plot:

        ax.plot(year_index, df[stat]/ games_per_year, 
                label=stat, lw=3)
    
    ax.legend(prop={'size': 20})
    ax.set_title("Yearly Totals Per Game Over Time", fontsize=20)
    ax.set_xlabel("Year")
    plt.show


def _connect_mongo(host, port, username, password, db):
    """ A util for making a connection to mongo """

    if username and password:
        mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
        conn = MongoClient(mongo_uri)
    else:
        conn = MongoClient(host, port)

    return conn[db]


def read_database(db, query={}, host='localhost', port=27018, username=None, password=None, no_id=True): 
    """ Read from Mongo and Store into DataFrame 
    Typically port 27017 is used"""

    # Connect to MongoDB
    db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)

    dfs = []

    # Get list of collections
    for name in db.list_collection_names():
        # Make a query to the specific DB and Collection
        cursor = db[name].find(query)
        # Expand the cursor and construct the DataFrame
        df =  pd.DataFrame(list(cursor))
        # Delete the _id
        if no_id:
            del df['_id']
        dfs.append(df)
        
    return pd.concat(dfs)

def read_collection(db, collection, query={}, host='localhost', port=27018, username=None, password=None, no_id=True):
    """ Read from Mongo and Store into DataFrame 
    Typically port 27017 is used"""

    # Connect to MongoDB
    db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)

    # Make a query to the specific DB and Collection
    cursor = db[collection].find(query)

    # Expand the cursor and construct the DataFrame
    df =  pd.DataFrame(list(cursor))

    # Delete the _id
    if no_id:
        del df['_id']

    return df