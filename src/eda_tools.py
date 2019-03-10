'''
A suite of functions to make eda plots with baseball data
'''
import matplotlib.pyplot as plt
import pandas as pd

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
