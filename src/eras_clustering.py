import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 2000)
pd.set_option('display.max_rows', 2000)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.ion()
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
import streamlit as st


def set_constants():
    repo = '/Users/Jeremy/GoogleDrive/Data_Science/Projects/Baseball/'
    return repo

def evaluate_init(n_runs=5):
    ''' Try two Kmeans models with two initiation methods to determine best baseline clusterer.'''
    # k-means models can do several random inits so as to be able to trade CPU time for convergence robustness
    n_init_range = np.array([1, 5, 10, 15, 20])

    # quantitative eval of various init methods
    plt.figure(figsize=(12,10))
    plots = []
    legends = []

    cases = [
        (KMeans, 'k-means++', {}),
        (KMeans, 'random', {}),
        (MiniBatchKMeans, 'k-means++', {'max_no_improvement': 3}),
        (MiniBatchKMeans, 'random', {'max_no_improvement': 3,   'init_size': 500}),
    ]

    for factory, init, params in cases:
        print("Evaluation of %s with %s init" % (factory.__name__, init))
        inertia = np.empty((len(n_init_range), n_runs))

        for run_id in range(n_runs):
            # X, y = make_data(run_id, n_samples_per_center, grid_size,   scale)
            for i, n_init in enumerate(n_init_range):
                km = factory(n_clusters=5, init=init,  random_state=run_id,n_init=n_init, **params).fit(X)
                inertia[i, run_id] = km.inertia_
        p = plt.errorbar(n_init_range, inertia.mean(axis=1), inertia.std(axis=1))
        plots.append(p[0])
        legends.append("%s with %s init" % (factory.__name__, init))

    plt.xlabel('n_init')
    plt.ylabel('inertia')
    plt.legend(plots, legends)
    plt.title("Mean inertia for various k-means init across %d runs" %  n_runs)
    plt.show()

def kMeans_silhouette_analysis(X, range_n_clusters = [2, 3, 4, 5, 6], scale_data=True):
    ''' Create silhouette plots to evaluate optimal number of clusters'''

    if scale_data:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    figs = []
    raw_scores = []
    scores = []
    for n_clusters in range_n_clusters:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(10, 4)
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        raw_scores.append(silhouette_avg)
        scores.append(f"For n_clusters = {n_clusters}, The average silhouette_score is : {silhouette_avg}")
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        colors = cm.nipy_spectral(cluster_labels.astype(float) /    n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
        centers = clusterer.cluster_centers_
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')
                        
        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
        plt.suptitle(("Silhouette analysis for KMeans clustering on baseball seasonal totals with n_clusters = %d" % n_clusters),fontsize=14, fontweight='bold')
        figs.append(fig)

    elbow_fig, elbow_ax = plt.subplots()
    elbow_ax.plot(range_n_clusters, raw_scores)
    elbow_ax.grid()
    elbow_ax.set_title("Silhouette Score for Each Value of K")
    elbow_ax.set_xlabel("K: Number of Clusters")
    elbow_ax.set_ylabel("Average Silhouette Score")
    st.pyplot(elbow_fig)
    eras = st.slider("Choose Number of Eras", 
                min_value=2, max_value=10, step=1)
    st.write(scores[eras-2])
    st.write(figs[eras-2])
    return figs

@st.cache
def load_and_prep_data(repo: str):
    # load data
    column_mapper = {'W_p': 'Wins', 'L_p': 'Losses', 'CG_p': 'Complete Games',
                    'SHO_p': 'Shutouts', 'SV_p': 'Saves', 'H_p': 'Hits',
                    'ER_p': 'Earned Runs', 'HR_p': 'Home Runs', 'BB_p': 'Walks',
                    'SO_p': 'Strikeouts', 'IBB_p': 'Int Walks', 'WP_p': 'Wild Pitches',
                    'HBP_p': 'Hit by Pitch', 'BK_p': 'Balks', 'R_p': 'Runs',
                    'SH_p': 'Sac Hits', 'SF_p': "Sac Flies", 'GIDP_p': 'GIDP'}
    df = pd.read_csv(repo + "data/total_yearly.csv", index_col="yearID")
    df = df.rename(column_mapper, axis=1)
    # normalize by number of games for that year
    df_per_game = df.loc[::].div(df["games_per_year"], axis=0)
    # drop irrelevant columns
    # df_per_game.drop(['W_p', 'L_p', 'GS_p', 'games_per_year', 'GS_f', 'PO_f', 'WP_f', 'ZR_f'], axis=1, inplace=True)
    return df_per_game, df

@st.cache
def load_and_prep_lahman_teams_data(repo: str):
    df = pd.read_csv(repo + 'data/baseballdatabank-master/core/Teams.csv')
    grouped  = df.groupby('yearID').sum()
    grouped['ERA'] = grouped['ER'] / (grouped['IPouts'] / 27)
    grouped_per_game = grouped.copy().drop(["Rank", "Ghome"], axis=1)
    for c in grouped_per_game.columns:
        if c not in ['G', 'ERA']:
            grouped_per_game[c] = grouped_per_game[c] / grouped_per_game['G']
    return df, grouped.sort_index(ascending=False), grouped_per_game.sort_index(ascending=False)

def df_line_chart(df, default=None):
    features = st.multiselect("Choose Features", 
                list(df.columns), default=default)
    st.write("Values Through The Years: Per Game")
    st.line_chart(df[features], width=40)

def main():
    # setup
    repo = set_constants()
    annual_data, annual_data_grouped, annual_data_grouped_per_game = load_and_prep_lahman_teams_data(repo)

    # app stuff
    st.title("Empirically Defining Baseball Eras")
    st.write("Let's use Data to define what an 'era' should be in the history of baseball. We can try some common clustering techniques, such as K-Means, DBSCAN, etc.")
    st.write("Raw Data: Yearly Totals")
    if st.checkbox("Chart stats over time."):
        df_line_chart(annual_data_grouped_per_game, default=['HR', 'SO'])
    if st.checkbox("Show raw data."):
        st.dataframe(annual_data_grouped_per_game)
    st.write("Choose a clustering method.")
    kMeans_silhouette_analysis(annual_data_grouped_per_game, 
                                range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10])

##############################################################
if __name__ == "__main__":
    main()
    