import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 2000)
pd.set_option('display.max_rows', 2000)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.ion()
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

def kMeans_silhouette_analysis(X, range_n_clusters = [2, 3, 4, 5, 6], 
                                scale_data=True, with_PCA=True):
    ''' Create silhouette plots to evaluate optimal number of clusters'''

    if scale_data:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    if with_PCA:
        pca = PCA()
        X = pca.fit_transform(X)

    st.header("KMeans: Explore possible values for K")
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
        plt.suptitle(("Silhouette Analysis for KMeans Clustering on Baseball Season Totals, %d Clusters" % n_clusters),fontsize=14, fontweight='bold')
        figs.append(fig)

    eras = st.slider("Choose Value for K", 
                min_value=2, max_value=10, value=6, step=1)
    st.write(scores[eras-2])
    st.write(figs[eras-2])

    elbow_fig, elbow_ax = plt.subplots()
    elbow_ax.plot(range_n_clusters, raw_scores)
    elbow_ax.grid()
    elbow_ax.set_title("K-Means Clustering\nSilhouette Score for Each Value of K")
    elbow_ax.set_xlabel("K: Number of Clusters")
    elbow_ax.set_ylabel("Average Silhouette Score")
    if st.checkbox("Show Full Y-Axis"):
        elbow_ax.set_ylim([0,1])
    st.pyplot(elbow_fig)


def dbscan(X, scale_data=True, with_PCA=True):
    
    if scale_data:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    if with_PCA:
        pca = PCA()
        X = pca.fit_transform(X)

    st.header("DBSCAN: Explore possible values for epsilon")
    epsilons = []
    n_noise = []
    sil_scores = []
    scores = []
    figs = []
    
    for e in np.linspace(0.00001, 4, num=20):
        epsilons.append(e)
        db = DBSCAN(eps=e, min_samples=3).fit(X)
        n_noise.append( (db.labels_ == -1).sum() )
        n_clusters = np.max(db.labels_ + 1)
        try:
            silhouette_avg = silhouette_score(X, db.labels_)

        except:
            silhouette_avg = 0

        sil_scores.append(silhouette_avg)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax2.grid()
        fig.set_size_inches(10, 4)
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
        scores.append(f"For n_clusters = {n_clusters}, The average silhouette_score is : {silhouette_avg}")
        try:
            sample_silhouette_values = silhouette_samples(X, db.labels_)
        except:
            continue
        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = \
                sample_silhouette_values[db.labels_ == i]

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
        colors = cm.nipy_spectral(db.labels_.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
        
        # centers = np.zeros(shape=(len(db.labels_), X.shape[1]))
        # for i, l in enumerate(db.labels_):
        #     centers[i] = np.random.choice(X[l])
        # ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
        #             c="white", alpha=1, s=200, edgecolor='k')
        # for i, c in enumerate(centers):
        #     ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
        #                 s=50, edgecolor='k')
                        
        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
        plt.suptitle((f"Silhouette Analysis for DBSCAN e={e:.2f} on Baseball Season Totals, {n_clusters} Clusters"),fontsize=14, fontweight='bold')
        figs.append(fig)


    st.header("Choose Value of Epsilon")
    st.subheader("Maximum Possible Distance Between Points in the Same Cluster")
    ep = st.selectbox("", options=epsilons, index=10, 
                        format_func=lambda x: round(x, 2))

    epidx = epsilons.index(ep)
    st.write(scores[epidx])
    st.write(figs[epidx])

    fig, (sil_ax, noise_ax) = plt.subplots(1,2, figsize=(10,5))
    sil_ax.plot(epsilons, sil_scores)
    sil_ax.grid()
    sil_ax.set_title("DBSCAN: Silhouette Scores\nFor Values of Epsilon")
    sil_ax.set_xlabel("Epsilon")
    sil_ax.set_ylabel("Silhouettes Score")
    noise_ax.plot(epsilons, n_noise)
    noise_ax.grid()
    noise_ax.set_title("DBSCAN: Number of Noise Points\nFor Values of Epsilon")
    noise_ax.set_xlabel("Epsilon")
    noise_ax.set_ylabel("Number of Noise Points Out of 149")
    plt.tight_layout()
    st.pyplot(fig)

    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise_ = list(labels).count(-1)
    # st.write('Estimated number of clusters: %d' % n_clusters_)
    # st.write('Estimated number of noise points: %d' % n_noise_)
    # st.write("Silhouette Coefficient: %0.3f" % silhouette_score(X, labels))

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

    # Page Setup
    st.title("Empirically Defining Baseball Eras")
    st.write("Let's use Data to define what an 'era' should be in the history of baseball. We can try some common clustering techniques, such as K-Means, DBSCAN, etc.")
    st.write("Here is some text explaining the data and where they come from.")

    # app stuff
    sel = st.sidebar.selectbox("Choose a clustering method.", 
                                ("KMeans", "DBSCAN"))
    if sel == "KMeans":
        kMeans_silhouette_analysis(annual_data_grouped_per_game, 
                                    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10])
    elif sel == "DBSCAN":
        dbscan(annual_data_grouped_per_game)

    
    st.header("Explore the Data")
    if st.checkbox("Chart stats over time."):
        df_line_chart(annual_data_grouped_per_game, default=['HR', 'SO'])
    if st.checkbox("Show raw data."):
        st.dataframe(annual_data_grouped_per_game)

##############################################################
if __name__ == "__main__":
    main()
    