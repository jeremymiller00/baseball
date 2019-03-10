import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 2000)
pd.set_option('display.max_rows', 2000)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.ion()
# import pyspark as ps 
# from pyspark.ml.clustering import KMeans
# from pyspark.ml.evaluation import ClusteringEvaluator
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score


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

def silhouette_analysis(X, range_n_clusters = [2, 3, 4, 5, 6]):
    ''' Create silhouette plots to evaluate optimal number of clusters'''

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this   example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between    silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random   generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the  samples.
        # This gives a perspective into the density and separation of the   formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at  the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) /    n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on baseball seasonal totals with n_clusters = %d" % n_clusters),fontsize=14, fontweight='bold')

    plt.show()

# df.head()
##############################################################
if __name__ == "__main__":

    repo = '/Users/jeremymiller/GoogleDrive/Data_Science/Projects/Baseball/'

    # spark = (ps.sql.SparkSession.builder
    #     .appName("baseball_era")
    #     .getOrCreate()
    #     )

    # sc = spark.sparkContext

    # load data
    df = pd.read_csv(repo + "data/total_yearly.csv", index_col="yearID")
    # data = (spark.read.format("csv").
    #     option("header", "true").
    #     option("inferSchema", "true").
    #     load(repo + "data/total_yearly.csv"))

    # normalize by number of games for that year
    df_per_game = df.loc[::].div(df["games_per_year"], axis=0)

    # drop irrelevant columns
    df_per_game.drop(['W_p', 'L_p', 'GS_p', 'games_per_year', 'GS_f', 'PO_f', 'WP_f', 'ZR_f'], axis=1, inplace=True)

    X = df_per_game.values

    # determine init with lowest inertia
    evaluate_init()
    # kmeans ++ init has lowest inertia

    # determnine optimum K with silhouette analysis
    silhouette_analysis(X, range_n_clusters = [2, 3, 4, 5])
    # optimum k value is 2

    model = KMeans(n_clusters=2)
    model.fit(X)
    years = np.array(df_per_game.index)
    group0 = years[~model.labels_.astype(bool)]
    group1 = years[model.labels_.astype(bool)]
