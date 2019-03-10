import pandas as pd 
import pyspark as ps 
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


# data.withColumn("1B_b_pg", dataset["1B_b"] / dataset["games_per_year"])



# df.head()
##############################################################
if __name__ == "__main__":

    repo = '/Users/jeremymiller/GoogleDrive/Data_Science/Projects/Baseball/'

    spark = (ps.sql.SparkSession.builder
        .appName("baseball_era")
        .getOrCreate()
        )

    sc = spark.sparkContext

    # load data
    # df = pd.read_csv(repo + "data/total_yearly.csv")
    data = (spark.read.format("csv").
        option("header", "true").
        option("inferSchema", "true").
        load(repo + "data/total_yearly.csv"))