import pandas as pd 
import pyspark as ps 
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator





df.head()
##############################################################
if __name__ == "__main__":

    repo = '/Users/jeremymiller/GoogleDrive/Data_Science/Projects/Baseball/'

    # load data
    df = pd.read_csv(repo + "data/total_yearly.csv")