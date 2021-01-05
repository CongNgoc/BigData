# import findspark
# findspark.init()
from functools import reduce
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark import SparkContext
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
import helper
import kmeans
from sklearn.cluster import KMeans
import pyspark.sql.functions as func
from pyspark.sql.functions import col

# DECLARE GLOBAL VARIABLE 
MOVIE_FILE = "../resources/movies_metadata.csv"
RATING_FILE = "../resources/ratings_small.csv"


# =========== LOAD DATA ==================
spark = SparkSession.builder \
        .master("local") \
        .appName("Big Data") \
        .config(conf=SparkConf()) \
        .getOrCreate()

movies = spark.read.format("csv") .option("header", "true") .load(MOVIE_FILE)
ratings = spark.read.format("csv") .option("header", "true") .load(RATING_FILE)

movies.createOrReplaceTempView("movie_data")
movies = spark.sql("select id, original_title, genres from movie_data")
movies.show()
print(movies.count())

ratings.show()
print(ratings.count())

expected_genres = ['Drama', 'Romance'];

expected_movies_Romance = movies.filter(movies.genres.contains("Romance")).where(~movies.genres.contains("Drama"))
                            
expected_movies_Drama = movies.filter(movies.genres.contains("Drama")).where(~movies.genres.contains("Romance"))

expected_movies_Drama.show()
print(expected_movies_Drama.count())

expected_movies_Romance.show()
print(expected_movies_Romance.count())

join_movies_Romance = expected_movies_Romance.join(ratings, expected_movies_Romance.id == 
    ratings.movieId).drop(expected_movies_Romance.id)
join_movies_Romance.show()
print(join_movies_Romance.count())

join_movies_Drama = expected_movies_Drama.join(ratings, expected_movies_Drama.id == 
    ratings.movieId).drop(expected_movies_Drama.id)
join_movies_Drama.show()
print(join_movies_Drama.count())

# Group by UserId
group_data_Romance = join_movies_Romance.groupBy(join_movies_Romance.userId).agg(func.round(func.avg("rating"), 2).alias("avg_Romance"))
group_data_Romance.show()
print(group_data_Romance.count())

group_data_Drama = join_movies_Drama.groupBy(join_movies_Drama.userId).agg(func.round(func.avg("rating"), 2).alias("avg_Drama"))
group_data_Drama.show()
print(group_data_Drama.count())

data = group_data_Romance.join(group_data_Drama, group_data_Romance.userId == group_data_Drama.userId).drop(group_data_Romance.userId).orderBy(col("userId").asc())
data.show()
print(data.count())

# ====== SHOW DATA ==========
helper.draw_scatterplot(data.select('avg_Drama').collect(),'Avg Drama rating', data.select('avg_Romance').collect(), 'Avg romance rating')

X = np.array(data.select('avg_Drama', 'avg_Romance').collect())

(centers, labels, it) = kmeans.kmeanAgs(X, 2)
print('Centers found by our algorithm:')
print(centers[-1])

kmeans.kmeans_display(X, labels[-1])


# ======= EBOW ===============
distortions = []
K = range(1,15)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X)
    distortions.append(kmeanModel.inertia_)

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# ====== CHOOSE K ==========
# possible_k_values = range(2, len(X)+1, 5)

# # Calculate error values for all k values we're interested in
# errors_per_k = [helper.clustering_errors(k, X) for k in possible_k_values]

# # Plot the each value of K vs. the silhouette score at that value
# fig, ax = plt.subplots(figsize=(16, 6))
# ax.set_xlabel('K - number of clusters')
# ax.set_ylabel('Silhouette Score (higher is better)')
# ax.plot(possible_k_values, errors_per_k)

# # Ticks and grid
# xticks = np.arange(min(possible_k_values), max(possible_k_values)+1, 5.0)
# ax.set_xticks(xticks, minor=False)
# ax.set_xticks(xticks, minor=True)
# ax.xaxis.grid(True, which='both')
# yticks = np.arange(round(min(errors_per_k), 2), max(errors_per_k), .05)
# ax.set_yticks(yticks, minor=False)
# ax.set_yticks(yticks, minor=True)
# ax.yaxis.grid(True, which='both')

# plt.show()




