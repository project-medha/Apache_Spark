from pyspark.mllib.clustering import KMeans
from numpy import array, random
#the mllib requires numpy arrays as input 
#random function is used to create dataset 
from math import sqrt
from pyspark import SparkConf, SparkContext #importing both objects
from sklearn.preprocessing import scale
#we use Scikit-learn with assumption that it is installed on every 
#station in cluster as it wont scale itself up like mllib 
K = 7
#global variable K is the no of clusters in our KMeans 

#Setting up Spark Environment 
#creating Spark configuration object with the app name SparkKMeans
conf = SparkConf().setMaster("local").setAppName("SparkKMeans")
#creating Spark environment using SparkContext object we use to create RDD that will run on local machine
sc = SparkContext(conf = conf)

#Create fake income/age clusters for N people in k clusters
def createClusteredData(N, k):
    random.seed(10)
    pointsPerCluster = float(N)/k
    X = []
    for i in range (k):
        incomeCentroid = random.uniform(20000.0, 200000.0)
        ageCentroid = random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([random.normal(incomeCentroid, 10000.0), random.normal(ageCentroid, 2.0)])
    X = array(X)
    return X

random.seed(0)

# Loading the fake data by parallelizing it into RDD created through "createClusteredData" function 
# normalizing it with scale() to make it comparable 
data = sc.parallelize(scale(createClusteredData(100, K)))

# Building the K-Means model (cluster the data)
clusters = KMeans.train(data, K, maxIterations=6,initializationMode="random")
#Parameters : data,no. of clusters, no. of iterations, runs, (puts upper bound to processing)
# default initialization - random  (it will randomly pick the initial centroids to our clusters)
# Printing out the cluster assignments
resultRDD = data.map(lambda point: clusters.predict(point)).cache()
# taking each datapoint from our data RDD using map
# lambda function transforming each point into cluster no. predicted by our model  
# using predict function and storing it into our resultRDD 

# importance of cache : in Spark if we call more than one function on RDD
# we cache it first  
print("Counts by value:")
counts = resultRDD.countByValue() 
# storing in RDD using countByValue which gives us how many points predicted in each cluster 
print(counts)

print("Cluster assignments:")
results = resultRDD.collect()
# calling collect function on resultRDD that shows cluster id assoiciated with each datapoint
print(results)


# Evaluate clustering accuracy by computing Within Set Sum of Squared Errors metric
def error(point):
    center = clusters.centers[clusters.predict(point)]
    #obtaining the centroid coordinates as per the cluster id assigned to the point 
    return sqrt(sum([x**2 for x in (point - center)]))
    #taking the euclidean distance of point from cluster centroid
WSSSE = data.map(lambda point: error(point)).reduce(lambda x, y: x + y)
#chaining different operations together 
#1. computing error by calling error function on every datapoint using lambda function 
#2. map function creates LabelPoint for each datapoint 
#3. we calling reduce on the LabelPoint to get final total of the error

print("Within Set Sum of Squared Error = " + str(WSSSE))
sc.stop()

'''
 Different values of K 
 K=5, WSSSE = 22.27; K=7, WSSSE = 20.18; K= 10 ,WSSSE = 40.
 
 Not normalizing the input data 
 data = sc.parallelize(createClusteredData(100,K))
 K=7, WSSSE= 552823.17 very erroronous model
 
 Different values of maxIterations 
 mI = 5 WSSSE = 20.29 ; mI = 6 WSSSE = 19.42;mI = 15 WSSSE = 31.5; 
 
'''