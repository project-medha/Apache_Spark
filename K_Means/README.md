# K-Means Clustering using Pyspark 

This project demonstrates the implementation of the K-Means Clustering algorithm using PySpark's MLlib. The code generates synthetic data, clusters it into groups, and builds a K-Means model to identify the clusters.

**Features of the code**
1. Data Generation:
  - Synthetic data for income and age is generated using the random module.
  - Data is grouped into k clusters with random centroids.
2. Data Normalization:
  - The generated data is normalized using scale() from Scikit-learn for better comparability.
3. K-Means Clustering:
  - PySpark's MLlib KMeans module is used to cluster the normalized data into k clusters.
  - Configured with 7 clusters (K=7), 6 iterations, and random centroid initialization.
