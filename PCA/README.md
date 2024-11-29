# TF-IDF and PCA Using PySpark MLlib

This project demonstrates how to compute TF-IDF (Term Frequency-Inverse Document Frequency) and apply Principal Component Analysis (PCA) to text data using PySpark's MLlib. It involves processing a dataset of documents, extracting meaningful features, and reducing dimensionality for analysis.

**Project Overview**
1. TF-IDF Computation:
  - Converts textual data into numeric feature vectors using hashing for term frequency (TF).
  - Computes the inverse document frequency (IDF) to weight the term frequencies.
2. Principal Component Analysis (PCA):
  - Applies PCA to the TF-IDF vectors to reduce dimensionality to two principal components.
3. Dataset:
  - A tab-separated file (subset-small.tsv) where the fourth column contains text data, and the second column holds document names.
