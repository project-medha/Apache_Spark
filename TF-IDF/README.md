# Wikipedia Search Algorithm Using Apache Spark and MLlib
This project implements a scalable search algorithm using Apache Spark's MLlib to process and analyze a small subset of Wikipedia articles. It computes the TF-IDF (Term Frequency-Inverse Document Frequency) of terms in the dataset to rank their importance and enables efficient search queries across the dataset.

**Features**
1. Data Processing:
  - Extracts and processes Wikipedia article content from a tab-separated values (TSV) file.
  - Splits metadata into individual fields for structured processing.
  - Identifies and hashes terms from article bodies for further analysis.
2. TF-IDF Computation:
  - Uses Spark's HashingTF to compute term frequencies (TF).
  - Applies IDF to weigh term frequencies by their inverse document frequency.
3. Scalability:
  - Built using Apache Spark, allowing for distributed computation on large datasets.
