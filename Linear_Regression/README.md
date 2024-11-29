# Linear Regression Using PySpark (ML API)
This project demonstrates the implementation of Linear Regression using PySpark's ML library, which is DataFrame-based. The code processes data from a text file, splits it into training and testing sets, and applies a linear regression model to predict outcomes.

**Project Overview**
1. SparkSession: Initializes the Spark environment using SparkSession, a unified entry point for Spark SQL and ML operations.
2. Data Preprocessing:
  - Reads a text file containing linearly correlated data (e.g., heights and weights).
  - Converts the data into a DataFrame with label and features columns, required by PySpark ML.
3. Model Training and Testing:
  - Splits the data into training and testing sets (50:50).
  - Trains a linear regression model using configurable parameters.

