from __future__ import print_function

from pyspark.ml.regression import LinearRegression
#importing ML instead of MLLib as the former is dataframe based API
from pyspark.sql import SparkSession
# Using SparkSession object from Spark Sql interface to initiate Spark environment 
# (instead of Spark Context) 
from pyspark.ml.linalg import Vectors
#Vectors object required as we need dense vector as feature input in dataframes

if __name__ == "__main__":
    
    spark = SparkSession.builder.config("spark.sql.warehouse.dir", "file:///C:/tmp").appName("LinearRegression").getOrCreate()
    '''
    Creating a SparkSession, and storing it as directory in C:/tmp folder, giving it an app name 
    Using getOrCreate() function so that the session directory can act as checkpoint for recovery
    and restart in case of unexpected session termination. 
    '''
    # Load up our data and convert it to the format ML expects.
    inputLines = spark.sparkContext.textFile("regression.txt")
    #loading textfile containing linearly correlated data in two columns 
    #with comma as delimiter and arbitrary representation may be heights and weights 
    # No scaling/normalization needed as the data is normalized 
    data = inputLines.map(lambda x: x.split(",")).map(lambda x: (float(x[0]), Vectors.dense(float(x[1]))))
    #Splitting our data RDD through delimiter mapping target as floating point column and features as dense vector 
    #the above format is the requirement for dataframe in MLLib 
    colNames = ["label", "features"]
    df = data.toDF(colNames)
    # converting data RDD to dataframe using toDF() function and assigning names to the fields of data RDD 
    '''
    the input data frame contains two columns,label and features, where the label is a floating point
    height, and the features column is a dense vector of floating point weights
    '''


    # Splitting our dataframe into training data and testing data 50:50 
    trainTest = df.randomSplit([0.5, 0.5])
    trainingDF = trainTest[0] # the training data is in field- [0] of dataframe 
    testDF = trainTest[1]# the testing data is in field- [1] of dataframe 

    lir = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    #creating our linear regression model 

    # Training the model using our training data 
    model = lir.fit(trainingDF)

    fullPredictions = model.transform(testDF).cache()
    '''
    Generating predictions from our model on test dataframe
    it adds 'prediction' column along with 'features' and 'label' column of dataframe
    using cache to hold the predictions to prevent recomputation 
    '''
    predictions = fullPredictions.select("prediction").rdd.map(lambda x: x[0])
    labels = fullPredictions.select("label").rdd.map(lambda x: x[0])
    # Extracting the predictions and the "known" correct labels 
    # from the fullPredictions by transforming dataframe 
    # using select function and storing them as RDDs 
    
    predictionAndLabel = predictions.zip(labels).collect()
    #zipping the predictions and labels rdd together  
    for prediction in predictionAndLabel:
      print(prediction)
    # Printing out the predicted and actual values for each point


    # Stop the session
    spark.stop()
