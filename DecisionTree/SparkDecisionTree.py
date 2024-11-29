
#import modules from mllib that will run across cluster 
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
#importing necessary modules from Spark which will create Spark environment to run our code
from pyspark import SparkConf, SparkContext
from numpy import array

#to create Spark environment using Spark Context we first create  configuration object 
conf = SparkConf().setMaster("local").setAppName("SparkDecisionTree")
#the setMaster("local") means that the executing machine is our system 
#the application name is SparkDecisionTree which will be visible on Spark Console 
sc = SparkContext(conf = conf)

# Some functions that convert our CSV input data into numerical
# features for each job candidate
def binary(YN):
    if (YN == 'Y'):
        return 1
    else:
        return 0

def mapEducation(degree):
    if (degree == 'BS'):
        return 1
    elif (degree =='MS'):
        return 2
    elif (degree == 'PhD'):
        return 3
    else:
        return 0

# Each row (in form of list) passed to function is stored in field parameter
#to a LabeledPoint that MLLib can use. All data must be numerical...
def createLabeledPoints(fields):
    #ALL THE feature information is extracted from field parameter 
    yearsExperience = int(fields[0])#convered string to int
    employed = binary(fields[1])#binary function being called which returns 1 if field[1] is 'Y' and 0 if 'N'
    previousEmployers = int(fields[2])
    educationLevel = mapEducation(fields[3])#function being called which assigns the numerical value as per the level of education stored in fields[3]
    topTier = binary(fields[4])
    interned = binary(fields[5])
    hired = binary(fields[6])
    #fields=[yearsExperience, employed,previousEmployers,educationLeve;,topTier,interned,Hired]
    return LabeledPoint(hired, array([yearsExperience, employed,
        previousEmployers, educationLevel, topTier, interned]))

#Load up our CSV file sc.textFile loaded every row into an RDD called rawData
rawData = sc.textFile("PastHires.csv")
header = rawData.first()
#Extracting the header line with the column names
rawData = rawData.filter(lambda x:x != header)
#Removing the header line column names from the rawdata using lambda inline function 
#which stores every row if it is not equal to header 
#this is a new RDD with header row filtered out

# Split each line row into a list based on the comma delimiters into individual fields in the list
csvData = rawData.map(lambda x: x.split(","))

# Convert these lists to LabeledPoints 
#the map function takes individual row from csvData passes it to createLabeledPoints function
# then stores it as LabeledPoint in trainingData RDD 
trainingData = csvData.map(createLabeledPoints)

# Create a test candidate, with 10 years of experience, currently employed,
# 3 previous employers, a BS degree, but from a non-top-tier school where
# he or she did not do an internship. 
testCandidates = [ array([10, 1, 3, 1, 0, 0])]
#creating RDD out of testCandidate to be fed into Spark using parallelize function 
testData = sc.parallelize(testCandidates)

# Train our DecisionTree classifier from MLLib using our data set
model = DecisionTree.trainClassifier(trainingData, numClasses=2,
                                     categoricalFeaturesInfo={1:2, 3:4, 4:2, 5:2},
                                     impurity='gini', maxDepth=5, maxBins=32)
# Parameters - NumClasses :indicates that we only have two classes that we're trying to sort people into whether or not they're hired, yes or no.
# categoricalFeaturesInfo: is an array indicating which fields in our array are categorical
# impurity : indicates gini is used 
# maxDepth : the no of levels in tree 
# maxBins : the no of training batches created for Decisiontree



# Now get predictions for our unknown candidates. (Note, you could separate
# the source data into a training set and a test set while tuning
# parameters and measure accuracy as you go!)
predictions = model.predict(testData)
print('Hire prediction:')
results = predictions.collect()
#calling predictions.collect to get answer from Spark
#it initiates Spark to construct an optimal way to put all the data and distributing on cluster (if it were present)
#then it starts predicting for our testData 
for result in results:
    print(result)

# We can also print out the decision tree itself:
print('Learned classification tree model:')
print(model.toDebugString())
#toDebugString method on the decision tree model that will allow us to understand what's going on inside
#of the decision tree and what decisions it's making based on what criteria.
