#working search algorithm for a piece of Wikipedia using Apache Spark and mllib that is scalable
from pyspark import SparkConf, SparkContext 
#libraries required to create Spark Environment in python
from pyspark.mllib.feature import HashingTF #to compute term frequencies in document
from pyspark.mllib.feature import IDF #to compute inverse document frequencies

# creating spark environment to create our initial RDD
conf = SparkConf().setMaster("local").setAppName("SparkTFIDF")
sc = SparkContext(conf = conf)

rawData = sc.textFile("subset-small.tsv") #using spark context to create an RDD from file
'''
tsv file contains tab seperated values  representing small sample of Wikipedia article  
our rawData RDD contains each document/Wikipedia article in each line 
Each row is split up into tabular fields with various bits of metadata about article.
'''

fields = rawData.map(lambda x: x.split("\t"))
'''
Splitting the document metadata based on their tab limiters into a Python list 
create a new fields RDD that instead of raw input data contains Python lists of each field
'''
documents = fields.map(lambda x: x[3].split(" "))
# extracting the field - 3 (contains body of article) seperated by space 
# mapping our data (fields list) to documents RDD which now contains list of words 

 
# Extracting and storing the document names from field-1 for printing results:
documentNames = fields.map(lambda x: x[1])


hashingTF = HashingTF(100000)
'''
Mapping words to numbers
creating HashingTF object with 100K as parameter to hash every word into one of 
those 100K numerical values instead of representing them as string viz. inefficient
it will try to distribute each word to a unique hash value as evenly as possible
'''
tf = hashingTF.transform(documents)
'''
transforming our documents RDD with out hashingTF RDD
(taking list of word from article body and converting it to a list of hash values)

RDD of sparse vectors representing each document, more memory efficient 
any missing data is stripped out, we are not storing missing words in the document
each unique hash value is mapped to its term frequency 
'''

# computing the TF*IDF score of each term in each document
tf.cache()# we cache the tf RDD because of multiple use and avoiding recomputing
idf = IDF(minDocFreq=2).fit(tf)
#Parameter : minDocFreq = 2 ignore any word that doesnt appear at least twice in the document
#idf called and scaled to fit the tf dimensions
tfidf = idf.transform(tf) #TF-IDF score cFomputed for each word in each document 

# RDD of sparse vectors, where each value is the TFxIDF of each unique hash value for each document.

# search for "Gettysburg" (Lincoln gave a famous speech there)

# figure out hash value "Gettysburg" 
gettysburgTF = hashingTF.transform(["Gettysburg"])# labelpoint of Gettysburg obtained
gettysburgHashValue = int(gettysburgTF.indices[0]) # hashvalue obtained by extracting the first index

#Extracting the TF*IDF score for Gettsyburg's hash value into  a new RDD for each document:
gettysburgRelevance = tfidf.map(lambda x: x[gettysburgHashValue])

#Using zip function to combine the tfidf score with the correspoinding document name 
zippedResults = gettysburgRelevance.zip(documentNames)

print("Best document for Gettysburg is:")
print(zippedResults.max())# Printing the document name with the maximum TF*IDF value
sc.stop()