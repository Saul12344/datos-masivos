<p align="center">
    <img alt="Logo" src="https://www.tijuana.tecnm.mx/wp-content/uploads/2021/08/liston-de-logos-oficiales-educacion-tecnm-FEB-2021.jpg" width=850 height=250>
</p>
<H2><p align="center">Unit#4 branch. Big Data course.</p></H2>
<H2><p align="Center">STUDENT'S NAMES: </p></H2>

<H2><p align="Center">López Higuera Saúl Alfredo #18210493</p></H2>

<H2><p align="Center">Ramos Rivera Manuel Isaí #17212931</p></H2>
<H2><p align="Center">LINK DEL VIDEO:https://www.youtube.com/watch?v=eLcprAHl5yA</p></H2>

# Final project

# Introduction.
This is the final evaluation of the subject of massive data, in this practice it is intended to see how effective the learning models are, comparing the memory they use, the time they take to execute and seeing their precision, this will be done with a file csv provided by the teacher that has more than 45,000 data.
The objective of the classification is to predict if the client subscribed a term deposit (variable y) and we try to make a prediction with the models: SVM, decision tree, logistic regression and multilayer perceptron that are machine learning models.


# Theoretical framework of algorithms.

- Decision Three
- Logistic Regression
- Multilayer perceptron
- Support Vector Machine (SVM)


# Support Vector Machine (SVM).

It is a discriminative classifier formally defined by a separation hyperplane. In other words, given the labeled training data (supervised learning), the algorithm generates an optimal hyperplane that categorizes new examples. In two-dimensional spaces, this hyperplane is a line that divides a plane into two parts where each class is on each side. That is used in many classification and regression problems, including medical applications of signal processing, natural language processing, and image and speech recognition. The idea of SVM is simple: the algorithm creates a line or hyperplane that separates the data into classes.

![logo](/Img/1.PNG)  

Advantages and disadvantages

Advantage:
- Classifier algorithm based on solid theory. Risk minimization theorems are the state of the art in statistical learning.
- Can be applied to data represented in any Hilbert space (where you can define a distance measure).
- Relatively few parameters to estimate.
- New extensions can be formulated (flexibility).

Disadvantages:
- Determining the kernels to use is complex.
- It is just a binary classifier.

# Decision three classifier (DTC).

A decision tree is a decision support tool that uses a graph or model of decisions in the form of a tree and their possible consequences, including outcomes of chance events, resource costs, and utility. It is a way of showing an algorithm that only contains conditional control statements. A decision tree is a flowchart-like structure in which each internal node represents a "test" on an attribute (for example, whether a coin turns heads or tails), each branch represents the outcome of the test, and each leaf node represents a class label (decision made after calculating all attributes). The root-to-leaf paths represent classification rules.

Advantages and disadvantages


Advantage:
- Decision trees can generate understandable rules.
- Decision trees perform classification without requiring many calculations.
- Decision trees can handle continuous and categorical variables.
- Decision trees provide a clear indication of which fields are most important for prediction or classification.

Disadvantages:

- Decision trees are less appropriate for estimation tasks where the goal is to predict the value of a continuous attribute.
- Decision trees are prone to errors in classification problems with many classes and a relatively small number of training examples.

# Logistic regression (LR).

It is a classification algorithm used to assign observations to a discrete set of classes. Some of the examples of classification problems are spam, fraudulent or non-fraudulent online transactions, benign or malignant tumor. Logistic regression transforms its output using the logistic sigmoid function to return a probability value. Logistic regression is a machine learning algorithm used for classification problems, it is a predictive analysis algorithm and it is based on the concept of probability.
The logistic regression model is based on the probabilities of an outcome of 2 levels of interest. For simplicity, I assume that we have designated one of the outcomes level the event of interest and in the following text we will simply call it the event.
The probability of the event is the ratio of the probability that the event will happen divided by the probability that the event will not happen.

![logo](/Img/2.PNG)  

## How is logistic regression used?
Logistic regression should only be used when the target variables fall into discrete categories and if there is a continuous range of values ​​that could be the target value, then logistic regression should not be used. Examples of situations where you might use logistic regression include:
- Predict whether an email is spam or not spam
- Whether a tumor is malignant or benign
- Whether a mushroom is poisonous or edible.

Advantages and disadvantages.

Advantage:
- Logistic regression works well when the data set is linearly separable L2.
- Logistic regression is less prone to overfitting, but overfitting in data sets can be high dimension so consider regularization techniques (L1 and ) to avoid overfitting in these scenarios.

Disadvantages:

- The main limitation of logistic regression is the assumption of linearity between the dependent variables and the independent variables. In the real world, data is rarely linearly separated. Most of the time the data would be a mess.
- If the number of observations is less than the number of features, then logistic regression should not be used, otherwise it may cause overfitting.

# Multilayer perceptron (MLPC).

(MLP) is a supervised learning algorithm that learns a function f (.): 𝑅 by training on 𝑚 → 𝑅 𝑂𝑟 a data set, where is the number of dimensions for input and is the number of dimensions for output. Given a feature set X = x1, x2, ..., xm and a target "y", you can learn an approximator of nonlinear functions for classification or regression. It is different from logistics. regression, in which between the input layer and the output layer, there may be one or more nonlinear layers, called hidden layers.

![logo](/Img/3.PNG)  

A nonlinear dynamical model for a process system, namely a heat exchanger, is developed using the recurrent multilayer perceptron network as the underlying model structure. The perceptron is a dynamic neural network, which appears effective in input-output modeling of complex process systems. Dynamic gradient descent learning is used to train the recurrent multilayer perceptron, resulting in an order-of-magnitude improvement in the speed of convergence in a static learning algorithm used to train the same network.

![logo](/Img/4.PNG)  

Advantages and disadvantages

Advantage:
- Ability to learn nonlinear models.
- Ability to learn models in real time (online learning) using partial_fit.

Disadvantages:

- MLPs with hidden layers have a non-convex loss function when there is more than one local minimum. Therefore, different random weight initializations can lead to different validation accuracy.
- MLP requires adjusting a series of hyper parameters.
- MLP is scale sensitive.

# Implementation.

-Apache Spark

Apache Spark is a distributed processing engine responsible for orchestrating, distributing, and monitoring applications that consist of multiple data processing tasks across multiple machine jobs, which form a cluster.




-scala

Scala programs run in the Java virtual machine (JVM) and can interoperate with Java programs and application programming interfaces (APIs). It is a multi-paradigm programming language that natively supports imperative, object-oriented, and functional programming styles.
Also, using the flexible features of the language syntax, they create powerful libraries. The extensions provide actor-based concurrency-oriented programming and language-oriented programming facilities.

# Results.

## Decision Three
## Code
 
We import the necessary libraries with which we are going to work 
```r
for(i <- 0 to 30)
{
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DateType
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.log4j._
```
Remove the warnings
```r
Logger.getLogger("org").setLevel(Level.ERROR)
 ```
 ```r
Create a spark session and load the CSV data into a datraframe
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load(" bank-full.csv")
 ```
We change the column y for one with binary data 
```r
val change1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val change2 = change1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcolumn = change2.withColumn("y",'y.cast("Int"))
 ```
Generate the features table 
```r
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val fea = assembler.transform(newcolumn)
 ```
Change the y column to the label column
```r
val change = fea.withColumnRenamed("y", "label")
val feat = change.select("label","features")
 ```
DecisionTree
```r
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(feat)
 ```
Features with more than 4 distinct values are taken as continuous 
```r
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4)
 ```
Split the data between 70% and 30% in an array
```r
val Array(trainingData, testData) = feat.randomSplit(Array(0.7, 0.3))
``` 
Create a DecisionTree object
```r
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
 ```
Prediction branch
```r
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
 ```
Put the data together in a pipeline 
```r
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
 ```
Create a training model
```r
val model = pipeline.fit(trainingData)
 ```
Transformation of data in the model
```r
val predictions = model.transform(testData)
 ```
Evaluate accuracy 
```r
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"ACCURACY ACCURACY ACCURACY= ${accuracy}")
}
  ```
![logo](/Img/5.PNG)  
  
## Logistic Regression
## Code

Import the libraries
```r
for(i <- 0 to 30)
{
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
 
Logger.getLogger("org").setLevel(Level.ERROR)
 
val spark = SparkSession.builder().getOrCreate()
 ```
The CSV data is imported
```r
val data = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load(" bank-full.csv")
 ```
String variables are categorized to numeric value
```r
val yes = data.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val clean = yes.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val cleanData = clean.withColumn("y",'y.cast("Int"))
 ```
Creation of the Array with the selected data
```r
val featureCols = Array("age","previous","balance","duration")
 ```
Creation of the Vector based on the features
```r
val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
 ```
Transformation to a new DF
```r
val df2 = assembler.transform(cleanData)
 ```
Rename columns
```r
val featuresLabel = df2.withColumnRenamed("y", "label")
 ```
Selection of index
```r
val dataI = featuresLabel.select("label","features")
 ```
Creation of the Array with the training and test data
```r
val Array(training, test) = dataI.randomSplit(Array(0.7, 0.3), seed = 12345)
 ```
Regression Model
```r
val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
val lrModel = lr.fit(training)
var results = lrModel.transform(test)
 
import org.apache.spark.mllib.evaluation.MulticlassMetrics
 ```
Convert test results to RDD using .as and .rdd
```r
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
 ```
Initialize a MulticlassMetrics object
```r
val metrics = new MulticlassMetrics(predictionAndLabels)
 ```
Accuracy
 ```r
println(s"ACCURACY ACCURACY ACCURACY= ${metrics.accuracy}")
 
}  
```
