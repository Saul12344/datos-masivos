// Linear Support Vector Machine

// Import the "LinearSVC" library.
import org.apache.spark.ml.classification.LinearSVC

// Load training data
val training  = spark.read.format("libsvm").load("C:/Spark/spark-2.4.8-bin-hadoop2.7/data/mllib/sample_libsvm_data.txt")

//Set the maximum number of iterations and the regularization parameter 
val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)


// Fit the model
val lsvcModel = lsvc.fit(training)

// Print the coefficients and intercept for linear svc
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
