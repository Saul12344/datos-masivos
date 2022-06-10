unit-4
<p align="center">
    <img alt="Logo" src="https://www.tijuana.tecnm.mx/wp-content/uploads/2021/08/liston-de-logos-oficiales-educacion-tecnm-FEB-2021.jpg" width=850 height=250>
</p>
<H2><p align="center">Unit#4 branch. Big Data course.</p></H2>

unit-2
<p align="center">
    <img alt="Logo" src="https://www.tijuana.tecnm.mx/wp-content/uploads/2021/08/liston-de-logos-oficiales-educacion-tecnm-FEB-2021.jpg" width=850 height=250>
</p>
<H2><p align="center">Unit#2 branch. Big Data course.</p></H2>
 develop
<H2><p align="Center">STUDENT'S NAMES: </p></H2>

<H2><p align="Center">López Higuera Saúl Alfredo #18210493</p></H2>

unit-4
<H2><p align="Center">LINK DEL VIDEO:https://www.youtube.com/watch?v=eLcprAHl5yA</p></H2>

# Final project
[Decision Three.scala](https://github.com/Saul12344/datos-masivos/blob/unit-4/Evaluation/DTC.scala) 
<br>
[Logistic Regression.scala](https://github.com/Saul12344/datos-masivos/blob/unit-4/Evaluation/LR.scala) 
<br>
[Multilayer perceptron.scala](https://github.com/Saul12344/datos-masivos/blob/unit-4/Evaluation/MLP.scala) 
<br>
[Support Vector Machine.scala](https://github.com/Saul12344/datos-masivos/blob/unit-4/Evaluation/SVM.scala) 
<br>
[Final_project File](https://github.com/Saul12344/datos-masivos/blob/unit-4/Evaluation/README.md) 
=======

## Practice#1
[Practice link](https://github.com/Saul12344/datos-masivos/blob/unit-2/practices/Practice%231.md)  
[.scala File](https://github.com/Saul12344/datos-masivos/blob/unit-2/practices/Practice%231.scala)  
## Code

    package org.apache.spark.examples.mllib

    import org.apache.spark.{SparkConf, SparkContext}
    // $example on$
    import org.apache.spark.mllib.linalg._
    import org.apache.spark.mllib.stat.Statistics
    import org.apache.spark.rdd.RDD
    object CorrelationsExample {

      def main(){

     val conf = new SparkConf().setAppName("CorrelationsExample")
      val sc = new SparkContext(conf)

     // $example on$
    val seriesX: RDD[Double] = sc.parallelize(Array(1, 2, 3, 3, 5))  // a series
    // must have the same number of partitions and cardinality as seriesX
    val seriesY: RDD[Double] = sc.parallelize(Array(11, 22, 33, 33, 555))

      // compute the correlation using Pearson's method. Enter "spearman" for Spearman's method. If a
    // method is not specified, Pearson's method will be used by default.
    val correlation: Double = Statistics.corr(seriesX, seriesY, "pearson")
    println(s"Correlation is: $correlation")

     val data: RDD[Vector] = sc.parallelize(
    Seq(
        Vectors.dense(1.0, 10.0, 100.0),
        Vectors.dense(2.0, 20.0, 200.0),
        Vectors.dense(5.0, 33.0, 366.0))
       )  // note that each Vector is a row and not a column
        // calculate the correlation matrix using Pearson's method. Use "spearman" for Spearman's method
          // If a method is not specified, Pearson's method will be used by default.
    val correlMatrix: Matrix = Statistics.corr(data, "pearson")
    println(correlMatrix.toString)
    // $example off$

      sc.stop()
    }
    }


## Practice#2
[Practice link](https://github.com/Saul12344/datos-masivos/blob/unit-2/practices/Practice%232.md)  
[.scala File](https://github.com/Saul12344/datos-masivos/blob/unit-2/practices/Practice%232.scala)  
## Code

<br>

    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.classification.DecisionTreeClassificationModel
    import org.apache.spark.ml.classification.DecisionTreeClassifier 
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator 
    import org.apache.spark.ml.feature.
    {IndexToString, StringIndexer, VectorIndexer}

    // Load the data stored in LIBSVM format as a DataFrame.
    val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.

    val labelIndexer = new StringIndexer() .setInputCol("label") .setOutputCol("indexedLabel") .fit(data) 

    //Automatically identify categorical features, and index them.

    val featureIndexer = new VectorIndexer() .setInputCol("features") .setOutputCol("indexedFeatures") .setMaxCategories(4) 

    // features with > 4 distinct values are treated as continuous. .fit(data)

    // Split the data into training and test sets (30% held out for testing).

    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    // Train a DecisionTree model. 

    val dt = new DecisionTreeClassifier() .setLabelCol("indexedLabel") .setFeaturesCol("indexedFeatures")

    // Convert indexed labels back to original labels. 

    val labelConverter = new IndexToString() .setInputCol("prediction") .setOutputCol("predictedLabel") .setLabels(labelIndexer.labels)

    // Chain indexers and tree in a Pipeline. 

    val pipeline = new Pipeline() .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

    // Train model. This also runs the indexers.

    val model = pipeline.fit(trainingData)

    // Make predictions. 

    val predictions = model.transform(testData)

    // Select example rows to display. 

    predictions.select("predictedLabel", "label", "features").show(5)

    // Select (prediction, true label) and compute test error. 

    val evaluator = new MulticlassClassificationEvaluator() .setLabelCol("indexedLabel") .setPredictionCol("prediction") .setMetricName("accuracy") val accuracy = evaluator.evaluate(predictions) println(s"Test Error = ${(1.0 - accuracy)}")

    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel] println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
## Practice#3
[Practice link](https://github.com/Saul12344/datos-masivos/blob/unit-2/practices/Practice%233.md)  
[.scala File](https://github.com/Saul12344/datos-masivos/blob/unit-2/practices/Practice%233.scala)  
## Code

<br>

    import org.apache.spark.mllib.tree.RandomForest
    import org.apache.spark.mllib.tree.model.RandomForestModel
    import org.apache.spark.mllib.util.MLUtils

    // Load and parse the data file.
    val data = MLUtils.loadLibSVMFile(sc, "sample_libsvm_data.txt")
    // Split the data into training and test sets (30% held out for testing)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 3 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 32

    val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.map { point =>
     val prediction = model.predict(point.features)
     (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println(s"Test Error = $testErr")
    println(s"Learned classification forest model:\n ${model.toDebugString}")

    // Save and load model
    model.save(sc, "target/tmp/myRandomForestClassificationModel")
    val sameModel = RandomForestModel.load(sc, "target/tmp/myRandomForestClassificationModel")

    //Ejemplo Regresion
    import org.apache.spark.mllib.tree.RandomForest
    import org.apache.spark.mllib.tree.model.RandomForestModel
    import org.apache.spark.mllib.util.MLUtils

    // Load and parse the data file.
    val data = MLUtils.loadLibSVMFile(sc, "sample_libsvm_data.txt")
    // Split the data into training and test sets (30% held out for testing)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 3 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "variance"
    val maxDepth = 4
    val maxBins = 32

    val model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelsAndPredictions = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testMSE = labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2)}.mean()
    println(s"Test Mean Squared Error = $testMSE")
    println(s"Learned regression forest model:\n ${model.toDebugString}")

    // Save and load model
    model.save(sc, "target/tmp/myRandomForestRegressionModel")
    val sameModel = RandomForestModel.load(sc, "target/tmp/myRandomForestRegressionModel")

## Practice#4
[Practice link](https://github.com/Saul12344/datos-masivos/blob/unit-2/practices/Practice%234.md)  
[.scala File](https://github.com/Saul12344/datos-masivos/blob/unit-2/practices/Practice%234.scala)  
## Code
<BR>
</BR>

    import org.apache.spark.mllib.tree.GradientBoostedTrees
    import org.apache.spark.mllib.tree.configuration.BoostingStrategy
    import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
    import org.apache.spark.mllib.util.MLUtils

    // Load and parse the data file.
    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
    // Split the data into training and test sets (30% held out for testing)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // Train a GradientBoostedTrees model.
    // The defaultParams for Classification use LogLoss by default.
    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.numIterations = 3 // Note: Use more iterations in practice.
    boostingStrategy.treeStrategy.numClasses = 2
    boostingStrategy.treeStrategy.maxDepth = 5
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

    val model = GradientBoostedTrees.train(trainingData, boostingStrategy)

    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println(s"Test Error = $testErr")
    println(s"Learned classification GBT model:\n ${model.toDebugString}")

**Output.**

<img alt="Logo" src="https://github.com/Saul12344/datos-masivos/blob/unit-2/images/RP5.png" width=850 height=850>


   


    // Save and load model
    model.save(sc, "target/tmp/myGradientBoostingClassificationModel")
    val sameModel = GradientBoostedTreesModel.load(sc,
      "target/tmp/myGradientBoostingClassificationModel")
      
## Practice#5
[Practice link](https://github.com/Saul12344/datos-masivos/blob/unit-2/practices/Practice%235.md)  
[.scala File](https://github.com/Saul12344/datos-masivos/blob/unit-2/practices/Practice%235.scala)  
## Code
<BR>
</BR>

      import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
      import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

      // Load the data stored in LIBSVM format as a DataFrame. || Carga los datos almacenados en formato LIBSVM como DataFrame.

      //val data = spark.read.format("libsvm").load("data/mllib/sample_multiclass_classification_data.txt")
      val data = spark.read.format("libsvm").load("C:/Spark/spark-2.4.8-bin-hadoop2.7/data/mllib/sample_multiclass_classification_data.txt")

      // Split the data into train and test || Divide los datos
      val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
      val train = splits(0)
      val test = splits(1)

      // specify layers for the neural network: || especificar capas para la red neuronal:
      // input layer of size 4 (features), two intermediate of size 5 and 4 || capa de entrada de tamano 4 (features), dos intermedias de tamano 5 y 4
      // and output of size 3 (classes) || y salida de tamano 3 (classes) 
      val layers = Array[Int](4, 5, 4, 3)

      // create the trainer and set its parameters || Crea el trainer y establece sus parametros.
      val trainer = new MultilayerPerceptronClassifier()
        .setLayers(layers)
        .setBlockSize(128)
        .setSeed(1234L)
            .setMaxIter(100)

      // train the model || entrena el model
      val model = trainer.fit(train)

      // compute accuracy on the test set || precision de calculo en el conjunto de prueba
      val result = model.transform(test)
      val predictionAndLabels = result.select("prediction", "label")
      val evaluator = new MulticlassClassificationEvaluator()
        .setMetricName("accuracy")

    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
## Practice#6
[Practice link](https://github.com/Saul12344/datos-masivos/blob/unit-2/practices/Practice%236.md)  
[.scala File](https://github.com/Saul12344/datos-masivos/blob/unit-2/practices/Practice%236.scala)  
## Code
<BR>
</BR>

  ~~~
// Import the "LinearSVC" library.
import org.apache.spark.ml.classification.LinearSVC

// Load training data
val training = spark.read.format("libsvm").load("C:/Spark/spark-2.4.8-binhadoop2.7/data/mllib/sample_libsvm_data.txt")

//Set the maximum number of iterations and the regularization parameter
val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)

val lsvcModel = lsvc.fit(training)

println(s"Coefficients: ${lsvcModel.coefficients} Intercept:
${lsvcModel.intercept}")
~~~
![logo](/images/P6.PNG)  

## Practice#7
[Practice link](https://github.com/Saul12344/datos-masivos/blob/unit-2/practices/Practice%237.md)  
[.scala File](https://github.com/Saul12344/datos-masivos/blob/unit-2/practices/Practice%237.scala)  
## Code
<BR>
</BR>

~~~
//Importar las librerías necesarias
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession

//Cargar los datos especificando la ruta del archivo
val data = spark.read.format("libsvm").load("C:/spark/spark-2.4.8-bin-hadoop2.7/data/mllib/sample_libsvm_data.txt")

println ("Numero de lineas en el archivo de datos:" + data.count ())


//Mostrar las primeras 20 líneas por defecto

data.show()
~~~
![logo](/images/7.1.PNG)  
~~~
//Divida aleatoriamente el conjunto de datos en conjunto de entrenamiento y conjunto de prueba de acuerdo con los pesos proporcionados. También puede especificar una seed
val Array (trainingData, testData) = data.randomSplit (Array (0.7, 0.3), 100L)

// El resultado es el tipo de la matriz, y la matriz almacena los datos de tipo DataSet
//Incorporar al conjunto de entrenamiento (operación de ajuste) para entrenar un modelo bayesiano
val naiveBayesModel = new NaiveBayes().fit(trainingData)

//El modelo llama a transform() para hacer predicciones y generar un nuevo DataFrame.
val predictions = naiveBayesModel.transform(testData)


//Salida de datos de resultados de predicción
predictions.show()
 ~~~
 ![logo](/images/7.2.PNG)  
  ~~~
//Evaluación de la precisión del modelo
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")

// Precisión
val precision = evaluator.evaluate (predictions) 
 ~~~
 ![logo](/images/7.3.PNG)  
~~~
//Imprimir la tasa de error
println ("tasa de error =" + (1-precision))
 ~~~

![logo](/images/7.4.PNG)  

## Evaluative practice#2
[Evaluative practice#2](https://github.com/Saul12344/datos-masivos/blob/unit-2/evaluation%20practice/readme.md)  
[.scala File](https://github.com/Saul12344/datos-masivos/blob/unit-2/evaluation%20practice/Evaluation2.scala)  



