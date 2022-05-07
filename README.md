<p align="center">
    <img alt="Logo" src="https://www.tijuana.tecnm.mx/wp-content/uploads/2021/08/liston-de-logos-oficiales-educacion-tecnm-FEB-2021.jpg" width=850 height=250>
</p>
<H2><p align="center">Unit#2 branch. Big Data course.</p></H2>
<H2><p align="Center">STUDENT'S NAMES: </p></H2>

<H2><p align="Center">López Higuera Saúl Alfredo #18210493</p></H2>

<H2><p align="Center">Ramos Rivera Manuel Isaí #17212931</p></H2>

## Practice#1
[PDF link](https://github.com/Saul12344/datos-masivos/blob/unit-2/practices/Practice%213.md)  
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
[PDF link](https://github.com/Saul12344/datos-masivos/blob/unit-2/practices/Practice%223.md)  
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
[PDF link](https://github.com/Saul12344/datos-masivos/blob/unit-2/practices/Practice%233.md)  
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
