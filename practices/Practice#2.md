<p align="center">
    <img alt="Logo" src="https://www.tijuana.tecnm.mx/wp-content/uploads/2021/08/liston-de-logos-oficiales-educacion-tecnm-FEB-2021.jpg" width=850 height=250>
</p>

<H2><p align="Center">TECNOLÓGICO NACIONAL DE MÉXICO</p></H2>

<H2><p align="Center">INSTITUTO TECNOLÓGICO DE TIJUANA</p></H2>

<H2><p align="Center">SUBDIRECCIÓN ACADÉMICA</p></H2>

<H2><p align="Center">DEPARTAMENTO DE SISTEMAS Y COMPUTACIÓN</p></H2>

<H2><p align="Center">NOMBRE DE LOS ALUMNOS: </p></H2>

<H2><p align="Center">López Higuera Saúl Alfredo #18210493</p></H2>

<H2><p align="Center">Ramos Rivera Manuel Isaí #17212931</p></H2>

<H2><p align="Center">Carrera: Ingeniería Informática</p></H2>

<H2><p align="Center">Semestre: 9 </p></H2>

<H2><p align="Center">MATERIA: Datos masivos</p></H2>

<H2><p align="Center">PROFESOR: JOSE CHRISTIAN ROMERO HERNANDEZ</p></H2>

<H2><p align="Center">TRABAJOS: Practica 2</p></H2>

<H2><p align="Center">FECHA: 05/05/22</p></H2>

<br>
<br>
<br>
<br>
<br>

###

    import org.apache.spark.ml.Pipeline
    <br>
    import org.apache.spark.ml.classification.DecisionTreeClassificationModel
    <br>
    import org.apache.spark.ml.classification.DecisionTreeClassifier 
    <br>
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator 
    <br>
    import org.apache.spark.ml.feature.
    <br>
    {IndexToString, StringIndexer, VectorIndexer}
    <br>
    // Load the data stored in LIBSVM format as a DataFrame.
    <br>
    val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
    <br>
    // Index labels, adding metadata to the label column.
    <br>
    // Fit on whole dataset to include all labels in index.
    <br>
    val labelIndexer = new StringIndexer() .setInputCol("label") .setOutputCol("indexedLabel") .fit(data) 
    <br>
    //Automatically identify categorical features, and index them.
    <br>
    val featureIndexer = new VectorIndexer() .setInputCol("features") .setOutputCol("indexedFeatures") .setMaxCategories(4) 
    <br>
    // features with > 4 distinct values are treated as continuous. .fit(data)
    <br>
    // Split the data into training and test sets (30% held out for testing).
    <br>
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
    <br>
    // Train a DecisionTree model. 
    <br>
    val dt = new DecisionTreeClassifier() .setLabelCol("indexedLabel") .setFeaturesCol("indexedFeatures")
    <br>
    // Convert indexed labels back to original labels. 
    <br>
    val labelConverter = new IndexToString() .setInputCol("prediction") .setOutputCol("predictedLabel") .setLabels(labelIndexer.labels)
    <br>
    // Chain indexers and tree in a Pipeline. 
    <br>
    val pipeline = new Pipeline() .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
    <br>
    // Train model. This also runs the indexers.
    <br>
    val model = pipeline.fit(trainingData)
    <br>
    // Make predictions. 
    <br>
    val predictions = model.transform(testData)
    <br>
    // Select example rows to display. 
    <br>
    predictions.select("predictedLabel", "label", "features").show(5)
    <br>
    // Select (prediction, true label) and compute test error. 
    <br>
    val evaluator = new MulticlassClassificationEvaluator() .setLabelCol("indexedLabel") .setPredictionCol("prediction") .setMetricName("accuracy") val accuracy = evaluator.evaluate(predictions) println(s"Test Error = ${(1.0 - accuracy)}")
    <br>
    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel] println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

