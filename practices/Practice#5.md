<p align="center">
    <img alt="Logo" src="https://www.tijuana.tecnm.mx/wp-content/uploads/2021/08/liston-de-logos-oficiales-educacion-tecnm-FEB-2021.jpg" width=850 height=250>
</p>

<H2><p align="Center">TECNOLÓGICO NACIONAL DE MÉXICO</p></H2>

<H2><p align="Center">INSTITUTO TECNOLÓGICO DE TIJUANA</p></H2>

<H2><p align="Center">SUBDIRECCIÓN ACADÉMICA</p></H2>

<H2><p align="Center">DEPARTAMENTO DE SISTEMAS Y COMPUTACIÓN</p></H2>

<H2><p align="Center">Students's Names: </p></H2>

<H2><p align="Center">López Higuera Saúl Alfredo #18210493</p></H2>

<H2><p align="Center">Ramos Rivera Manuel Isaí #17212931</p></H2>

<H2><p align="Center">Career: Ingeniería en sistemas</p></H2>

<H2><p align="Center">Semester: 9 </p></H2>

<H2><p align="Center">Subject-matter: Datos masivos</p></H2>

<H2><p align="Center">Teacher's name: JOSE CHRISTIAN ROMERO HERNANDEZ</p></H2>

<H2><p align="Center">Homework: Practica 5</p></H2>

<H2><p align="Center">Date: 05/18/22</p></H2>


## Practice#5
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

