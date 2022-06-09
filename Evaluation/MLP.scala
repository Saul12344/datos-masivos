/*Importamos las librerias necesarias con las que vamos a trabajar*/
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
//codigo de nivel de error.
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
//sesión de spark.
val spark = SparkSession.builder.appName("MultilayerPerceptron").getOrCreate()
//cargamos el archivo csv.
val bank = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")

bank.show

//Mostrar el nombre de las columnas.
bank.columns
//indexación.
val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("indexedLabel").fit(bank)
val indexed = labelIndexer.transform(bank).drop("y").withColumnRenamed("indexedLabel", "label")
//Sumamos el vector de las columnas de categorías numéricas.
val vectorFeatures = (new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features"))
//transformar el valor indexado.
val features = vectorFeatures.transform(indexed)
//ajustar los índices y encontrar las etiquetas 0 y 1.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(indexed)
//correr 30 veces el modelo.
for(i <- Range(1,30))
{
//Iniciamos el temporizador.
val startTimeMillis = System.currentTimeMillis()

//separar los valores y dividir los datos en 70% y 30%.
val splits = features.randomSplit(Array(0.7, 0.3))
val trainingData = splits(0)
val testData = splits(1)
//Creamos la matriz de capas.
val layers = Array[Int](5, 4, 1, 2)
//crear el objeto Multilayer Perceptron.
val multilayerP = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
//datos de entrenamiento en el modelo.
val model = multilayerP.fit(trainingData)
//Transformación de los datos de prueba para las predicciones.
val prediction = model.transform(testData)
//seleccionar las columnas de predicción y etiqueta.
val predictionAndLabels = prediction.select("prediction", "label")
//Creamos un objeto Evaluador de Clasificación Multiclase.
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")


val endTimeMillis = System.currentTimeMillis()
val durationSeconds = (endTimeMillis - startTimeMillis) / 1000

//resultados.
println("|"+i + "|"+evaluator.evaluate(predictionAndLabels) +"|"+(1.0 - evaluator.evaluate(predictionAndLabels))+"|" +durationSeconds+ "|")}