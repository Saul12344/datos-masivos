//Bibliotecas necesarias.
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
//codigo de nivel de error.
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

//Sesión de spark.
val spark = SparkSession.builder.appName("SVM").getOrCreate()
//Importación CSV.
val bank = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")

bank.show
//nombre de las columnas.
bank.columns
//indexación.
val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("indexedLabel").fit(bank)
val indexed = labelIndexer.transform(bank).drop("y").withColumnRenamed("indexedLabel", "label")
//Sumamos el vector de las columnas de categorías numéricas.
val vectorFeatures = (new VectorAssembler().setInputCols(Array("duration","pdays","previous")).setOutputCol("features"))
//transformar el valor indexado.
val features = vectorFeatures.transform(indexed)

//Cambiamos el nombre de la columna "y" como una etiqueta.
val featuresLabel = features.withColumnRenamed("y", "label")
//Ahora hacemos unión de etiquetas y características como dataIndexed.
val dataIndexed = featuresLabel.select("label","features")
//creamos labelIndexer y featureIndexer para la canalización.features with distinct values > 4, are treated as continuous.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(dataIndexed)
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(dataIndexed)
//Correr 30 veces el modelo.
for(i <- Range(1,31))
{
//Iniciamos el temporizador
val startTimeMillis = System.currentTimeMillis()
//separar los valores y dividimos los datos en 70% y 30%.
val Array(training, test) = dataIndexed.randomSplit(Array(0.7, 0.3))
//crear el objeto Linear Vector Machine.
val supportVM = new LinearSVC().setMaxIter(10).setRegParam(0.1)

// datos de entrenamiento en el modelo.
val model = supportVM.fit(training)
//Transformando testData para el modelo de predicciones.
val predictions = model.transform(test)
//obtenemos las métricas.
val predictionAndLabels = predictions.select($"prediction",$"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)
//matriz de confusión.
println("Confusion matrix:")
println(metrics.confusionMatrix)
//terminar el tiempo para el modelo.
val endTimeMillis = System.currentTimeMillis()
val durationSeconds = (endTimeMillis - startTimeMillis) / 1000
//Resultados.
println("|"+i + "|"+ metrics.accuracy +"|"+ (1.0 - metrics.accuracy)+"|" +durationSeconds + "|")
}