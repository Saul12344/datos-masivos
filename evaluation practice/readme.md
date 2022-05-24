<p align="center">
    <img alt="Logo" src="https://www.tijuana.tecnm.mx/wp-content/uploads/2021/08/liston-de-logos-oficiales-educacion-tecnm-FEB-2021.jpg" width=850 height=250>
</p>
<H2><p align="center">Unit#2 branch. Big Data course.</p></H2>
<H2><p align="Center">STUDENT'S NAMES: </p></H2>

<H2><p align="Center">López Higuera Saúl Alfredo #18210493</p></H2>

<H2><p align="Center">Ramos Rivera Manuel Isaí #17212931</p></H2>

## Evaluative practice#2
[Practice link]()  
[.scala File]()  
## Code
  ~~~
//Utilice la libreria Mllib de Spark el algoritmo de Machine Learning multilayer perceptron

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
///Cargar sesion spark
var spark = SparkSession.builder().getOrCreate()
//Cargar Iris.csv
val df = spark.read.format("csv").option("inferSchema","true").option("header","true").csv("iris.csv")
  ~~~
![logo](/evaluation practice/img/a.2.PNG)  
  ~~~
//Cuales son los nombres de las columnas?
df.columns

//Como es el esquema?
df.printSchema()

//Imprime las primeras 5 columnas.
df.select($"sepal_length",$"sepal_width",$"petal_length",$"petal_width",$"species").show()

//Usa el metodo describe () para aprender mas sobre los datos del DataFrame.
df.describe()

//Haga la transformacion pertinente para los datos categoricos los cuales seran nuestras etiquetas a clasificar.

//Creamos un vector assambler y combinamos todas estas columnas

val Vassembler = new VectorAssembler().setInputCols(Array("sepal_length","sepal_width","petal_length","petal_width")).setOutputCol("features")
val output = Vassembler.transform(df)
output.show()

//Transformamos la columna especial en numerica y nombrandola como label
val labelIndexer = new StringIndexer().setInputCol("species").setOutputCol("label").fit(df) 
val indexed = labelIndexer.transform(output) 
indexed.show()


//Construya el modelo de clasificacion y explique su arquitectura.

//dividira en un 70% de entrenamiento y un 30% de prueba
val splits = indexed.randomSplit(Array(0.7, 0.3), seed = 1234L)
val train = splits(0)
val test = splits(1)

//Arreglo de las capas de la red neuronal,en este caso escogemos ciertos valores del mismo arreglo de la capa ya mencionada 
val layers = Array[Int](4, 4, 4, 3)

// Creacion del modelo de entrenamiento
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

// Variable para el modelo de entrenamiento
val model = trainer.fit(train)
 
// Valores de la precision
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
 
// Imprime los valores de precision
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")

//Muestra la distribucion de los datos 
println(s"train: ${train.count}, test: ${test.count()}")
//Mostrar el valor real de la tabla frente a la prediccion
result.select("features", "label", "prediction").show(test.count().asInstanceOf[Int])
  ~~~
