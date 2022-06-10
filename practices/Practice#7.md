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

<H2><p align="Center">Homework: Practice 7</p></H2>

<H2><p align="Center">Date: 20/05/22</p></H2>


## Practice#7
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
