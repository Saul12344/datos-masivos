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

<H2><p align="Center">Homework: Practice 6</p></H2>

<H2><p align="Center">Date: 19/05/22</p></H2>


## Practice#6 
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
