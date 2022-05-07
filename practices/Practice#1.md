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

<H2><p align="Center">Homework: Practica 2</p></H2>

<H2><p align="Center">Date: 05/05/22</p></H2>


## Practice#1 
<BR>
</BR>
          
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

