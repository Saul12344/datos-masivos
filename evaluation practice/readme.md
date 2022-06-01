<p align="center">
    <img alt="Logo" src="https://www.tijuana.tecnm.mx/wp-content/uploads/2021/08/liston-de-logos-oficiales-educacion-tecnm-FEB-2021.jpg" width=850 height=250>
</p>
<H2><p align="center">Unit#3 branch. Big Data course.</p></H2>
<H2><p align="Center">STUDENT'S NAMES: </p></H2>

<H2><p align="Center">López Higuera Saúl Alfredo #18210493</p></H2>

<H2><p align="Center">Ramos Rivera Manuel Isaí #17212931</p></H2>
<H2><p align="Center">LINK DEL VIDEO:https://youtu.be/FWt-n5UPNuE</p></H2>

## Evaluative practice#3
[.scala File](https://github.com/Saul12344/datos-masivos/blob/unit-3/evaluation%20practice/Evaluation3.scala)  

## Code
   ## 1. Import a simple Spark session.
    
    import org.apache.spark.sql.SparkSession

   ## 2. Use the lines of code to minimize errors
    import org.apache.log4j._
    Logger.getLogger("org").setLevel(Level.ERROR)

    // 3. Create an instance of the Spark session
    val spark = SparkSession.builder().getOrCreate()

    // 4. Import the Kmeans library for the clustering algorithm.
    import org.apache.spark.ml.clustering.KMeans

    // 5. Load the Wholesale Customers Data dataset
    val dataset = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Wholesale customers data.csv")

    // 6. Select the following columns: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen and call this set feature_data
    val feature_data = (dataset.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen"))

    // 7. Import Vector Assembler and Vector
    import org.apache.spark.ml.feature.VectorAssembler
    import org.apache.spark.ml.linalg.Vectors

    // 8. Create a new Vector Assembler object for the feature columns as a input set, remembering that there are no labels
    val assembler = (new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features"))

    // 9. Use the assembler object to transform feature_data
    val training_data = assembler.transform(feature_data).select($"features")

    // 10. Create a Kmeans model with K = 3
    val kmeans = new KMeans().setK(3).setSeed(1L)

    val model = kmeans.fit(training_data)

    // 11.Evaluate the groups using Within Set Sum of Squared Errors WSSSE and print the centroids.
    val WSSSE = model.computeCost(training_data)
    println(s"Within Set Sum of Squared Errors = $WSSSE")

    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)
    
   
   ## results
<img alt="Logo" src="https://github.com/Saul12344/datos-masivos/blob/unit-3/Img/1.PNG">
<img alt="Logo" src="https://github.com/Saul12344/datos-masivos/blob/unit-3/Img/2.PNG">
<img alt="Logo" src="https://github.com/Saul12344/datos-masivos/blob/unit-3/Img/3.PNG">
  ~~~
