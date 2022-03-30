1. Start a simple Spark session
import org.apache.spark.sql.SparkSession

val session = SparkSession.builder().getOrCreate

2. Upload Netflix Stock CSV file, have spark infer data types
val df_netflix = session.read.option("header", "true").option("inferSchema", true).csv("Netflix_2011_2016.csv")

3. What are the names of the columns?
df_netflix.columns

4. How is the scheme?
df_netflix.printSchema()

5. Print the first 5 columns
df_netflix.head(5)

6. Use describe () to learn about the DataFrame
df_netflix.describe().show

7. Create a new data frame with a new column called "HV Ratio" which is the relationship between the price in the "High" column versus the "Volume" column of shares traded for a day. Hint is an operation.
val df_netflix2 = df_netflix.withColumn("HV Ratio", df_netflix("High")/df_netflix("Volume"))

8. What day had the highest peak in the “Open” column?
df_netflix.select(max("Open")).show()


9. What is the meaning of the “Close” column in the context of financial information, explain it, there is no need to code anything?

//The Close column refers to the company action price at the end of the day's closing.
10. What is the maximum and minimum in the “Volume” column?
df_netflix.select(max("Volume")).show()
df_netflix.select(min("Volume")).show()

11. With Scala / Spark $ syntax answer the following:

a. How many days was the “Close” column less than $ 600?
val Day = df_netflix.where($"Close" < 600).count()

b. What percentage of the time was the “High” column greater than $ 500?
val Day = df_netflix.where($"High" > 500).count().toFloat

c. What is the Pearson correlation between the “High” column and the “Volume” column?
df_netflix.select(corr("High", "Volume")).show()

d. What is the maximum in the “High” column per year?
df_netflix.groupBy(year($"Date")).max("High").show()

e. What is the average in the “Close” column for each calendar month?
val df_netflix3 = df_netflix.groupBy(year($"Date"), month($"Date")).mean("Close"). toDF("Year","Month","Mean")
df_netflix3.orderBy($"Year",$"Month").show()