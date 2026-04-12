// ============================================================
// IT462 – Big Data Systems | Phase 5: Machine Learning
// File: 04_MachineLearning.scala
// Dataset: LinkedIn Job Postings
// Focus: Salary prediction using regression
// ------------------------------------------------------------
// Environment: Apache Spark 3.5.0
//              Scala 2.12.18
//              OpenJDK 11
// ============================================================
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression._

// ============================================================
// 1. LOAD DATA
// ============================================================

val df = spark.read.parquet("data/preprocessed_dataset.parquet")

println(s"Total rows: ${df.count()}")
df.printSchema()

// ============================================================
// 2. FILTER YEARLY SALARY DATA
// ============================================================

val dfYearly = df.filter(
  col("normalized_salary").isNotNull &&
  col("normalized_salary") >= 10000.0 &&
  col("normalized_salary") <= 1000000.0 &&
  col("pay_period_std") === "YEARLY"
)

println(s"Filtered rows: ${dfYearly.count()}")

// ============================================================
// 3. SELECT FEATURES
// ============================================================

val modelInputY = dfYearly.select(
  col("normalized_salary"),
  col("log_views"),
  col("log_applies"),
  col("title_len"),
  col("title_words"),
  col("listed_month"),
  col("listed_dow"),
  col("listed_hour"),
  col("work_type_std")
)

// ============================================================
// 4. CLEAN DATA (DROP NULLS)
// ============================================================

val cleanInputY = modelInputY.na.drop(
  Seq(
    "normalized_salary",
    "log_views",
    "log_applies",
    "title_len",
    "title_words",
    "listed_month",
    "listed_dow",
    "listed_hour",
    "work_type_std"
  )
)

println(s"Rows after cleaning: ${cleanInputY.count()}")

// ============================================================
// 5. TRAIN / TEST SPLIT
// ============================================================

val Array(trainRawY, testRawY) = cleanInputY.randomSplit(Array(0.7, 0.3), 42)

println(s"Training rows: ${trainRawY.count()}")
println(s"Test rows: ${testRawY.count()}")

// ============================================================
// 6. ENCODING (work_type_std)
// ============================================================

// String Indexer
val workTypeIndexerY = new StringIndexer()
  .setInputCol("work_type_std")
  .setOutputCol("work_type_index")
  .setHandleInvalid("keep")

val indexerModelY = workTypeIndexerY.fit(trainRawY)

val trainIndexedY = indexerModelY.transform(trainRawY)
val testIndexedY  = indexerModelY.transform(testRawY)

// One-Hot Encoder
val encoderY = new OneHotEncoder()
  .setInputCols(Array("work_type_index"))
  .setOutputCols(Array("work_type_vec"))

val encoderModelY = encoderY.fit(trainIndexedY)

val trainEncodedY = encoderModelY.transform(trainIndexedY)
val testEncodedY  = encoderModelY.transform(testIndexedY)

// ============================================================
// 7. FEATURE ASSEMBLY
// ============================================================

val assemblerY = new VectorAssembler()
  .setInputCols(Array(
    "log_views",
    "log_applies",
    "title_len",
    "title_words",
    "listed_month",
    "listed_dow",
    "listed_hour",
    "work_type_vec"
  ))
  .setOutputCol("features_raw")

val trainAssembledY = assemblerY.transform(trainEncodedY)
val testAssembledY  = assemblerY.transform(testEncodedY)

// ============================================================
// 8. FEATURE SCALING
// ============================================================

val scalerY = new StandardScaler()
  .setInputCol("features_raw")
  .setOutputCol("features")
  .setWithMean(true)
  .setWithStd(true)

val scalerModelY = scalerY.fit(trainAssembledY)

val trainFinalY = scalerModelY.transform(trainAssembledY)
val testFinalY  = scalerModelY.transform(testAssembledY)

// ============================================================
// 9. PREPARE FINAL DATA
// ============================================================

val trainDataY = trainFinalY.withColumnRenamed("normalized_salary", "label")
val testDataY  = testFinalY.withColumnRenamed("normalized_salary", "label")

// ============================================================
// 10. LINEAR REGRESSION
// ============================================================

val lrY = new LinearRegression()
  .setLabelCol("label")
  .setFeaturesCol("features")
  .setPredictionCol("prediction_lr")
  .setMaxIter(100)
  .setRegParam(0.1)
  .setElasticNetParam(0.0)

val lrModelY = lrY.fit(trainDataY)

println("Linear Regression model trained.")

// ============================================================
// 11. RANDOM FOREST
// ============================================================

val rfY = new RandomForestRegressor()
  .setLabelCol("label")
  .setFeaturesCol("features")
  .setPredictionCol("prediction_rf")
  .setNumTrees(50)
  .setMaxDepth(10)
  .setSeed(42)

val rfModelY = rfY.fit(trainDataY)

println("Random Forest model trained.")

// ============================================================
// 12. PREDICTIONS
// ============================================================

val lrPredictionsY = lrModelY.transform(testDataY)
val rfPredictionsY = rfModelY.transform(testDataY)

println("\n=== Linear Regression Predictions ===")
lrPredictionsY.select("label", "prediction_lr").show(20, false)

println("\n=== Random Forest Predictions ===")
rfPredictionsY.select("label", "prediction_rf").show(20, false)
