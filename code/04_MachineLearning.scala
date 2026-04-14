import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression._
import org.apache.spark.ml.evaluation.RegressionEvaluator

// ============================================================
// 1. LOAD DATA
// ============================================================

val df = spark.read.parquet("data/preprocessed_dataset.parquet")
println(s"Total rows: ${df.count()}")

// ============================================================
// 2. FILTER YEARLY SALARY DATA
// ============================================================

val dfYearly = df.filter(
  col("normalized_salary").isNotNull &&
  col("log_salary").isNotNull &&
  col("normalized_salary") >= 10000.0 &&
  col("normalized_salary") <= 1000000.0 &&
  col("pay_period_std") === "YEARLY"
)
println(s"Filtered rows: ${dfYearly.count()}")

// ============================================================
// 3. FEATURE ENGINEERING
// engagement = log_views + log_applies
// is_remote nulls filled with 0 to avoid dropping rows
// ============================================================

val dfFeatured = dfYearly
  .withColumn("engagement", col("log_views") + col("log_applies"))
  .withColumn("is_remote", coalesce(col("is_remote").cast("double"), lit(0.0)))
  .withColumn("formatted_experience_level",
    coalesce(col("formatted_experience_level"), lit("Unknown")))
  .withColumn("state",
    coalesce(col("state"), lit("UNKNOWN")))
  .withColumn("work_type_std",
    coalesce(col("work_type_std"), lit("UNKNOWN")))

// ============================================================
// 4. SELECT FEATURES
// Target: log_salary (verified mean ~11.24, true log scale)
// ============================================================

val modelInputY = dfFeatured.select(
  col("log_salary"),
  col("log_views"),
  col("log_applies"),
  col("engagement"),
  col("title_len"),
  col("title_words"),
  col("listed_month"),
  col("listed_dow"),
  col("listed_hour"),
  col("is_remote"),
  col("work_type_std"),
  col("formatted_experience_level"),
  col("state"),
  col("title_clean")
)

// Drop nulls only on columns that cannot be imputed
val cleanInputY = modelInputY.na.drop(Seq(
  "log_salary", "log_views", "log_applies",
  "title_len", "title_words",
  "listed_month", "listed_dow", "listed_hour",
  "title_clean"
))

println(s"Rows after cleaning: ${cleanInputY.count()}")

// ============================================================
// 5. TRAIN / TEST SPLIT
// 70/30 with fixed seed for reproducibility
// ============================================================

val Array(trainRawY, testRawY) = cleanInputY.randomSplit(Array(0.7, 0.3), 42)
println(s"Training rows: ${trainRawY.count()}")
println(s"Test rows:     ${testRawY.count()}")

// ============================================================
// 6. ENCODE CATEGORICAL FEATURES
// All encoders fit on training data only to prevent leakage
// ============================================================

val workTypeIndexerModelY = new StringIndexer()
  .setInputCol("work_type_std").setOutputCol("work_type_index")
  .setHandleInvalid("keep").fit(trainRawY)
val trainStep1Y = workTypeIndexerModelY.transform(trainRawY)
val testStep1Y  = workTypeIndexerModelY.transform(testRawY)

val expIndexerModelY = new StringIndexer()
  .setInputCol("formatted_experience_level").setOutputCol("experience_index")
  .setHandleInvalid("keep").fit(trainStep1Y)
val trainStep2Y = expIndexerModelY.transform(trainStep1Y)
val testStep2Y  = expIndexerModelY.transform(testStep1Y)

val stateIndexerModelY = new StringIndexer()
  .setInputCol("state").setOutputCol("state_index")
  .setHandleInvalid("keep").fit(trainStep2Y)
val trainIndexedY = stateIndexerModelY.transform(trainStep2Y)
val testIndexedY  = stateIndexerModelY.transform(testStep2Y)

val encoderModelY = new OneHotEncoder()
  .setInputCols(Array("work_type_index", "experience_index", "state_index"))
  .setOutputCols(Array("work_type_vec", "experience_vec", "state_vec"))
  .fit(trainIndexedY)
val trainEncodedY = encoderModelY.transform(trainIndexedY)
val testEncodedY  = encoderModelY.transform(testIndexedY)

// ============================================================
// 7. TF-IDF ON JOB TITLE
// Captures semantic salary signals from title text.
// All steps fit on training data only to prevent leakage.
// ============================================================

val tokenizerY = new Tokenizer()
  .setInputCol("title_clean").setOutputCol("title_tokens")
val trainTok = tokenizerY.transform(trainEncodedY)
val testTok  = tokenizerY.transform(testEncodedY)

val removerY = new StopWordsRemover()
  .setInputCol("title_tokens").setOutputCol("title_filtered")
val trainRem = removerY.transform(trainTok)
val testRem  = removerY.transform(testTok)

val hashingTFY = new HashingTF()
  .setInputCol("title_filtered").setOutputCol("title_tf").setNumFeatures(500)
val trainTF = hashingTFY.transform(trainRem)
val testTF  = hashingTFY.transform(testRem)

// IDF fit on training data only
val idfModelY = new IDF()
  .setInputCol("title_tf").setOutputCol("title_tfidf").fit(trainTF)
val trainTFIDF = idfModelY.transform(trainTF)
val testTFIDF  = idfModelY.transform(testTF)

println("TF-IDF features created.")

// ============================================================
// 8. FEATURE ASSEMBLY
// Combines all features into one vector for MLlib
// ============================================================

val assemblerY = new VectorAssembler()
  .setInputCols(Array(
    "log_views", "log_applies", "engagement",
    "title_len", "title_words",
    "listed_month", "listed_dow", "listed_hour",
    "is_remote",
    "work_type_vec", "experience_vec", "state_vec",
    "title_tfidf"
  ))
  .setOutputCol("features_raw")
  .setHandleInvalid("skip")

val trainAssembledY = assemblerY.transform(trainTFIDF)
val testAssembledY  = assemblerY.transform(testTFIDF)

// ============================================================
// 9. FEATURE SCALING
// withMean=false required for sparse TF-IDF vectors
// Scaler fit on training data only
// ============================================================

val scalerModelY = new StandardScaler()
  .setInputCol("features_raw").setOutputCol("features")
  .setWithMean(false).setWithStd(true).fit(trainAssembledY)

val trainFinalY = scalerModelY.transform(trainAssembledY)
val testFinalY  = scalerModelY.transform(testAssembledY)

// ============================================================
// 10. PREPARE LABELED DATA
// ============================================================

val trainDataY = trainFinalY.withColumnRenamed("log_salary", "label")
val testDataY  = testFinalY.withColumnRenamed("log_salary", "label")

// ============================================================
// 11. BASELINE — MEAN PREDICTION
// ============================================================

val trainMeanY = trainDataY.agg(avg("label")).first().getDouble(0)
val baselinePredictionsY = testDataY.withColumn("prediction_baseline", lit(trainMeanY))
println(s"\nBaseline mean log_salary: $trainMeanY")

// ============================================================
// 12. LINEAR REGRESSION
// ============================================================

val lrModelY = new LinearRegression()
  .setLabelCol("label").setFeaturesCol("features")
  .setPredictionCol("prediction_lr")
  .setMaxIter(200).setRegParam(0.05).setElasticNetParam(0.1)
  .fit(trainDataY)
println("Linear Regression trained.")

// ============================================================
// 13. RANDOM FOREST
// ============================================================

val rfModelY = new RandomForestRegressor()
  .setLabelCol("label").setFeaturesCol("features")
  .setPredictionCol("prediction_rf")
  .setNumTrees(100).setMaxDepth(10).setSeed(42)
  .fit(trainDataY)
println("Random Forest trained.")

// ============================================================
// 14. GRADIENT BOOSTED TREES
// stepSize=0.05 for more careful learning
// subsamplingRate=0.8 adds regularization
// ============================================================

val gbtModelY = new GBTRegressor()
  .setLabelCol("label").setFeaturesCol("features")
  .setPredictionCol("prediction_gbt")
  .setMaxIter(200).setMaxDepth(10)
  .setStepSize(0.05).setSubsamplingRate(0.8).setSeed(42)
  .fit(trainDataY)
println("GBT trained.")

// ============================================================
// 15. PREDICTIONS
// ============================================================

val lrPredictionsY  = lrModelY.transform(testDataY)
val rfPredictionsY  = rfModelY.transform(testDataY)
val gbtPredictionsY = gbtModelY.transform(testDataY)

// ============================================================
// 16. EVALUATION FUNCTION
// RMSE/MAE in log-salary units, R² fraction of variance explained
// ============================================================

def evalRegression(
  df: org.apache.spark.sql.DataFrame,
  labelCol: String,
  predCol: String,
  modelName: String
): (Double, Double, Double) = {
  val ev = new RegressionEvaluator()
    .setLabelCol(labelCol).setPredictionCol(predCol)
  val rmse = ev.setMetricName("rmse").evaluate(df)
  val mae  = ev.setMetricName("mae").evaluate(df)
  val r2   = ev.setMetricName("r2").evaluate(df)
  println(s"\n=== $modelName ===")
  println(f"RMSE = $rmse%.4f | MAE = $mae%.4f | R² = $r2%.4f")
  (rmse, mae, r2)
}

// ============================================================
// 17. EVALUATE ALL MODELS
// ============================================================

val baselineMetrics = evalRegression(baselinePredictionsY, "label", "prediction_baseline", "Baseline")
val lrMetrics       = evalRegression(lrPredictionsY,       "label", "prediction_lr",       "Linear Regression")
val rfMetrics       = evalRegression(rfPredictionsY,       "label", "prediction_rf",       "Random Forest")
val gbtMetrics      = evalRegression(gbtPredictionsY,      "label", "prediction_gbt",      "GBT")

// ============================================================
// 18. MODEL COMPARISON
// Uses Seq.minBy/maxBy — avoids shell if/else parsing errors
// ============================================================

println("\n============================================================")
println("MODEL COMPARISON SUMMARY")
println("============================================================")
println(f"Baseline          -> RMSE: ${baselineMetrics._1}%.4f | MAE: ${baselineMetrics._2}%.4f | R2: ${baselineMetrics._3}%.4f")
println(f"Linear Regression -> RMSE: ${lrMetrics._1}%.4f | MAE: ${lrMetrics._2}%.4f | R2: ${lrMetrics._3}%.4f")
println(f"Random Forest     -> RMSE: ${rfMetrics._1}%.4f | MAE: ${rfMetrics._2}%.4f | R2: ${rfMetrics._3}%.4f")
println(f"GBT               -> RMSE: ${gbtMetrics._1}%.4f | MAE: ${gbtMetrics._2}%.4f | R2: ${gbtMetrics._3}%.4f")

val allModelsY = Seq(
  ("Baseline",          baselineMetrics._1, baselineMetrics._2, baselineMetrics._3),
  ("Linear Regression", lrMetrics._1,       lrMetrics._2,       lrMetrics._3),
  ("Random Forest",     rfMetrics._1,       rfMetrics._2,       rfMetrics._3),
  ("GBT",               gbtMetrics._1,      gbtMetrics._2,      gbtMetrics._3)
)

println(s"\nBest by RMSE: ${allModelsY.minBy(_._2)._1}")
println(s"Best by MAE : ${allModelsY.minBy(_._3)._1}")
println(s"Best by R2  : ${allModelsY.maxBy(_._4)._1}")

// ============================================================
// 19. FEATURE IMPORTANCES
// ============================================================

val baseFeatureNamesY = Array(
  "log_views", "log_applies", "engagement",
  "title_len", "title_words",
  "listed_month", "listed_dow", "listed_hour", "is_remote"
)

def getFeatureName(idx: Int): String =
  if (idx < baseFeatureNamesY.length) baseFeatureNamesY(idx)
  else s"encoded_feature_${idx - baseFeatureNamesY.length}"

println("\n=== RF Feature Importances (Top 15) ===")
rfModelY.featureImportances.toArray.zipWithIndex
  .sortBy(-_._1).take(15)
  .foreach { case (imp, idx) =>
    println(f"${getFeatureName(idx)}%-30s -> $imp%.6f")
  }

println("\n=== GBT Feature Importances (Top 15) ===")
gbtModelY.featureImportances.toArray.zipWithIndex
  .sortBy(-_._1).take(15)
  .foreach { case (imp, idx) =>
    println(f"${getFeatureName(idx)}%-30s -> $imp%.6f")
  }

// ============================================================
// 20. BACK-TRANSFORMATION TO SALARY SCALE
// log_salary = log1p(salary) so salary = exp(label) - 1
// ============================================================

println("\n=== GBT Predictions — Salary Scale ===")
gbtPredictionsY
  .withColumn("actual_salary",    exp(col("label")) - 1)
  .withColumn("predicted_salary", exp(col("prediction_gbt")) - 1)
  .select("label", "prediction_gbt", "actual_salary", "predicted_salary")
  .show(15, false)

println("\nMachine Learning phase completed successfully.")
