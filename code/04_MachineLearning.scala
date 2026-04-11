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

// Load dataset from Phase 2
val df = spark.read.parquet("data/preprocessed_dataset.parquet")
println(s"Total rows loaded: ${df.count()}")

// ============================================================
// Step 1: Filter valid salary data
// ============================================================

val df_ml = df
  .filter(col("normalized_salary").isNotNull)
  .filter(col("normalized_salary") >= 10000.0 && col("normalized_salary") <= 1000000.0)

println(s"Rows after salary filtering: ${df_ml.count()}")

// ============================================================
// Step 2: Select relevant features
// ============================================================

val modelInput = df_ml.select(
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

// Drop nulls only from important columns
val cleanInput = modelInput.na.drop(
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

println(s"Rows after cleaning: ${cleanInput.count()}")

// ============================================================
// Step 3: Train/Test Split
// ============================================================

val Array(trainRaw, testRaw) = cleanInput.randomSplit(Array(0.7, 0.3), 42)

println(s"Training rows: ${trainRaw.count()}")
println(s"Test rows: ${testRaw.count()}")

// ============================================================
// Step 4: Encode categorical feature (work_type_std)
// ============================================================

val workTypeIndexer = new StringIndexer()
  .setInputCol("work_type_std")
  .setOutputCol("work_type_index")
  .setHandleInvalid("keep")

val indexerModel = workTypeIndexer.fit(trainRaw)

val trainIndexed = indexerModel.transform(trainRaw)
val testIndexed = indexerModel.transform(testRaw)

// One-hot encoding
val encoder = new OneHotEncoder()
  .setInputCols(Array("work_type_index"))
  .setOutputCols(Array("work_type_vec"))

val encoderModel = encoder.fit(trainIndexed)

val trainEncoded = encoderModel.transform(trainIndexed)
val testEncoded = encoderModel.transform(testIndexed)

// ============================================================
// Step 5: Assemble features
// ============================================================

val assembler = new VectorAssembler()
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

val trainAssembled = assembler.transform(trainEncoded)
val testAssembled = assembler.transform(testEncoded)

// ============================================================
// Step 6: Feature scaling
// ============================================================

val scaler = new StandardScaler()
  .setInputCol("features_raw")
  .setOutputCol("features")
  .setWithMean(true)
  .setWithStd(true)

val scalerModel = scaler.fit(trainAssembled)

val trainFinal = scalerModel.transform(trainAssembled)
val testFinal = scalerModel.transform(testAssembled)

// ============================================================
// Step 7: Final dataset for ML
// ============================================================

val trainData = trainFinal.withColumnRenamed("normalized_salary", "label")
val testData = testFinal.withColumnRenamed("normalized_salary", "label")

trainData.select("label", "features").show(5, false)
testData.select("label", "features").show(5, false)

// ============================================================
// End of Phase 5 – Machine Learning (Person 1)
// ============================================================
