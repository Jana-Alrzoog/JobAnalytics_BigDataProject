import org.apache.spark.sql.functions._

// ============================================================
// IT462 – Big Data Systems | Phase 2: Cleaning + Reduction
// File: cleaning.scala
// Dataset: LinkedIn Job Postings
// ============================================================


// ============================================================
// 1) LOAD RAW DATA
// ============================================================

val df = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .option("multiLine", "true")
  .option("quote", "\"")
  .option("escape", "\"")
  .option("mode", "PERMISSIVE")
  .csv("data/raw_dataset.csv")

val initialCount = df.count()


// ============================================================
// 2) DATA CLEANING
// ============================================================

// Duplicates
val df_no_dupes = df.dropDuplicates(Seq("job_id"))
val dupesCount = df_no_dupes.count()

// Missing critical fields
val df_dropped_nulls = df_no_dupes.na.drop(Seq("job_id", "title", "company_name"))
val nullsDroppedCount = df_dropped_nulls.count()

// Impute selected numeric nulls
val df_imputed = df_dropped_nulls.na.fill(0.0, Seq("views", "applies", "remote_allowed"))

// Errors / inconsistencies / outliers
val df_valid = df_imputed
  .filter(col("max_salary").isNull || (col("max_salary") > 0 && col("max_salary") <= 5000000))
  .filter(col("views") >= 0)
  .filter(col("applies") >= 0)
  .filter(col("min_salary").isNull || col("max_salary").isNull || col("min_salary") <= col("max_salary"))

val validCount = df_valid.count()

// Standardization
val df_clean = df_valid
  .withColumn(
    "listed_time_ts",
    when(col("listed_time").isNotNull,
      to_timestamp(from_unixtime(col("listed_time").cast("long") / 1000))
    )
  )
  .withColumn(
    "closed_time_ts",
    when(col("closed_time").isNotNull,
      to_timestamp(from_unixtime(col("closed_time").cast("long") / 1000))
    )
  )
  .withColumn("work_type_std", upper(trim(col("formatted_work_type"))))
  .withColumn("currency_std", upper(trim(col("currency"))))
  .drop("listed_time", "closed_time")

df_clean.write.mode("overwrite").parquet("data/cleaned_data.parquet")

println(s"=== CLEANING SUMMARY ===")
println(s"Raw rows        : $initialCount")
println(s"After dedup     : $dupesCount  (removed: ${initialCount - dupesCount})")
println(s"After null drop : $nullsDroppedCount  (removed: ${dupesCount - nullsDroppedCount})")
println(s"After outliers  : $validCount  (removed: ${nullsDroppedCount - validCount})")
println(s"Columns         : ${df.columns.length} → ${df_clean.columns.length}")


// ============================================================
// 3) DATA REDUCTION
// ============================================================

val rowsBefore = df_clean.count()
val colsBefore = df_clean.columns.length

println("\n=== DATA REDUCTION (BEFORE) ===")
println(s"Rows : $rowsBefore")
println(s"Cols : $colsBefore")

// ------------------------------------------------------------
// STEP 1: FEATURE SELECTION
// ------------------------------------------------------------
val columnsToDrop = Seq(
  "company_id",           // Removed per request
  "job_posting_url",
  "application_url",
  "application_type",
  "posting_domain",
  "zip_code",
  "fips",
  "description",
  "skills_desc",
  "compensation_type",
  "formatted_work_type",  
  "currency",             
  "sponsored"             
)

val df_reduced = df_clean.drop(columnsToDrop.filter(df_clean.columns.contains): _*)

val rowsAfterFS = df_reduced.count()
val colsAfterFS = df_reduced.columns.length

println("\n--- STEP 1: FEATURE SELECTION ---")
println(s"Rows: $rowsBefore → $rowsAfterFS (removed: ${rowsBefore - rowsAfterFS})")
println(s"Cols: $colsBefore → $colsAfterFS (removed: ${colsBefore - colsAfterFS})")


// ------------------------------------------------------------
// STEP 2: TARGET-BASED ROW REDUCTION (ML DATASET ONLY)
// ------------------------------------------------------------
val df_salary = df_reduced.filter(col("normalized_salary").isNotNull)
val salaryRows = df_salary.count()

println("\n--- STEP 2: SALARY FILTER (ML ONLY) ---")
println(s"Rows: $rowsAfterFS → $salaryRows (removed: ${rowsAfterFS - salaryRows}, " +
  f"${(rowsAfterFS - salaryRows).toDouble / rowsAfterFS * 100}%.2f%%)")


// ------------------------------------------------------------
// STEP 3: SALARY OUTLIER TRIMMING (ML DATASET ONLY)
// ------------------------------------------------------------
val MIN_SALARY = 10000.0
val MAX_SALARY = 1000000.0

val df_ml = df_salary.filter(
  col("normalized_salary") >= MIN_SALARY &&
  col("normalized_salary") <= MAX_SALARY
)
val mlRows = df_ml.count()

println("\n--- STEP 3: SALARY TRIM (ML ONLY) ---")
println(s"Rows: $salaryRows → $mlRows (removed: ${salaryRows - mlRows})")

df_ml.select("normalized_salary")
  .summary("count","min","25%","50%","75%","max","mean")
  .show(false)


// ------------------------------------------------------------
// STEP 4: AGGREGATION (RUBRIC)
// ------------------------------------------------------------
val df_state = df_reduced.withColumn("state",
  when(size(split(col("location"), ",")) >= 3, trim(element_at(split(col("location"), ","), -2)))
    .when(size(split(col("location"), ",")) === 2, trim(element_at(split(col("location"), ","), -1)))
    .otherwise(trim(col("location")))
)

val agg_state_counts = df_state
  .groupBy("state")
  .agg(
    count("*").alias("total_postings"),
    sum("applies").alias("total_applies"),
    countDistinct("company_name").alias("distinct_companies") 
  )

val agg_state_salary = df_state
  .filter(col("normalized_salary").isNotNull)
  .groupBy("state")
  .agg(
    count("*").alias("salary_postings"),
    round(avg("normalized_salary"), 2).alias("avg_salary"),
    round(percentile_approx(col("normalized_salary"), lit(0.5), lit(10000)), 2).alias("median_salary")
  )

val df_agg_location = agg_state_counts
  .join(agg_state_salary, Seq("state"), "left")
  .orderBy(desc("total_postings"))

println("\n--- STEP 4: AGGREGATION OUTPUTS ---")
println(s"By state rows    : ${df_agg_location.count()}")
println("\nTop 15 states:");     df_agg_location.show(15, false)


// ------------------------------------------------------------
// STEP 5: OPTIONAL SAMPLE (EDA ONLY)
// ------------------------------------------------------------
val df_sample = df_reduced
  .withColumn("salary_present", when(col("normalized_salary").isNotNull, "yes").otherwise("no"))
  .stat.sampleBy("salary_present", Map("yes" -> 0.20, "no" -> 0.20), 42L)
  .drop("salary_present")

val sampleRows = df_sample.count()

println("\n--- STEP 5: SAMPLE (OPTIONAL) ---")
println(s"Sample rows (20%): $sampleRows")


// ============================================================
// SAVE OUTPUTS
// ============================================================

df_reduced.write.mode("overwrite").parquet("data/reduced_dataset.parquet")
df_ml.write.mode("overwrite").parquet("data/ml_dataset_salary.parquet")
df_agg_location.write.mode("overwrite").parquet("data/agg_by_location.parquet")
df_sample.write.mode("overwrite").parquet("data/sample_20pct.parquet")

println("\nSaved:")
println(" - data/cleaned_data.parquet")
println(" - data/reduced_dataset.parquet")
println(" - data/ml_dataset_salary.parquet            (ML later)")
println(" - data/agg_by_location.parquet             (report/SQL)")
println(" - data/sample_20pct.parquet                (EDA only)")

// ============================================================
// 4) DATA TRANSFORMATION
// ============================================================

println("\n=== TRANSFORMATION INPUT ===")
println(s"Rows: ${df_reduced.count()}")
println(s"Cols: ${df_reduced.columns.length}")

// ---------------------------------------------------
// (A) Text Normalization
// ---------------------------------------------------
df_reduced
  .withColumn("title_clean",    lower(trim(col("title"))))
  .withColumn("company_clean",  lower(trim(col("company_name"))))
  .withColumn("location_clean", lower(trim(col("location")))) 
  .withColumn("pay_period_std", upper(trim(col("pay_period"))))
// ---------------------------------------------------
// (B) Binary Encoding – Remote Flag
// ---------------------------------------------------
  .withColumn("is_remote",
    when(col("remote_allowed").cast("string") === "1", 1)
    .when(lower(col("remote_allowed").cast("string")) === "true", 1)
    .otherwise(0))
// ---------------------------------------------------
// (C) Log Transform – Engagement Features
// ---------------------------------------------------
  .withColumn("log_views",   log1p(coalesce(col("views").cast("double"),   lit(0.0))))
  .withColumn("log_applies", log1p(coalesce(col("applies").cast("double"), lit(0.0))))

// ---------------------------------------------------
// (D) DateTime Decomposition
// ---------------------------------------------------
  .withColumn("listed_date",  to_date(col("listed_time_ts")))
  .withColumn("listed_month", month(col("listed_time_ts")))
  .withColumn("listed_dow",   dayofweek(col("listed_time_ts")))
  .withColumn("listed_hour",  hour(col("listed_time_ts")))
// ---------------------------------------------------
// (E) Feature Extraction – State from Location
// ---------------------------------------------------
  .withColumn("state",
    when(col("location_clean").isNull || length(col("location_clean")) === 0, lit("UNKNOWN"))
    .when(size(split(col("location_clean"), ",")) >= 3,
      trim(element_at(split(col("location_clean"), ","), -2)))
    .when(size(split(col("location_clean"), ",")) === 2,
      trim(element_at(split(col("location_clean"), ","), -1)))
    .otherwise(trim(col("location_clean"))))
// ---------------------------------------------------
// (F) Feature Engineering – Title Features
// ---------------------------------------------------
  .withColumn("title_len",   length(col("title_clean")))
  .withColumn("title_words", size(split(trim(col("title_clean")), "\\s+")))
// ---------------------------------------------------
// (G) Log Transform – Salary
// ---------------------------------------------------
  .withColumn("normalized_salary", col("normalized_salary").cast("double"))
  .withColumn("log_salary",
    when(col("normalized_salary").isNotNull, log1p(col("normalized_salary"))))
  .write.mode("overwrite").parquet("data/preprocessed_dataset.parquet")

val df_final = spark.read.parquet("data/preprocessed_dataset.parquet")

println("\n=== TRANSFORMATION OUTPUT ===")
println(s"Rows: ${df_final.count()}")
println(s"Cols: ${df_final.columns.length}")

val snap = df_final.limit(10).cache()

println("\n--- TABLE 1: Job Info ---")
snap.select("job_id", "title_clean", "company_clean", "work_type_std", "is_remote").show(10, false)

println("\n--- TABLE 2: Salary Info ---")
snap.select("job_id", "min_salary", "max_salary", "normalized_salary", "log_salary").show(10, false)

println("\n--- TABLE 3: Location Info ---")
snap.select("job_id", "location_clean", "state", "pay_period_std", "formatted_experience_level").show(10, false)

println("\n--- TABLE 4: Engagement ---")
snap.select("job_id", "views", "log_views", "applies", "log_applies").show(10, false)

println("\n--- TABLE 5: Time & Title ---")
snap.select("job_id", "listed_month", "listed_dow", "listed_hour", "title_len", "title_words").show(10, false)

println("\n=== SCHEMA: FINAL PREPROCESSED DATASET ===")
df_final.printSchema()
