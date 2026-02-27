import org.apache.spark.sql.functions._

// ============================================================
// IT462 – Phase 2: Data Transformation
// Input : data/reduced_dataset.parquet
// Output: data/preprocessed_dataset.parquet
// ============================================================

val df_reduced = spark.read.parquet("data/reduced_dataset.parquet")
println("\n=== TRANSFORMATION INPUT ===")
println(s"Rows: ${df_reduced.count()}")
println(s"Cols: ${df_reduced.columns.length}")

// -------------------------------------------------------
// STEP 1: Normalize + Remote + Engagement → tmp1
// -------------------------------------------------------
df_reduced
  .withColumn("title_clean",    lower(trim(col("title"))))
  .withColumn("company_clean",  lower(trim(col("company_name"))))
  .withColumn("location_clean", trim(col("location")))
  .withColumn("pay_period_std", upper(trim(col("pay_period"))))
  .withColumn("is_remote",
    when(col("remote_allowed").cast("string") === "1", 1)
    .when(lower(col("remote_allowed").cast("string")) === "true", 1)
    .otherwise(0))
  .withColumn("log_views",   log1p(coalesce(col("views").cast("double"),   lit(0.0))))
  .withColumn("log_applies", log1p(coalesce(col("applies").cast("double"), lit(0.0))))
  .write.mode("overwrite").parquet("data/tmp1.parquet")

println("Step 1 done.")

// -------------------------------------------------------
// STEP 2: Time + State + Title + Salary → preprocessed
// -------------------------------------------------------
spark.read.parquet("data/tmp1.parquet")
  .withColumn("listed_date",  to_date(col("listed_time_ts")))
  .withColumn("listed_month", month(col("listed_time_ts")))
  .withColumn("listed_dow",   dayofweek(col("listed_time_ts")))
  .withColumn("listed_hour",  hour(col("listed_time_ts")))
  .withColumn("state",
    when(col("location_clean").isNull || length(col("location_clean")) === 0, lit("UNKNOWN"))
    .when(size(split(col("location_clean"), ",")) >= 3,
      trim(element_at(split(col("location_clean"), ","), -2)))
    .when(size(split(col("location_clean"), ",")) === 2,
      trim(element_at(split(col("location_clean"), ","), -1)))
    .otherwise(trim(col("location_clean"))))
  .withColumn("title_len",   length(col("title_clean")))
  .withColumn("title_words", size(split(trim(col("title_clean")), "\\s+")))
  .withColumn("normalized_salary", col("normalized_salary").cast("double"))
  .withColumn("log_salary", when(col("normalized_salary").isNotNull, log1p(col("normalized_salary"))))
  .write.mode("overwrite").parquet("data/preprocessed_dataset.parquet")

println("Step 2 done.")
println("Saved: data/preprocessed_dataset.parquet")

// -------------------------------------------------------
// VERIFY + SNAPSHOT
// -------------------------------------------------------
val df = spark.read.parquet("data/preprocessed_dataset.parquet")

println("\n=== TRANSFORMATION OUTPUT ===")
println(s"Rows: ${df.count()}")
println(s"Cols: ${df.columns.length}")

val snap = df.limit(5).cache()

println("\n--- TABLE 1: Job Info ---")
snap.select("job_id", "title_clean", "company_clean", "work_type_std", "is_remote").show(5, false)

println("\n--- TABLE 2: Salary Info ---")
snap.select("job_id", "min_salary", "max_salary", "normalized_salary", "log_salary").show(5, false)

println("\n--- TABLE 3: Location Info ---")
snap.select("job_id", "location_clean", "state", "pay_period_std", "formatted_experience_level").show(5, false)

println("\n--- TABLE 4: Engagement ---")
snap.select("job_id", "views", "log_views", "applies", "log_applies").show(5, false)

println("\n--- TABLE 5: Time & Title ---")
snap.select("job_id", "listed_month", "listed_dow", "listed_hour", "title_len", "title_words").show(5, false)

println("\n=== SCHEMA ===")
df.printSchema()