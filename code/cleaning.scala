import org.apache.spark.sql.functions._

// 1. LOAD THE DATA
val df = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .option("multiLine", "true")
  .option("quote", "\"")
  .option("escape", "\"")
  .option("mode", "PERMISSIVE")
  .csv("data/raw_dataset.csv")

val initialCount = df.count()

// 2. APPLY RUBRIC CLEANING RULES

// Duplicates
// Detect and remove duplicate records based on the unique job_id
val df_no_dupes = df.dropDuplicates(Seq("job_id"))
val dupesCount = df_no_dupes.count()

// Missing Values
// Drop records missing critical text fields (decision: drop unrecoverable rows)
val df_dropped_nulls = df_no_dupes.na.drop(Seq("job_id", "title", "company_name"))
val nullsDroppedCount = df_dropped_nulls.count()

// Impute missing numerical fields with 0.0 (decision: missing implies zero)
val df_imputed = df_dropped_nulls.na.fill(0.0, Seq("views", "applies", "remote_allowed"))

// Errors, Inconsistencies, and Outliers
// - Remove negative views/applies (impossible values)
// - Ensure min_salary is not greater than max_salary (inconsistent logic)
// - Cap max_salary at 5,000,000 to remove fake/spam outlier postings (outliers)
val df_valid = df_imputed
  .filter(col("max_salary").isNull || (col("max_salary") > 0 && col("max_salary") <= 5000000))
  .filter(col("views") >= 0)
  .filter(col("applies") >= 0)
  .filter(col("min_salary").isNull || col("max_salary").isNull || col("min_salary") <= col("max_salary"))
val validCount = df_valid.count()

// Standardization
// - Convert epoch dates to standard SQL Timestamps
// - Convert work_type and currency to uppercase for consistent format
val df_clean = df_valid
  .withColumn("listed_time_ts", to_timestamp(col("listed_time") / 1000))
  .withColumn("closed_time_ts", to_timestamp(col("closed_time") / 1000))
  .withColumn("work_type_std", upper(col("formatted_work_type")))
  .withColumn("currency_std", upper(col("currency")))
  .drop("listed_time", "closed_time")
