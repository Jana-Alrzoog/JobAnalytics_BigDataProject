// ============================================================
// IT462 – Big Data Systems | Phase 4: SQL Operations
// File: 03_SQLOperations.scala
// Dataset: LinkedIn Job Postings
// Focus: Salary analysis using Spark SQL and DataFrames
// ------------------------------------------------------------
// Environment: Apache Spark 3.5.0
//              Scala 2.12.18
//              OpenJDK 11
// ============================================================

// Load the preprocessed dataset from Phase 2.
val df = spark.read.parquet("data/preprocessed_dataset.parquet")

// Register the DataFrame as a temporary view so we can query it using SQL.
df.createOrReplaceTempView("job_postings")
println(s"Total rows loaded: ${df.count()}")

// ============================================================
// Q1: Average Salary by Work Type and Pay Period
// Question: What is the average salary for each combination
//           of employment type and pay period?
// Techniques: GROUP BY, HAVING, aggregation functions
// ============================================================

println("\n==== Q1: Average Salary by Work Type and Pay Period ====")

spark.sql("""
  SELECT
    work_type_std,
    pay_period_std,
    COUNT(*)                          AS total_postings,
    ROUND(AVG(normalized_salary), 2)  AS avg_salary,
    ROUND(MIN(normalized_salary), 2)  AS min_salary,
    ROUND(MAX(normalized_salary), 2)  AS max_salary
  FROM job_postings
  WHERE normalized_salary IS NOT NULL
    AND work_type_std     IS NOT NULL
    AND pay_period_std    IS NOT NULL
  GROUP BY work_type_std, pay_period_std
  HAVING COUNT(*) > 50
  ORDER BY avg_salary DESC
""").show()

// ============================================================
// Q2: Top 10 States by Average Salary
// Question: Which states offer the highest average salaries,
//           and how many job postings do they have?
// Techniques: GROUP BY, HAVING, ORDER BY, LIMIT, PERCENTILE
// ============================================================

println("\n==== Q2: Top 10 States by Average Salary ====")

spark.sql("""
  SELECT
    state,
    COUNT(*)                                              AS total_postings,
    ROUND(AVG(normalized_salary), 2)                     AS avg_salary,
    ROUND(PERCENTILE_APPROX(normalized_salary, 0.5), 2)  AS median_salary,
    CAST(SUM(applies) AS BIGINT)                         AS total_applications
  FROM job_postings
  WHERE normalized_salary IS NOT NULL
    AND state IS NOT NULL
  GROUP BY state
  HAVING COUNT(*) >= 100
  ORDER BY avg_salary DESC
  LIMIT 10
""").show()

// ============================================================
// Q3: Salary by Experience Level vs Overall Average
// Question: How does the average salary at each experience
//           level compare to the overall average salary?
// Techniques: GROUP BY, Window Function (AVG OVER)
// ============================================================

println("\n==== Q3: Salary by Experience Level vs Overall Average ====")

spark.sql("""
  SELECT
    formatted_experience_level,
    COUNT(*)                                        AS postings_count,
    ROUND(AVG(normalized_salary), 2)               AS avg_salary,
    ROUND(AVG(AVG(normalized_salary)) OVER (), 2)  AS overall_avg_salary,
    ROUND(
      AVG(normalized_salary) -
      AVG(AVG(normalized_salary)) OVER (), 2
    )                                               AS diff_from_overall
  FROM job_postings
  WHERE normalized_salary          IS NOT NULL
    AND formatted_experience_level IS NOT NULL
  GROUP BY formatted_experience_level
  ORDER BY avg_salary DESC
""").show()

// ============================================================
// Q4: Top 15 Most Common Job Titles and Their Salaries
// Question: What are the most frequently posted job titles,
//           and what is the average salary for each?
// Techniques: CTE (WITH), Window Function (RANK OVER),
//             GROUP BY, HAVING, COUNT DISTINCT
// ============================================================

println("\n==== Q4: Top 15 Most Common Job Titles and Their Salaries ====")

spark.sql("""
  WITH title_stats AS (
    SELECT
      title_clean,
      COUNT(*)                          AS posting_count,
      ROUND(AVG(normalized_salary), 2)  AS avg_salary,
      COUNT(DISTINCT company_name)      AS distinct_companies
    FROM job_postings
    WHERE title_clean IS NOT NULL
    GROUP BY title_clean
    HAVING COUNT(*) >= 30
  ),
  ranked_titles AS (
    SELECT *,
      RANK() OVER (ORDER BY posting_count DESC) AS popularity_rank
    FROM title_stats
  )
  SELECT *
  FROM ranked_titles
  WHERE popularity_rank <= 15
  ORDER BY popularity_rank
""").show()

// ============================================================
// Q5: Job Engagement Analysis by Work Type
// Question: Which employment type attracts the most applicants,
//           and what is the application-to-view conversion rate?
// Techniques: GROUP BY, aggregation functions, derived metric
// ============================================================

println("\n==== Q5: Job Engagement Analysis by Work Type ====")

spark.sql("""
  SELECT
    work_type_std                                   AS work_type,
    COUNT(*)                                        AS total_postings,
    ROUND(AVG(normalized_salary), 2)               AS avg_salary,
    ROUND(AVG(applies), 2)                          AS avg_applications,
    ROUND(AVG(views), 2)                            AS avg_views,
    ROUND(
      AVG(applies) / NULLIF(AVG(views), 0)
      * 100, 2)                                     AS application_rate_pct
  FROM job_postings
  WHERE work_type_std IS NOT NULL
  GROUP BY work_type_std
  ORDER BY avg_applications DESC
""").show()

// ============================================================
// End of Phase 4 – SQL Operations
// ============================================================
