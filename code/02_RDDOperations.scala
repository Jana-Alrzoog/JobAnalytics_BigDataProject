// ============================================================
// IT462 – Big Data Systems | Phase 3: RDD Operations
// File: 02_RDDOperations.scala
// Dataset: LinkedIn Job Postings
// Focus: Salary analysis and labor market patterns
// ------------------------------------------------------------
// Environment: Apache Spark 3.5.0
//              Scala 2.12.18
//              OpenJDK 11
// ============================================================

// Load the preprocessed dataset from Phase 2
val df = spark.read.parquet("data/preprocessed_dataset.parquet")

// Convert DataFrame to RDD[Row]
val rdd = df.rdd

println("=== PHASE 3 – RDD OPERATIONS AND ANALYSIS TASKS ===")
println(s"Input rows: ${rdd.count()}")
println("=" * 60)


// ============================================================
// TASK 1 – filter
// Keep only FULL-TIME postings with a known normalized salary.
// This creates a consistent subset for salary-focused analysis.
// ============================================================

println("\n--- TASK 1: Filter full-time postings with known salary ---")
println("Goal: Retain only FULL-TIME job postings with non-null normalized_salary")

val rdd_fulltime_salary = rdd.filter { row =>
  val workType = Option(row.getAs[String]("work_type_std")).getOrElse("").trim.toUpperCase
  val salary   = row.getAs[Any]("normalized_salary")
  workType == "FULL-TIME" && salary != null
}.cache()

val task1Count = rdd_fulltime_salary.count()
println(s"Rows after filtering: $task1Count")

println("\nSample output (5 rows):")
rdd_fulltime_salary
  .map(row => (
    row.getAs[Any]("job_id"),
    row.getAs[String]("title_clean"),
    row.getAs[String]("work_type_std"),
    row.getAs[Any]("normalized_salary")
  ))
  .take(5)
  .foreach { case (id, title, workType, salary) =>
    println(f"  job_id=$id | title=$title | work_type=$workType | salary=$salary")
  }

/*
 * INTERPRETATION:
 * This filtering step removes postings that are not full-time or do not
 * contain salary information. It ensures that the later salary analyses
 * are based on comparable job records.
 */


// ============================================================
// TASK 2 – map
// Assign each full-time salaried posting to a salary tier
// (LOW / MID / HIGH) using approximate percentile thresholds.
// This derives an intermediate feature from normalized_salary.
// ============================================================

println("\n\n--- TASK 2: Derive salary tiers from normalized salary ---")
println("Goal: Map each full-time salaried posting into LOW, MID, or HIGH salary tiers")

val salaryValues: Array[Double] = {
  rdd_fulltime_salary
    .flatMap(row => Option(row.getAs[Any]("normalized_salary")).map(_.toString.toDouble))
    .collect()
    .sorted
}

val totalSalaryRows = salaryValues.length
require(totalSalaryRows > 0, "No salary values found after filtering")

val p33 = salaryValues(math.min((totalSalaryRows * 0.33).toInt, totalSalaryRows - 1))
val p67 = salaryValues(math.min((totalSalaryRows * 0.67).toInt, totalSalaryRows - 1))

println("\nApproximate percentile thresholds:")
println(f"  33rd percentile (LOW/MID boundary) : $$${p33}%,.0f")
println(f"  67th percentile (MID/HIGH boundary): $$${p67}%,.0f")

val rdd_salary_bucket = rdd_fulltime_salary.map { row =>
  val jobId    = row.getAs[Any]("job_id")
  val title    = Option(row.getAs[String]("title_clean")).getOrElse("unknown")
  val location = Option(row.getAs[String]("state")).getOrElse("UNKNOWN")
  val salary   = Option(row.getAs[Any]("normalized_salary")).map(_.toString.toDouble).getOrElse(0.0)

  val bucket =
    if (salary < p33) "LOW"
    else if (salary < p67) "MID"
    else "HIGH"

  (jobId, title, location, salary, bucket)
}

println("\nSample output (5 rows):")
rdd_salary_bucket
  .take(5)
  .foreach { case (id, title, location, salary, bucket) =>
    println(f"  job_id=$id | location=$location%-15s | salary=$$${salary}%,.0f | bucket=$bucket")
  }

println("\nSalary tier distribution:")
val bucketOrder = Map("LOW" -> 0, "MID" -> 1, "HIGH" -> 2)

rdd_salary_bucket
  .map { case (_, _, _, _, bucket) => (bucket, 1) }
  .reduceByKey(_ + _)
  .collect()
  .sortBy { case (bucket, _) => bucketOrder.getOrElse(bucket, 99) }
  .foreach { case (bucket, count) =>
    val pct = count.toDouble / totalSalaryRows * 100
    println(f"  $bucket%-5s : $count%,d postings (${pct}%.1f%%)")
  }

/*
 * INTERPRETATION:
 * This task converts raw salary values into broader salary categories,
 * making the compensation pattern easier to interpret. It also creates
 * a useful derived feature for later comparison and analysis.
 */


// ============================================================
// TASK 3 – flatMap
// Split cleaned job titles into individual keywords and count them
// to identify the most common role-related terms in the dataset.
// ============================================================

println("\n\n--- TASK 3: Find top job title keywords by frequency ---")
println("Goal: Extract and count the most frequent keywords appearing in job titles")

val stopWords = Set(
  "and", "of", "the", "for", "in", "a", "an", "to",
  "or", "with", "at", "&", "-", "/", "i", "ii", "iii",
  "full", "time"
)

val rdd_title_words = rdd
  .filter(row => Option(row.getAs[String]("title_clean")).exists(_.trim.nonEmpty))
  .flatMap { row =>
    val title = row.getAs[String]("title_clean").trim
    title
      .split("\\s+")
      .map(_.toLowerCase.replaceAll("[^a-z]", ""))
      .filter(word => word.length > 2 && !stopWords.contains(word))
      .map(word => (word, 1))
  }

val topTitleWords = rdd_title_words
  .reduceByKey(_ + _)
  .sortBy(_._2, ascending = false)
  .take(20)

println("\nTop 20 job title keywords:")
topTitleWords.zipWithIndex.foreach { case ((word, count), i) =>
  println(f"  ${i + 1}%2d. $word%-25s : $count%,d occurrences")
}

/*
 * INTERPRETATION:
 * This task shows the most common job title terms in the dataset,
 * which helps identify dominant role types and hiring trends in
 * the labor market data.
 */


// ============================================================
// TASK 4 – reduceByKey
// Aggregate posting statistics by location to compare regions
// in terms of posting volume, total applications, and average salary.
// ============================================================

println("\n\n--- TASK 4: Compute location-level posting and salary statistics ---")
println("Goal: Calculate total postings, total applies, and average salary per location")

val rdd_location_raw = rdd
  .filter(row => Option(row.getAs[String]("state")).exists(_.trim.nonEmpty))
  .map { row =>
    val location   = row.getAs[String]("state").trim.toUpperCase
    val applies    = Option(row.getAs[Any]("applies")).map(_.toString.toDouble).getOrElse(0.0)
    val salaryOpt  = Option(row.getAs[Any]("normalized_salary")).map(_.toString.toDouble)
    val salaryVal  = salaryOpt.getOrElse(0.0)
    val salaryFlag = if (salaryOpt.isDefined) 1 else 0

    (location, (applies, salaryVal, salaryFlag, 1))
  }

val rdd_location_agg = rdd_location_raw
  .reduceByKey { case ((a1, s1, sc1, pc1), (a2, s2, sc2, pc2)) =>
    (a1 + a2, s1 + s2, sc1 + sc2, pc1 + pc2)
  }
  .map { case (location, (totalApplies, totalSalary, salaryCount, postingCount)) =>
    val avgSalary = if (salaryCount > 0) totalSalary / salaryCount else 0.0
    (location, totalApplies, avgSalary, postingCount)
  }
  .cache()

println("\nTop 15 locations by total job postings:")
println(f"  ${"Location"}%-30s | ${"Postings"}%9s | ${"Total Applies"}%13s | ${"Avg Salary"}%12s")
println("  " + "-" * 78)

rdd_location_agg
  .sortBy(_._4, ascending = false)
  .take(15)
  .foreach { case (location, applies, avgSalary, postingCount) =>
    println(f"  $location%-30s | $postingCount%9d | $applies%13.0f | $$${avgSalary}%11.0f")
  }

/*
 * INTERPRETATION:
 * Grouping postings by location reveals differences in labor market
 * activity across regions. It allows direct comparison of how many jobs,
 * applications, and salary levels are associated with each location.
 */


// ============================================================
// TASK 5 – sortByKey
// Rank locations by average salary to identify higher-paying
// and lower-paying job markets.
// ============================================================

println("\n\n--- TASK 5: Rank locations by average salary ---")
println("Goal: Sort locations from highest to lowest average salary")

val rdd_sorted_by_salary = rdd_location_agg
  .filter { case (_, _, avgSalary, postingCount) => avgSalary > 0 && postingCount >= 50 }
  .map { case (location, applies, avgSalary, postingCount) =>
    (avgSalary, (location, applies, postingCount))
  }
  .sortByKey(ascending = false)

println("\nTop 20 locations by average salary (minimum 50 postings):")
println(f"  ${"Rank"}%5s | ${"Location"}%-30s | ${"Avg Salary"}%12s | ${"Postings"}%9s")
println("  " + "-" * 66)

rdd_sorted_by_salary
  .take(20)
  .zipWithIndex
  .foreach { case ((avgSalary, (location, _, postingCount)), i) =>
    println(f"  ${i + 1}%5d | $location%-30s | $$${avgSalary}%11.0f | $postingCount%9d")
  }

println("\nBottom 10 locations by average salary (minimum 50 postings):")
println(f"  ${"Rank"}%5s | ${"Location"}%-30s | ${"Avg Salary"}%12s | ${"Postings"}%9s")
println("  " + "-" * 66)

rdd_sorted_by_salary
  .collect()
  .takeRight(10)
  .reverse
  .zipWithIndex
  .foreach { case ((avgSalary, (location, _, postingCount)), i) =>
    println(f"  ${i + 1}%5d | $location%-30s | $$${avgSalary}%11.0f | $postingCount%9d")
  }

/*
 * INTERPRETATION:
 * Sorting locations by average salary highlights geographic differences
 * in compensation. This helps identify which job markets tend to offer
 * higher or lower salaries within the dataset.
 */


println("\n=== PHASE 3 RDD ANALYSIS COMPLETED ===")
