// ============================================================
// IT462 – Big Data Systems | Phase 3: RDD Operations
// File: 02_RDDOperations.scala
// Dataset: LinkedIn Job Postings
// Focus: Salary analysis & labor market patterns
// ------------------------------------------------------------
// Environment: Apache Spark 3.5.0
//              Scala 2.12.18
//              OpenJDK 11
// ============================================================

// Load the preprocessed dataset (output of Phase 2)
val df = spark.read.parquet("data/preprocessed_dataset.parquet")

// Convert to RDD of Row for all transformations
val rdd = df.rdd

println("=== PHASE 3 – RDD TRANSFORMATIONS ===")
println(s"Input rows: ${rdd.count()}")
println("=" * 50)


// ============================================================
// TRANSFORMATION 1 – filter
// Retain FULL-TIME postings with a non-null normalized_salary.
// Scopes all downstream RDD analysis to the same population
// used for salary prediction.
// ============================================================

println("\n--- TRANSFORMATION 1: filter ---")
println("Goal: Retain only FULL-TIME postings with a known normalized_salary")

val rdd_fulltime_salary = rdd.filter { row =>
  val workType = Option(row.getAs[String]("work_type_std")).getOrElse("").trim.toUpperCase
  val salary   = row.getAs[Any]("normalized_salary")
  workType == "FULL-TIME" && salary != null
}.cache()

val t1Count = rdd_fulltime_salary.count()
println(s"Rows after filter: $t1Count")

println("\nSample output (5 rows):")
{
  rdd_fulltime_salary
    .map(row => (
      row.getAs[Any]("job_id"),
      row.getAs[String]("title_clean"),
      row.getAs[String]("work_type_std"),
      row.getAs[Any]("normalized_salary")
    ))
    .take(5)
    .foreach { case (id, title, wtype, sal) =>
      println(f"  job_id=$id | title=$title | work_type=$wtype | salary=$sal")
    }
}

/*
 * INTERPRETATION:
 * Restricting to full-time roles with known salaries removes structurally
 * incomparable records from all subsequent aggregations. This file reads
 * preprocessed_dataset.parquet prior to the extreme-value trimming applied
 * in Phase 2, so the row count exceeds the 35,118-row
 * ml_dataset_salary.parquet used in Phase 5.
 */


// ============================================================
// TRANSFORMATION 2 – map
// Assign each posting a salary tier (LOW / MID / HIGH) using
// data-derived thresholds (approx. 33rd and 67th percentiles)
// rather than fixed dollar values, which are not role-neutral.
// ============================================================

println("\n\n--- TRANSFORMATION 2: map ---")
println("Goal: Map each full-time salaried posting to a data-driven salary tier")

val salaryValues: Array[Double] = {
  rdd_fulltime_salary
    .flatMap(row => Option(row.getAs[Any]("normalized_salary")).map(_.toString.toDouble))
    .collect()
    .sorted
}

val totalSalaryRows = salaryValues.length

require(totalSalaryRows > 0, "No salary rows found after filtering — cannot compute percentiles")
val p33 = salaryValues(math.min((totalSalaryRows * 0.33).toInt, totalSalaryRows - 1))
val p67 = salaryValues(math.min((totalSalaryRows * 0.67).toInt, totalSalaryRows - 1))

println(f"\n  Approximate percentile thresholds computed from sorted salary values:")
println(f"    33rd percentile (LOW/MID boundary) : $$${p33}%,.0f")
println(f"    67th percentile (MID/HIGH boundary): $$${p67}%,.0f")

val rdd_salary_bucket = rdd_fulltime_salary.map { row =>
  val jobId  = row.getAs[Any]("job_id")
  val title  = Option(row.getAs[String]("title_clean")).getOrElse("unknown")
  val location = Option(row.getAs[String]("state")).getOrElse("UNKNOWN")
  val salary = Option(row.getAs[Any]("normalized_salary"))
                 .map(_.toString.toDouble).getOrElse(0.0)

  val bucket =
    if (salary < p33) "LOW"
    else if (salary < p67) "MID"
    else "HIGH"

  (jobId, title, location, salary, bucket)
}

println("\nSample output (5 rows):")
{
  rdd_salary_bucket
    .take(5)
    .foreach { case (id, title, location, sal, bucket) =>
      println(f"  job_id=$id | location=$location%-15s | salary=$$${sal}%,.0f | bucket=$bucket")
    }
}

println("\nSalary tier distribution (expected ~33% each):")
{
  val bucketOrder = Map("LOW" -> 0, "MID" -> 1, "HIGH" -> 2)

  rdd_salary_bucket
    .map { case (_, _, _, _, bucket) => (bucket, 1) }
    .reduceByKey(_ + _)
    .collect()
    .sortBy { case (bucket, _) => bucketOrder.getOrElse(bucket, 99) }
    .foreach { case (bucket, cnt) =>
      val pct = cnt.toDouble / totalSalaryRows * 100
      println(f"  $bucket%-5s : $cnt%,d postings (${pct}%.1f%%)")
    }
}

/*
 * INTERPRETATION:
 * Percentile-based tiers reflect relative compensation within this
 * dataset rather than external benchmarks. Deviation from an equal
 * three-way split is attributable to salary clustering at round numbers
 * near the boundary thresholds. The distribution characterises the
 * target variable range the Phase 5 regression model predicts across.
 */


// ============================================================
// TRANSFORMATION 3 – flatMap
// Tokenise job titles into individual keywords to identify
// the most frequent role types across all postings.
// ============================================================

println("\n\n--- TRANSFORMATION 3: flatMap ---")
println("Goal: Extract and count individual keywords from job titles")

val stopWords = Set(
  "and", "of", "the", "for", "in", "a", "an", "to",
  "or", "with", "at", "&", "-", "/", "i", "ii", "iii",
  "full", "time"
)

val rdd_title_words = {
  rdd
    .filter(row => Option(row.getAs[String]("title_clean")).exists(_.trim.nonEmpty))
    .flatMap { row =>
      val title = row.getAs[String]("title_clean").trim
      title
        .split("\\s+")
        .map(_.toLowerCase.replaceAll("[^a-z]", ""))
        .filter(w => w.length > 2 && !stopWords.contains(w))
        .map(word => (word, 1))
    }
}

val topTitleWords = {
  rdd_title_words
    .reduceByKey(_ + _)
    .sortBy(_._2, ascending = false)
    .take(20)
}

println("\nTop 20 job title keywords:")
topTitleWords.zipWithIndex.foreach { case ((word, count), i) =>
  println(f"  ${i + 1}%2d. $word%-25s : $count%,d occurrences")
}

/*
 * INTERPRETATION:
 * The keyword frequency distribution reveals the dominant role types
 * in the dataset and contextualises the title_len and title_words
 * features engineered in Phase 2 as potential predictors in Phase 5.
 */


// ============================================================
// TRANSFORMATION 4 – reduceByKey
// Aggregate total applications and average salary by location
// to examine the relationship between hiring volume and
// compensation across geographies.
// ============================================================

println("\n\n--- TRANSFORMATION 4: reduceByKey ---")
println("Goal: Compute total applies and average normalized salary per location")

val rdd_location_raw = {
  rdd
    .filter(row => Option(row.getAs[String]("state")).exists(_.trim.nonEmpty))
    .map { row =>
      val location   = row.getAs[String]("state").trim.toUpperCase
      val applies    = Option(row.getAs[Any]("applies")).map(_.toString.toDouble).getOrElse(0.0)
      val salaryOpt  = Option(row.getAs[Any]("normalized_salary")).map(_.toString.toDouble)
      val salaryVal  = salaryOpt.getOrElse(0.0)
      val salaryFlag = if (salaryOpt.isDefined) 1 else 0

      (location, (applies, salaryVal, salaryFlag, 1))
    }
}

val rdd_location_agg = {
  rdd_location_raw
    .reduceByKey { case ((a1, s1, sc1, pc1), (a2, s2, sc2, pc2)) =>
      (a1 + a2, s1 + s2, sc1 + sc2, pc1 + pc2)
    }
    .map { case (location, (totalApplies, totalSalary, salaryCount, postingCount)) =>
      val avgSalary = if (salaryCount > 0) totalSalary / salaryCount else 0.0
      (location, totalApplies, avgSalary, postingCount)
    }
    .cache()
}

println("\nTop 15 locations by total job postings:")
println(f"  ${"Location"}%-30s | ${"Postings"}%9s | ${"Total Applies"}%13s | ${"Avg Salary"}%12s")
println("  " + "-" * 78)
{
  rdd_location_agg
    .sortBy(_._4, ascending = false)
    .take(15)
    .foreach { case (location, applies, avgSal, count) =>
      println(f"  $location%-30s | $count%9d | $applies%13.0f | $$${avgSal}%11.0f")
    }
}

/*
 * INTERPRETATION:
 * The per-location aggregation enables direct comparison of posting volume
 * against average compensation, supporting the project's analysis of
 * how geographic location influences salary. Because this field contains
 * mixed values (states, broader regions, and named areas), it is treated
 * here as a general location field rather than a strictly standardised state.
 */


// ============================================================
// TRANSFORMATION 5 – sortByKey
// Rank locations by average salary (descending) to produce a
// geographic compensation hierarchy from the T4 aggregation.
// ============================================================

println("\n\n--- TRANSFORMATION 5: sortByKey ---")
println("Goal: Rank all locations by average salary (highest to lowest)")

val rdd_sorted_by_salary = {
  rdd_location_agg
    .filter { case (_, _, avgSal, count) => avgSal > 0 && count >= 50 }
    .map { case (location, applies, avgSal, count) => (avgSal, (location, applies, count)) }
    .sortByKey(ascending = false)
}

println("\nTop 20 locations by average salary (min. 50 postings):")
println(f"  ${"Rank"}%5s | ${"Location"}%-30s | ${"Avg Salary"}%12s | ${"Postings"}%9s")
println("  " + "-" * 66)
{
  rdd_sorted_by_salary
    .take(20)
    .zipWithIndex
    .foreach { case ((avgSal, (location, _, count)), i) =>
      println(f"  ${i + 1}%5d | $location%-30s | $$${avgSal}%11.0f | $count%9d")
    }
}

println("\nBottom 10 locations by average salary (min. 50 postings):")
println(f"  ${"Rank"}%5s | ${"Location"}%-30s | ${"Avg Salary"}%12s | ${"Postings"}%9s")
println("  " + "-" * 66)
{
  rdd_sorted_by_salary
    .collect()
    .takeRight(10)
    .reverse
    .zipWithIndex
    .foreach { case ((avgSal, (location, _, count)), i) =>
      println(f"  ${i + 1}%5d | $location%-30s | $$${avgSal}%11.0f | $count%9d")
    }
}

/*
 * INTERPRETATION:
 * The minimum-50-postings threshold excludes locations with insufficient
 * data for a reliable average. The resulting ranking directly addresses
 * the project's question of how location affects salary and informs the
 * expected predictive weight of geographic information in the Phase 5 ML pipeline.
 */

