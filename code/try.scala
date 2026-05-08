// 1. Calculate approximate percentile thresholds
val p33 = salaryValues(math.min((totalSalaryRows * 0.33).toInt, totalSalaryRows - 1))
val p67 = salaryValues(math.min((totalSalaryRows * 0.67).toInt, totalSalaryRows - 1))

// 2. Map continuous salary data into discrete analytical tiers
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
