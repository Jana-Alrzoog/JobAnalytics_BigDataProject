// 1. Filter for reliability, map to (key, value) pairs, and sort by salary descending
val rdd_sorted_by_salary = rdd_location_agg
  .filter { case (_, _, avgSalary, postingCount) => avgSalary > 0 && postingCount >= 50 }
  .map { case (location, applies, avgSalary, postingCount) =>
    (avgSalary, (location, applies, postingCount))
  }
  .sortByKey(ascending = false)

// 2. Action to retrieve the Top 20 highest-paying locations
val top20Locations = rdd_sorted_by_salary.take(20)

// 3. Action to retrieve the Bottom 10 lowest-paying locations
// Requires collect() to bring data to the driver for local extraction
val bottom10Locations = rdd_sorted_by_salary.collect().takeRight(10).reverse
