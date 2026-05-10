object Utils {

  // Normalize salary to yearly
  def normalizeToYearly(salary: Double, payPeriod: String): Double = {
    payPeriod.toUpperCase match {
      case "HOURLY"  => salary * 40 * 52
      case "MONTHLY" => salary * 12
      case "YEARLY"  => salary
      case _         => salary
    }
  }

  // Clean and standardize text fields
  def cleanText(text: String): String = {
    Option(text).getOrElse("").toLowerCase.trim
  }

  // Extract state from location string
  def extractState(location: String): String = {
    Option(location)
      .getOrElse("")
      .split(",")
      .lastOption
      .getOrElse("")
      .trim
      .toLowerCase
  }

  // Stop words for title keyword analysis
  val stopWords: Set[String] = Set(
    "and", "or", "the", "for", "with",
    "of", "in", "a", "an", "to", "at"
  )

}