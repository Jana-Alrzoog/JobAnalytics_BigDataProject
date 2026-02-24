<h1 align="center">Analyzing LinkedIn Job Postings using Apache Spark</h1>

<p align="center">
<img width="900" alt="job" src="https://github.com/user-attachments/assets/61960486-db2a-4461-946e-86f66aa9f1ec" />
</p>

---

## Project Overview
This project analyzes real-world LinkedIn job posting data using Apache Spark in a distributed big data environment.

The main objective is to explore labor market patterns and build a predictive model to estimate salary ranges based on job attributes.

---

## Dataset
- Source: Kaggle – LinkedIn Job Postings (2023–2024)
- Size: 124,000+ records (~500MB)
- Type: Structured multi-attribute job market dataset

Main attributes include:
- Job Title
- Location
- Employment Type
- Salary Range
- Pay Period

---

## Project Goals
- Analyze factors affecting salary levels
- Explore job market trends
- Perform large-scale preprocessing and analytics using Spark
- Build a machine learning model for salary prediction

---

## Technologies Used
- Apache Spark (RDD, SQL, MLlib)
- Scala
- Big Data Processing
- Distributed Computing

---

## Project Structure
```

JobAnalytics_BigDataProject/
├── README.md
├── FinalReport.pdf
├── Presentation_slides.pdf
├── code/
│   ├── 01_DataPreprocessing.scala
│   ├── 02_RDDOperations.scala
│   ├── 03_SQLOperations.scala
│   ├── 04_MachineLearning.scala
│   └── utility_functions.scala
├── data/
│   ├── raw_dataset.csv (download separately)
│   └── preprocessed_dataset.parquet
└── results/
├── rdd_output.txt
├── sql_results.csv
└── ml_metrics.txt

---

## Contributors
- Jana Alruzuq
- Ghadeer Alnuwaysir
- Rana Albridi
- Ghena Almogayad

---

**Course:** IT462 – Big Data Systems  
**University:** King Saud University
