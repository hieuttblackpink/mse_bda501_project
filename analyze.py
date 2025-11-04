from pyspark.sql import SparkSession
from pyspark.sql.functions import col

import os

# Constant
HDFS_PATH_BASE = "hdfs://localhost:19000/user/bda501/brain_tumor/"
PROCESSED_PATH = HDFS_PATH_BASE + "processed/"

# Root folder path
ROOT_FOLDER_PATH = "C:/Users/Hieu/OneDrive/MSE - FPT/Fall 2025/BDA501/Project/"
RESULT_FILE = "analyze_result.txt"

RESULT_FILE_PATH = os.path.join(ROOT_FOLDER_PATH, RESULT_FILE)

# Initialize Spark Session
def init_spark():
    spark = (
        SparkSession.builder.appName("BrainTumorAnalysis")
        .master("yarn")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    return spark

# Analyze processed data
def main():
    spark = init_spark()
    print("Spark Session initialized.")
    
    # Read PARQUEST data
    try:
        final_df = spark.read.parquet(PROCESSED_PATH)
    except Exception as e:
        print(f"Error reading parquet data: {e}")
        spark.stop()
        return
    
    # Analyze data
    final_df.printSchema()

    label_counts = final_df.groupBy("label").count()
    label_counts.show()

    total_count = final_df.count()
    print(f"Total number of records: {total_count}")

    # Save analysis results to text file
    try:
        with open(RESULT_FILE_PATH, "w") as f:
            f.write("Label Counts:\n")
            for row in label_counts.collect():
                f.write(f"{row['label']}: {row['count']}\n")
            f.write(f"\nTotal number of records: {total_count}\n")
        print("Analysis results saved to analysis_results.txt")
    except Exception as e:
        print(f"Error writing analysis results: {e}")
    
    spark.stop()

if __name__ == "__main__":
    main()