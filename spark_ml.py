from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import RandomForestClassifier, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Constant
HDFS_PATH_BASE = "hdfs://localhost:19000/user/bda501/brain_tumor/"
PROCESSED_PATH = HDFS_PATH_BASE + "processed/"

# Initialize Spark Session
def init_spark():
    spark = (
        SparkSession.builder.appName("BrainTumorClassification")
        .master("yarn")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "1")
        .getOrCreate()
    )
    return spark

# Train and evaluate Decision Tree model
def main():
    spark = init_spark()
    print("Spark Session initialized.")

    # Read processed data
    processed_df = spark.read.parquet(PROCESSED_PATH)

    # Prepare data for ML
    list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
    df_with_vectors = processed_df.withColumn(
        "ml_features", list_to_vector_udf(col("features"))
    ).select("path", "label", "ml_features")
    df_with_vectors.printSchema()

    # Split data into training and test sets
    (trainingData, testData) = df_with_vectors.randomSplit([0.8, 0.2], seed=42)

    print(f"Training Data Count: {trainingData.count()}")
    print(f"Test Data Count: {testData.count()}")
    
    # Pipeline stages
    label_indexer = StringIndexer(
        inputCol="label", 
        outputCol="indexedLabel"
    ).setHandleInvalid("keep")

    rf = RandomForestClassifier(
        labelCol="indexedLabel", 
        featuresCol="ml_features",
        numTrees=200,
        maxDepth=20
    )

    mlp = MultilayerPerceptronClassifier(
        labelCol="indexedLabel",
        featuresCol="ml_features",
        maxIter=200,
        layers=[1280, 256, 64, 4],
        blockSize=128,
        seed=42
    )

    pipeline = Pipeline(stages=[label_indexer, mlp])

    # Train model
    model = pipeline.fit(trainingData)

    # Evaluate model
    predictions = model.transform(testData)

    predictions.select("path", "label", "prediction", "indexedLabel").show(20)

    # Show report
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", 
        predictionCol="prediction", 
        metricName="accuracy"
    )
    accuracy = evaluator_acc.evaluate(predictions)

    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="indexedLabel",
        predictionCol="prediction",
        metricName="f1"
    )
    f1_score = evaluator_f1.evaluate(predictions)

    print(f"Test Accuracy = {accuracy}")
    print(f"Test F1 Score = {f1_score}")

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()