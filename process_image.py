import io
import pandas as pd
import numpy as np

from PIL import Image

# Spark
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, StructField, StringType, BinaryType, ArrayType, FloatType

# Deep Learning Libraries
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights

# Constant
HDFS_PATH = "hdfs://localhost:19000/user/bda501/brain_tumor/"
TRAIN_PATH = HDFS_PATH + "Training"
TEST_PATH = HDFS_PATH + "Testing"
OUTPUT_PATH = HDFS_PATH + "processed/"

IMG_WIDTH = 224
IMG_HEIGHT = 224

# Global variables
model_singleton = None

PREPROCESS_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_model():
    global model_singleton
    if model_singleton is None:
        # Load pre-trained MobileNetV2 model
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        model = models.mobilenet_v2(weights=weights).to(device)

        # Remove the final classification layer
        model.classifier = torch.nn.Identity()

        # Put model in evaluation mode
        model.eval()
        model_singleton = model
    return model_singleton

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Spark Session
def init_spark():
    spark = (
        SparkSession.builder.appName("BrainTumorFeatureExtraction")
        .master("yarn")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    return spark

# Extract label function
def extract_label_from_path(path):
    try:
        return path.split("/")[-2]
    except Exception as e:
        return None
    
label_extractor_udf = udf(extract_label_from_path, StringType())

# Image extraction function
@pandas_udf(ArrayType(FloatType()))
def extract_features_udf(content_series: pd.Series) -> pd.Series:
    # Load pre-trained MobileNetV2 model
    # Using the latest weights
    model = get_model()

    # Define preprocessing images pineline
    preprocess_transform = PREPROCESS_TRANSFORM

    # Function to process a single image
    def preprocess_image(content_bytes):
        img = Image.open(io.BytesIO(content_bytes)).convert("RGB")
        return preprocess_transform(img)
    
    # Process all images in the batch
    img_batch_tensor = torch.stack([preprocess_image(content) for content in content_series]).to(device)

    # Feature extraction
    with torch.no_grad():
        features_batch = model(img_batch_tensor)

    # Convert tensor to numpy arrays and then to pd.Series
    features_np = features_batch.cpu().numpy()
    return pd.Series(list(features_np))

# Main processing function
def main():
    spark = init_spark()
    sc = spark.sparkContext
    print("Spark Session initialized and connected to YARN.")

    # Read data
    image_rdd = sc.binaryFiles(TRAIN_PATH + "/*/*.jpg")
    image_df = image_rdd.toDF(schema=StructType([
        StructField("path", StringType(), True),
        StructField("content", BinaryType(), True)
    ]))
    image_df = image_df.repartition(8)
    print(f"Image found: {image_df.count()}")

    # Process data
    df_with_label = image_df.withColumn("label", label_extractor_udf(col("path")))

    print("Starting feature extraction...")
    df_with_features = df_with_label.withColumn(
        "features", extract_features_udf(col("content"))
    )

    # Create dataframe
    final_df = df_with_features.select("path", "label", "features")
    print("Feature extraction completed.")
    final_df.printSchema()
    final_df.show(5)

    # Save data
    print(f"Saving processed data into {OUTPUT_PATH} ...")
    final_df.write.mode("overwrite").parquet(OUTPUT_PATH)
    print("Data saved successfully.")

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    print(f"Using {device}")
    main()