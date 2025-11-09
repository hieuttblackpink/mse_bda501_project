# BDA501: Brain Tumor Classification Pipeline with Spark & HDFS

This is the final project for the Big Data (BDA501) course, implementing an end-to-end Big Data processing pipeline using Apache Spark on a pseudo-distributed Hadoop (HDFS + YARN) cluster running natively on Windows 11.

The pipeline processes 1.15GB of brain MRI images, extracts features using a Deep Learning model (PyTorch/MobileNetV2), and trains a Machine Learning model (Spark MLlib) to classify four types of brain tumors.

## Pipeline Architecture
The data flow for this project is as follows:

1.  **Ingestion:** Raw image data (`.jpg`) from Kaggle is uploaded to **HDFS**.
2.  **ETL & Feature Extraction (Stage 2):**
    * `process_images.py` reads binary files from HDFS.
    * It uses a **Pandas UDF** (PySpark) for batch image processing.
    * A **MobileNetV2 (PyTorch)** model is invoked to extract a 1280-dimension feature vector for each image.
    * The resulting DataFrame (path, label, features) is saved back to HDFS in **Parquet** format.
3.  **Analysis (Stage 3):**
    * `analyze.py` reads the processed Parquet file.
    * It performs descriptive statistics (`groupBy().count()`) to check for data imbalance.
4.  **Training (Stage 4):**
    * `train_model.py` reads the Parquet file.
    * Prepares data for ML (`StringIndexer`, `VectorAssembler`).
    * Trains a **Random Forest** or **MLPClassifier** model (Spark MLlib).
    * Evaluates the model (Accuracy, F1-Score).
5.  **Artifacts (Local FS):**
    * The trained model is saved to the **Local File System** (e.g., `D:/BDA_Project/`).
    * A **Confusion Matrix** image (`.png`) is saved to the Local File System.

## Technology Stack
* **Cluster:** Apache Hadoop 3.3.x (HDFS + YARN)
* **Processing:** Apache Spark 3.5.x (PySpark)
* **Environment:** Windows 11 (Native)
* **Language:** Python 3.10
* **AI/ML:** PyTorch (for feature extraction), Spark MLlib (for training)
* **Support Libraries:** PyArrow, Pandas, Matplotlib, Seaborn

---

## Environment Setup Guide (Windows 11)
This is the most complex part, requiring precise configuration to run Hadoop/Spark natively on Windows.

### 1. Software Installation
1.  **Java JDK 11:** Install (e.g., Microsoft Build of OpenJDK 11) and set the `JAVA_HOME` environment variable.
2.  **Python 3.10:** Install (from python.org) and set the `PYSPARK_PYTHON` environment variable to point to the `python.exe` file.
3.  **Apache Hadoop:** Download the binary, unpack it to `C:\hadoop`.
4.  **Apache Spark:** Download the pre-built binary for Hadoop 3.3+, unpack it to `C:\spark`.
5.  **`winutils.exe`:** Download `winutils.exe` and `hadoop.dll` for Hadoop 3.3.0 (e.g., from [this repo](https://github.com/cdarlint/winutils)) and place them in `C:\hadoop\bin`.

### 2. Environment Variables
Add the following to **System variables**:
* `HADOOP_HOME`: `C:\hadoop`
* `SPARK_HOME`: `C:\spark`
* `HADOOP_CONF_DIR`: `C:\hadoop\etc\hadoop`
* `JAVA_HOME`: `C:\PROGRA~1\Microsoft\jdk-11.x.x` (Using the 8.3 short name is **mandatory**. Do **NOT** use "Program Files").
* `PYSPARK_PYTHON`: `C:\path\to\your\python3.10\python.exe`

Edit the system `Path` variable and add:
* `%JAVA_HOME%\bin`
* `%HADOOP_HOME%\bin`
* `%SPARK_HOME%\bin`

### 3. Hadoop Configuration
Edit the following files in `C:\hadoop\etc\hadoop`:

1.  **`core-site.xml`:** Configure the HDFS NameNode (using port 19000 as an example).
    ```xml
    <configuration>
        <property>
            <name>fs.defaultFS</name>
            <value>hdfs://localhost:19000</value>
        </property>
    </configuration>
    ```
2.  **`hdfs-site.xml`:** Configure directories for NameNode and DataNode.
    ```xml
    <configuration>
        <property>
            <name>dfs.namenode.name.dir</name>
            <value>file:///C:/hadoop/data/namenode</value>
        </property>
        <property>
            <name>dfs.datanode.data.dir</name>
            <value>file:///C:/hadoop/data/datanode</value>
        </property>
    </configuration>
    ```
3.  **`yarn-site.xml`:** Disable the disk health check (fixes a common permission error on Windows).
    ```xml
    <configuration>
        <property>
            <name>yarn.nodemanager.disk-health-checker.enable</name>
            <value>false</value>
        </property>
    </configuration>
    ```
4.  **`hadoop-env.cmd`:** Add the following flags to the end of the file for Java 11+ compatibility.
    ```batch
    set HADOOP_OPTS=%HADOOP_OPTS% --add-opens=java.base/java.lang=ALL-UNNAMED
    set HADOOP_OPTS=%HADOOP_OPTS% --add-opens=java.base/java.nio=ALL-UNNAMED
    set HADOOP_OPTS=%HADOOP_OPTS% --add-opens=java.base/java.net=ALL-UNNAMED
    set HADOOP_OPTS=%HADOOP_OPTS% --add-opens=java.base/java.util=ALL-UNNAMED
    set HADOOP_OPTS=%HADOOP_OPTS% --add-opens=java.base/java.util.concurrent=ALL-UNNAMED
    set HADOOP_OPTS=%HADOOP_OPTS% --add-opens=java.base/sun.nio.ch=ALL-UNNAMED
    set HADOOP_OPTS=%HADOOP_OPTS% --add-opens=java.base/sun.util.calendar=ALL-UNNAMED
    set HADOOP_OPTS=%HADOOP_OPTS% --add-opens=java.base/java.io=ALL-UNNAMED
    ```

### 4. Install Python Libraries
```bash
# Use the correct Python 3.10 executable
C:\path\to\your\python3.10\python.exe -m pip install pyspark pandas pyarrow
C:\path\to\your\python3.10\python.exe -m pip install torch torchvision
C:\path\to\your\python3.10\python.exe -m pip install matplotlib seaborn
```

### 5. Format HDFS
Open Command Prompt (as Administrator) and run:
```batch
hdfs namenode -format
```

---

## How to Run the Pipeline

Important: You must run **`start-dfs.cmd`** and **`start-yarn.cmd`** from a Command Prompt (as `Administrator`).

### 1. Start the Cluster
Open two separate CMD (`Admin`) windows:

```batch
REM --- Window 1: Start HDFS ---
cd C:\hadoop\sbin
start-dfs.cmd
```

```batch
REM --- Window 2: Start YARN ---
cd C:\hadoop\sbin
start-yarn.cmd
```

(You will see 4 new service windows pop up. Keep them running in the background.)

### 2. Stage 1: Upload Data to HDFS
Assuming your unzipped data is at `D:\datasets\brain_tumor\`

```batch
REM Open a new, non-admin CMD window
hdfs dfs -mkdir -p /user/bda501/brain_tumor

hdfs dfs -put D:\datasets\brain_tumor\Training /user/bda501/brain_tumor/
hdfs dfs -put D:\datasets\brain_tumor\Testing /user/bda501/brain_tumor/
```

### 3. Stage 2: Feature Extraction (ETL)
Run the `process_images.py` script.

```bash
spark-submit --master yarn --deploy-mode client process_images.py
```
(This will take 10-20 minutes as it processes > 700MB of images)

### 4. Stage 3: Analyze Data
Run the `analyze.py` script to see the data statistics.

```bash
spark-submit --master yarn --deploy-mode client analyze.py
```

### 5. Stage 4: Train Model
Run the `spark_ml.py` script to train and evaluate the model.

```bash
spark-submit --master yarn --deploy-mode client train_model.py
```

### 6. Stop the Cluster
When finished, you can shut down all services:

```bash
REM Open a CMD (Admin) window
cd C:\hadoop\sbin
stop-all.cmd
```

---

## Troubleshooting & Key Fixes

This section documents the classic errors encountered during the Windows 11 native setup and their solutions:

* Error: `Incompatible clusterIDs` (HDFS).

    => Solution: Stop HDFS (stop-dfs.cmd), delete the C:\hadoop\data directory, and re-run hdfs namenode -format.

* Error: `Permissions incorrectly set` (YARN).

    => Solution: Add 
    ```xml
    <name>
        yarn.nodemanager.disk-health-checker.enable
    </name>
    <value>false</value>
    ``` 
    to `yarn-site.xml`.

* Error: `InaccessibleObjectException` (Java 11).

    => Solution: Add the 8 `--add-opens` flags to `hadoop-env.cmd`.

* Error: YARN AM dies with `'"C:\Program Files\..."'` is not recognized.

    => Solution: Change the `JAVA_HOME` variable to use the 8.3 short name (e.g., `C:\PROGRA~1\...`).

* Error: `TimeoutError: timed out` in Pandas UDF.

    => Solution: Optimize the UDF to load the PyTorch model only once (using a global singleton) instead of on every batch.

---

## Results
* Accuracy: `~91%` (with MLP Classifier).