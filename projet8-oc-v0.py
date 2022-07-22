# Databricks notebook source
pip install tensorflow

# COMMAND ----------

# "Classic" Python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Streams
import io

# Images
from PIL import Image

# Pyspark
import pyspark
from pyspark.ml.image import ImageSchema
from pyspark.sql.functions import col, pandas_udf, PandasUDFType, udf, element_at, split
from pyspark.ml.feature import PCA, StandardScaler
from pyspark.ml.linalg import Vectors, VectorUDT, DenseVector

# Transfer learning
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# COMMAND ----------

# Define the path to load all the images contained in directory light_training_4
path_train_set_light = 'dbfs:/FileStore/tables/resources/light_training_4/*/*'

# COMMAND ----------

# Load the images with spark.read.format('image') from path_train_set_light
img_train_light = spark.read.format('image').load(path_train_set_light, inferschema=True)

# COMMAND ----------

# Print the schema of img_train_light dataframe in a tree format
img_train_light.printSchema()

# COMMAND ----------

img_train_light.show()

# COMMAND ----------

# MAGIC %md
# MAGIC https://docs.databricks.com/data/data-sources/image.html  
# MAGIC 
# MAGIC Databricks recommends that you use the binary file data source to load image data into the Spark DataFrame as raw bytes
# MAGIC Indeed, the image data source decodes the image files during the creation of the Spark DataFrame, increases the data size, and introduces limitations in the following scenarios:
# MAGIC 
# MAGIC 1) Persisting the DataFrame: If you want to persist the DataFrame into a Delta table for easier access, you should persist the raw bytes instead of the decoded data to save disk space.
# MAGIC 
# MAGIC 2) Shuffling the partitions: Shuffling the decoded image data takes more disk space and network bandwidth, which results in slower shuffling. You should delay decoding the image as much as possible.
# MAGIC 
# MAGIC 3) Choosing other decoding method: The image data source uses the Image IO library of javax to decode the image, which prevents you from choosing other image decoding libraries for better performance or implementing customized decoding logic.
# MAGIC 
# MAGIC -> Those limitations can be avoided by using the **binary file** data source to load image data and decoding only as needed.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load binaryfile images
# MAGIC 
# MAGIC https://docs.databricks.com/data/data-sources/binary-file.html
# MAGIC 
# MAGIC Databricks Runtime supports the binary file data source, which reads binary files and converts each file into a single record that contains the raw content and metadata of the file. The binary file data source produces a DataFrame with the following columns and possibly partition columns:
# MAGIC 
# MAGIC - path (StringType): The path of the file.
# MAGIC 
# MAGIC - modificationTime (TimestampType): The modification time of the file. In some Hadoop FileSystem implementations, this parameter might be unavailable and the value would be set to a default value.
# MAGIC 
# MAGIC - length (LongType): The length of the file in bytes.
# MAGIC 
# MAGIC - content (BinaryType): The contents of the file.
# MAGIC 
# MAGIC To read binary files, specify the data source format as binaryFile.

# COMMAND ----------

# Create dataframe by loading train_set_light with binaryFile format
binary_train = spark.read.format("binaryFile").load(path_train_set_light)

# Print the schema of binary_train dataframe in a tree format
binary_train.printSchema()

# COMMAND ----------

# Display image data loaded using the binary data source
# (image preview is automatically enabled for file names with an image extension)
display(binary_train)

# COMMAND ----------

# Print all the rows of binary_train
binary_train.show(n=30, 
                  truncate=90) # truncate strings longer than 90 characters

# COMMAND ----------

# Print "content" column only
binary_train.select("content").show(truncate=200)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Featurization for transfer learning
# MAGIC 
# MAGIC https://docs.databricks.com/applications/machine-learning/preprocess-data/transfer-learning-tensorflow.html
# MAGIC 
# MAGIC 1) Start with a pre-trained deep learning model, in this case an image classification model from tensorflow.keras.applications.
# MAGIC 
# MAGIC 2) Truncate the last layer(s) of the model. The modified model produces a tensor of features as output, rather than a prediction.
# MAGIC 
# MAGIC 3) Compute features using a Scalar Iterator pandas UDF

# COMMAND ----------

# MAGIC %md
# MAGIC ### VGG16_max

# COMMAND ----------

# MAGIC %md
# MAGIC Prepare model VGG16 with a global **max** pooling layer as top layer : https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/VGG16

# COMMAND ----------

help(tf.keras.applications.vgg16.VGG16)

# COMMAND ----------

# Prepare VGG16 model without the classification layer and with global max pooling
model_VGG16_max = VGG16(
    include_top=False,  # Do not include the last layer (classification layer)
    input_shape=(100, 100, 3),  # The shape of the images is (100, 100, 3)
    pooling='max'  #  Global max pooling is applied to the output of the last convolutional block
)

# Verify that the top layer is removed
model_VGG16_max.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC Define image loading and featurization logic in a Pandas UDF

# COMMAND ----------

# Function to preprocess images: images are converted from RGB to BGR
# then each color channel is zero-centered with respect to the ImageNet dataset
def preprocess(content):
    """
    Preprocesses raw image bytes for prediction.
    """
    img = Image.open(io.BytesIO(content))  # Open image kept as bytes in memory buffer
    arr = img_to_array(img)  # Transform img into numpy.array
    # images are converted from RGB to BGR then each color channel is zero-centered with respect to the ImageNet dataset
    return preprocess_input(arr)   


# COMMAND ----------

# Function to featurize a pd.Series of images
def featurize_series(model, content_series):
    """
    Featurize a pd.Series of raw images using the input model.
    :return: a pd.Series of image features
    """
    # Join a sequence of arrays obtained by substituting each value of content_series
    # with function preprocess
    input = np.stack(content_series.map(preprocess))
    # Predict with the chosen CNN model
    preds = model.predict(input)
    # For some layers, output features will be multi-dimensional tensors.
    # We flatten the feature tensors to vectors for easier storage in Spark DataFrames.
    output = [p.flatten() for p in preds]
    return pd.Series(output)

# COMMAND ----------

@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_udf(content_series_iter):
    '''
    This method is a Scalar Iterator pandas UDF wrapping our featurization function.
    The decorator specifies that this returns a Spark DataFrame column of type ArrayType(FloatType).
  
    :param content_series_iter: This argument is an iterator over batches of data, where each batch
                              is a pandas Series of image data.
    '''
    # With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
    # for multiple data batches.  This amortizes the overhead of loading big models.
    model = model_VGG16_max
    # Iterate function featurize_series with the chosen CNN model for each image
    for content_series in content_series_iter:
        yield featurize_series(model, content_series)

# COMMAND ----------

# MAGIC %md
# MAGIC Apply featurization to the DataFrame of images

# COMMAND ----------

# Pandas UDFs on large records (e.g., very large images) can run into Out Of Memory (OOM) errors.
# If you hit such errors in the cell below, try reducing the Arrow batch size via `maxRecordsPerBatch`.
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")

# COMMAND ----------

# featurization on binary_train.
features_max = binary_train.repartition(3).select(col("path"), featurize_udf("content").alias("features"))

# COMMAND ----------

# Print the new dataframe with 512 features per image
features_max.show(truncate=100)

# COMMAND ----------

# Print the column 'features' of the new dataframe
features_max.select('features').show(n=27, truncate=200)

# COMMAND ----------

# MAGIC %md
# MAGIC ### VGG16_avg
# MAGIC VGG16 with a global **average** pooling layer as top layer

# COMMAND ----------

# Prepare VGG16 model without the classification layer and with global average pooling
model_VGG16_avg = VGG16(
    include_top=False,  # Do not include the last layer (classification layer)
    input_shape=(100, 100, 3),  # The shape of the images is (100, 100, 3)
    pooling='avg'  #  Global average pooling is applied to the output of the last convolutional block
)

# Verify that the top layer is removed
model_VGG16_avg.summary()

# COMMAND ----------

@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_udf_avg(content_series_iter):
    '''
    This method is a Scalar Iterator pandas UDF wrapping our featurization function.
    The decorator specifies that this returns a Spark DataFrame column of type ArrayType(FloatType).
  
    :param content_series_iter: This argument is an iterator over batches of data, where each batch
                              is a pandas Series of image data.
    '''
    # With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
    # for multiple data batches.  This amortizes the overhead of loading big models.
    model = model_VGG16_avg
    for content_series in content_series_iter:
        yield featurize_series(model, content_series)

# COMMAND ----------

# featurization on binary_train.
features_avg = binary_train.repartition(3).select(col("path"), featurize_udf_avg("content").alias("features"))

# COMMAND ----------

# Print the new dataframe with 512 features per image
features_avg.show(truncate=100)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Choice between global max pooling and global average pooling
# MAGIC global max pooling is chosen as the last layer. Indeed it seems that global average pooling smoothen too much the features compared to global max pooling

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dimensionality Reduction with PCA on features_max (features obtained with VGG16_max)

# COMMAND ----------

# MAGIC %md
# MAGIC 1) without standardization of features

# COMMAND ----------

# Transform 'features' into dense vectors and add its in features_max (new column 'vectors')
transform_into_dense_vector = udf(lambda v: Vectors.dense(v), VectorUDT())
features_max = features_max.withColumn('vectors', transform_into_dense_vector('features'))

# COMMAND ----------

# Print the columns 'features' and 'vectors' for comparison
features_max.select('features', 'vectors').show(truncate=50)

# COMMAND ----------

# MAGIC %md
# MAGIC Searching the number of principal components that can explain 99% of the total of explained variance

# COMMAND ----------

# Define pca with the maximum number of components
pca = PCA(k=512,  # number of principal components = number of features = output dimension of the last layer of model VGG16
          inputCol='vectors',  # input column name
          outputCol='features_pca')  # output column name

# Fit on features_max
pca = pca.fit(features_max)

# Calculate explained variance for each feature
explainedvariance = pca.explainedVariance
explainedvariance

# COMMAND ----------

# Plot explainedvariance.cumsum
sns.set_theme()

plt.figure(figsize=(20,6))
g = plt.plot(np.arange(len(explainedvariance)) + 1,
             explainedvariance.cumsum(),
             marker='o',
             markersize=5,
             label='explainedvariance.cumsum')

plt.xticks(np.arange(0, 521, step = 20), size = 13)
plt.yticks(np.arange(0, 1.01, step = 0.1), size = 13)
plt.xlim(-10,520)
plt.xlabel('Number of principal components', fontsize=15)
plt.ylabel('Cumulated sum of explained variance', fontsize=15)
plt.legend(fontsize=15)

plt.show()

# COMMAND ----------

# Plot explainedvariance.cumsum #2
sns.set_theme()

plt.figure(figsize=(20,8))
g = plt.plot(np.arange(len(explainedvariance)) + 1,
             explainedvariance.cumsum(),
             marker='o',
             markersize=7,
             label='explainedvariance.cumsum')
h = plt.axhline(y=0.99, color='r', label='explainedvariance.cumsum = 99%')

plt.xticks(np.arange(0, 521, step=1), size=13)
plt.yticks(np.arange(0, 1.01, step=0.05), size=13)
plt.xlim(-1,31)
plt.ylim(0.3,1.02)
plt.xlabel('Number of principal components', fontsize=15)
plt.ylabel('Cumulated sum of explained variance', fontsize=15)
plt.legend(fontsize=16)

plt.show()

# COMMAND ----------

# Calculate the minimum number of components to reach 99% of variance
for component in range(512):
    var = explainedvariance.cumsum()[component]
    if var >= 0.99:
        n_components = component + 1
        print(f'{n_components} components explain 99% of variance')
        break

# COMMAND ----------

# MAGIC %md
# MAGIC Applying PCA with n_components=22

# COMMAND ----------

# Define pca
pca = PCA(k=n_components, inputCol='vectors', outputCol='features_pca')

# Fit on features_max_dense
pca = pca.fit(features_max)

# Transform features
features_max = pca.transform(features_max)

# COMMAND ----------

features_max.show(truncate=50)

# COMMAND ----------

# MAGIC %md
# MAGIC 2  with standardization of features

# COMMAND ----------

# Standardize features with StandardScaler
scaler = StandardScaler(inputCol="vectors", outputCol="vectors_std")
scaler = scaler.fit(features_max)
features_max_std = scaler.transform(features_max)

# COMMAND ----------

# Display the new dataframe features_max_dense_std
features_max_std.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Searching the number of principal components that can explain 99% of the total of explained variance

# COMMAND ----------

# Define pca
pca = PCA(k=512, inputCol='vectors_std', outputCol='features_std_pca')

# Fit on features_max_dense
pca = pca.fit(features_max_std)

# Calculate explained variance for each feature
explainedvariance_std = pca.explainedVariance
explainedvariance_std

# COMMAND ----------

# Plot explainedvariance_std.cumsum
sns.set_theme()

plt.figure(figsize=(20,6))
g = plt.plot(np.arange(len(explainedvariance_std)) + 1,
             explainedvariance_std.cumsum(),
             color='g',
             marker='o',
             markersize=5,
             label='explainedvariance_std.cumsum')

plt.xticks(np.arange(0, 521, step = 20), size = 13)
plt.yticks(np.arange(0, 1.01, step = 0.1), size = 13)
plt.xlim(-10,520)
plt.xlabel('Number of principal components', fontsize=15)
plt.ylabel('Cumulated sum of explained variance', fontsize=15)
plt.legend(fontsize=15)

plt.show()

# COMMAND ----------

# Plot explainedvariance_std.cumsum #2
sns.set_theme()

plt.figure(figsize=(20,8))
g = plt.plot(np.arange(len(explainedvariance_std)) + 1,
             explainedvariance_std.cumsum(),
             color='g',
             marker='o',
             markersize=7,
             label='explainedvariance_std.cumsum')
h = plt.axhline(y=0.99, color='r', label='explainedvariance_std.cumsum = 99%')

plt.xticks(np.arange(0, 521, step=1), size=13)
plt.yticks(np.arange(0, 1.01, step=0.05), size=13)
plt.xlim(-1,31)
plt.ylim(0.1,1.02)
plt.xlabel('Number of principal components', fontsize=15)
plt.ylabel('Cumulated sum of explained variance', fontsize=15)
plt.legend(fontsize=16)

plt.show()

# COMMAND ----------

# Calculate the minimum number of components to reach 99.9% of variance
for component in range(512):
    var = explainedvariance_std.cumsum()[component]
    if var >= 0.99:
        n_components = component + 1
        print(f'{n_components} components explain 99% of variance')
        break

# COMMAND ----------

# Compare explainedvariance and explainedvariance_std
sns.set_theme()

plt.figure(figsize=(20,8))
g = plt.plot(np.arange(len(explainedvariance)) + 1,
             explainedvariance.cumsum(),
             color='blue',
             marker='o',
             markersize=7,
             label='explainedvariance.cumsum')
g_std = plt.plot(np.arange(len(explainedvariance_std)) + 1,
                 explainedvariance_std.cumsum(),
                 color='green',
                 marker='o',
                 markersize=7,
                 label='explainedvariance_std.cumsum')
h = plt.axhline(y=0.99, color='r', label='explainedvariance_std.cumsum = 99%')

plt.xticks(np.arange(0, 521, step=1), size=13)
plt.yticks(np.arange(0, 1.01, step=0.05), size=13)
plt.xlim(-1,31)
plt.ylim(0.1,1.02)
plt.xlabel('Number of principal components', fontsize=15)
plt.ylabel('Cumulated sum of explained variance', fontsize=15)
plt.legend(fontsize=16)

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Applying PCA with n_components=24

# COMMAND ----------

# Define pca
pca = PCA(k=n_components, inputCol='vectors_std', outputCol='features_std_pca')

# Fit on features_max_std
pca = pca.fit(features_max_std)

# Transform features
features_max_std = pca.transform(features_max_std)

# COMMAND ----------

features_max_std.show(truncate=30)

# COMMAND ----------

# Select useful columns before the save
features_max_std_pca = features_max_std.select('path', 'features_std_pca')

# COMMAND ----------

# Create the column "class" containing the fruit class of each image
# by extracting the fruit class from the image path
features_max_std_pca = features_max_std_pca.withColumn(
    colName="class", 
    col=element_at(
        col=split(
            features_max_std_pca["path"], 
            "/"),  # split the path with the "/" character
        extraction=-2))  # extract the penultimate string in the binary_train["path"]

# COMMAND ----------

features_max_std_pca.show(truncate=80)

# COMMAND ----------

# Save the last dataframe features_max_std_pca
features_max_std_pca.write.mode("overwrite").parquet("dbfs:/FileStore/tables/features_max_std_pca")

# COMMAND ----------


