# genelde büyük verilerde batch processing ve periodic processing yaparız. Realtime ve Near Realtime az karşımıza çıkar.
#orta çıkan veri işleme güçlüğü verinin hacmi, çeşitliliği ve hızı ile alakalıdır.
# yaygın bir yanılgı bu özelliklerin birlikte olduğunda ancak veriye büyük veri denilebileceğidir.
# büyük veri araçları veriden faydalı bilgi çıkarma süreçleri için çok güçlü bir araçtır.
# Apache Hadoop : Büyük veri teknolojilerini temelini oluşturur. Big table, map reduce gibi google makalelerinden çıkmadır.
# hadoop common
# hdfs
# hadoop yarn
# hadoop MapReduce
# apache spark, apache hadoop'un alternatifi değil mapreduce alternatifidir.
# mapreduce modelinde yer alan disk bazlı çalışma sisteminin yarattığı maliyetlerden dolayı ortaya çıkmıştır. mapreduce paralel yazılım uygulamasıdır.
# genelleştiricidir: SparkSQL, SparkMLlib, Spark Streaming, GraphX aynı uygulama da kullanılabilir.
# dayanıklı dağıtık data setler (RDDs) ,spark lazy evaluation olarak da adlandırılır.
# hadoop veriyi saklamak(dağıtık) için, spark işlemek içindir.
#

# vertica, hp'nin hadoopunun kardeşidir.ikizidir.
# sc.stop() dmeeyi unutmayın

##################################################
# Churn Prediction using PySpark
##################################################

# Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirebilir misiniz?
# Amaç bir bankanın müşterilerinin bankayı terk etme ya da terk etmeme durumunun tahmin edilmesidir.

# 1. Kurulum
# 2. Exploratory Data Analysis
# 3. SQL Sorguları
# 4. Data Preprocessing & Feature Engineering
# 5. Modeling

##################################################
# Kurulum
##################################################

# https://spark.apache.org/downloads.html
# username/spark dizinin altına indir.

# pyarrow: Farklı veri yapıları arasında çalışma kolaylığı sağlayan bir modul.

# pip install pyspark
# pip install findspark
# pip install PyArrow


import warnings
import findspark
import pandas as pd
import seaborn as sns
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.ml.feature import Bucketizer


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

findspark.init("C:\spark")

spark = SparkSession.builder \
    .master("local") \
    .appName("pyspark_giris") \
    .getOrCreate()

sc = spark.sparkContext

# http://mvahit-mbp.lan:4040/jobs/
# sc.stop()


##################################################
# Exploratory Data Analysis
##################################################

############################
# Pandas df ile Spark df farkını anlamak.
############################


spark_df = spark.read.csv("datasets/churn2.csv", header=True, inferSchema=True)
spark_df
type(spark_df)




spark_df.head()


spark_df.dtypes

# spark_df.ndim


# Reading a json file
# df = spark.read.json(json_file_path)
#
# # Reading a text file
# df = spark.read.text(text_file_path)

# # Reading a parquet file
# df = spark.read.load(parquet_file_path) # or
# df = spark.read.parquet(parquet_file_path)

############################
# Exploratory Data Analysis
############################

# Gözlem ve değişken sayısı
print("Shape: ", (spark_df.count(), len(spark_df.columns)))

# Değişken tipleri
spark_df.printSchema()
spark_df.dtypes

# Değişken seçme
spark_df.Age

# Bir değişkeni görmek
spark_df.select(spark_df.columns).show()

# Head
spark_df.show(3, truncate=True)
spark_df.take(5)
spark_df.head()

# Değişken isimlerinin küçültülmesi
spark_df = spark_df.toDF(*[c.lower() for c in spark_df.columns])
spark_df.show(5)

# özet istatistikler
spark_df.describe().show()

# sadece belirli değişkenler için özet istatistikler
spark_df.describe(["age", "exited"]).show()
spark_df.describe(spark_df.columns).show()

# Kategorik değişken sınıf istatistikleri
spark_df.groupby("exited").count().show()

# Eşsiz sınıflar
spark_df.select("exited").distinct().show()
spark_df.select("customerid").distinct().show()

# select(): Değişken seçimi
spark_df.select("age", "names").show(5)

# filter(): Gözlem seçimi / filtreleme
spark_df.filter(spark_df.age > 40).show()
spark_df.filter(spark_df.age > 40).count()

# groupby işlemleri
spark_df.groupby("exited").count().show()
spark_df.groupby("exited").agg({"age": "mean"}).show()
spark_df.groupby("exited").agg({"age": "mean",
                                "creditscore":"mean"}).show()


# Tüm numerik değişkenlerin seçimi ve özet istatistikleri
num_cols = [col[0] for col in spark_df.dtypes if col[1] != 'string']
spark_df.select(num_cols).describe().toPandas().transpose()

# Tüm kategorik değişkenlerin seçimi ve özeti
cat_cols = [col[0] for col in spark_df.dtypes if col[1] == 'string']

for col in cat_cols:
    spark_df.select(col).distinct().show()

# Churn'e göre sayısal değişkenlerin özet istatistikleri
for col in [col.lower() for col in num_cols]:
    spark_df.groupby("exited").agg({col: "mean"}).show()


##################################################
# SQL Sorguları
##################################################

spark_df.createOrReplaceTempView("tbl_df")
spark.sql("show databases").show()
spark.sql("show tables").show()
spark.sql("select age from tbl_df limit 5").show()
spark.sql("select exited, avg(age) from tbl_df group by exited").show()


##################################################
# Data Preprocessing & Feature Engineering
##################################################

############################
# Missing Values
############################

from pyspark.sql.functions import when, count, col
spark_df.select([count(when(col(c).isNull(), c)).alias(c) for c in spark_df.columns]).toPandas().T

# eksik değere sahip satırları silmek
spark_df.dropna().show()

# tüm veri setindeki eksiklikleri belirli bir değerle doldurmak
spark_df.fillna(50).show()

# eksik değerleri değişkenlere göre doldurmak
spark_df.na.fill({'age': 50, 'names': 'unknown'}).show()

############################
# Feature Interaction
############################

spark_df = spark_df.withColumn('age_total_purchase', spark_df.age / spark_df.total_purchase)
spark_df.show(5)


############################
# Bucketization / Bining / Num to Cat
############################

spark_df.select('age').describe().toPandas().transpose()

bucketizer = Bucketizer(splits=[18, 25,35, 45,60 ,92], inputCol="age", outputCol="age_cat")

spark_df = bucketizer.setHandleInvalid("keep").transform(spark_df)

bucketizer = Bucketizer(splits=[350, 580,670, 740,800 ,850], inputCol="creditscore", outputCol="creditscore_cat")

spark_df = bucketizer.setHandleInvalid("keep").transform(spark_df)

spark_df.show(20)

# Bucketizer isim vermeye sıfırdan başlıyor.
# bu sebeple üzerine 1 ekleyelim:
spark_df = spark_df.withColumn('age_cat', spark_df.age_cat + 1) # değişkenler üzerinde işlem yapmak için withColumns ile yaparız.
spark_df = spark_df.withColumn('creditscore_cat', spark_df.creditscore_cat + 1) # değişkenler üzerinde işlem yapmak için withColumns ile yaparız.

spark_df.groupby("age_cat").count().show()
spark_df.groupby("age_cat").agg({'exited': "mean"}).show()

spark_df.groupby("creditscore_cat").count().show()
spark_df.groupby("creditscore_cat").agg({'exited': "mean"}).show()


# ondalık ifadelerden kurtulmak
spark_df = spark_df.withColumn("age_cat", spark_df["age_cat"].cast("integer"))
spark_df.groupby("age_cat").agg({'exited': "mean"}).show()

spark_df = spark_df.withColumn("creditscore_cat", spark_df["creditscore_cat"].cast("integer"))
spark_df.groupby("creditscore_cat").agg({'exited': "mean"}).show()


############################
# when ile Değişken Türetmek (segment)
############################

spark_df = spark_df.withColumn('segment', when(spark_df['years'] < 5, "segment_b").otherwise("segment_a"))

############################
# when ile Değişken Türetmek (age_cat_2)
############################

spark_df.withColumn('age_cat_2',
                    when(spark_df['age'] < 36, "young").
                    when((35 < spark_df['age']) & (spark_df['age'] < 46), "mature").
                    otherwise("senior")).show()

############################
# Label Encoding
############################

spark_df.show(5)
indexer = StringIndexer(inputCol="gender", outputCol="gender_label")
indexer.fit(spark_df).transform(spark_df).show(5)

indexer = StringIndexer(inputCol="geography", outputCol="geography_label")

indexer.fit(spark_df).transform(spark_df).show(5)

temp_sdf = indexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("geography_label", temp_sdf["geography_label"].cast("integer"))

temp_sdf = indexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("gender_label", temp_sdf["gender_label"].cast("integer"))



# eski segmenti uçuralım:
spark_df = spark_df.drop('gender')
spark_df = spark_df.drop('geography')
spark_df = spark_df.drop('surname')

spark_df.show()

############################
# One Hot Encoding
############################
spark_df.show(5)
cat_cols
encoder = OneHotEncoder(inputCols=["age_cat","creditscore_cat","geography_label"], outputCols=["age_cat_ohe","cs_cat_label","geography_ohe"])
spark_df = encoder.fit(spark_df).transform(spark_df)

cat_cols = [col[0] for col in spark_df.dtypes if col[1] == 'string']
num_cols = [col[0] for col in spark_df.dtypes if col[1] != 'string']


############################
# TARGET'ın Tanımlanması
############################

# TARGET'ın tanımlanması
stringIndexer = StringIndexer(inputCol='exited', outputCol='label')
temp_sdf = stringIndexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("label", temp_sdf["label"].cast("integer"))

spark_df.show(5)
spark_df.columns
############################
# Feature'ların Tanımlanması
############################

cols = ['rownumber',
 'customerid',
 'creditscore',
 'age',
 'tenure',
 'balance',
 'numofproducts',
 'hascrcard',
 'isactivemember',
 'estimatedsalary',
 'age_cat',
 'creditscore_cat',
 'gender_label',
 'geography_label',
 'age_cat_ohe',
 'cs_cat_label',
 'geography_ohe']

# Vectorize independent variables.
va = VectorAssembler(inputCols=cols, outputCol="features")
va_df = va.transform(spark_df)
va_df.show()

# Final sdf
final_df = va_df.select("features", "label")
final_df.show(5)

# StandardScaler
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
final_df = scaler.fit(final_df).transform(final_df)

# Split the dataset into test and train sets.
train_df, test_df = final_df.randomSplit([0.7, 0.3], seed=17)
train_df.show(10)
test_df.show(10)

print("Training Dataset Count: " + str(train_df.count()))
print("Test Dataset Count: " + str(test_df.count()))

##################################################
# Modeling
##################################################

############################
# Logistic Regression
############################

log_model = LogisticRegression(featuresCol='features', labelCol='label').fit(train_df)
y_pred = log_model.transform(test_df)
y_pred.show()

y_pred.select("label", "prediction").show()

# accuracy
y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()

#
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName='areaUnderROC')
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

acc = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "accuracy"})
precision = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "precisionByLabel"})
recall = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "recallByLabel"})
f1 = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "f1"})
roc_auc = evaluator.evaluate(y_pred)

print("accuracy: %f, precision: %f, recall: %f, f1: %f, roc_auc: %f" % (acc, precision, recall, f1, roc_auc))

############################
# Gradient Boosted Tree Classifier
############################

gbm = GBTClassifier(maxIter=100, featuresCol="features", labelCol="label")
gbm_model = gbm.fit(train_df)
y_pred = gbm_model.transform(test_df)
y_pred.show(5)

y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()


############################
# Model Tuning
############################

evaluator = BinaryClassificationEvaluator()

gbm_params = (ParamGridBuilder()
              .addGrid(gbm.maxDepth, [2, 4, 6])
              .addGrid(gbm.maxBins, [20, 30])
              .addGrid(gbm.maxIter, [10, 20])
              .build())

cv = CrossValidator(estimator=gbm,
                    estimatorParamMaps=gbm_params,
                    evaluator=evaluator,
                    numFolds=5)

cv_model = cv.fit(train_df)


y_pred = cv_model.transform(test_df)
ac = y_pred.select("label", "prediction")
ac.filter(ac.label == ac.prediction).count() / ac.count()


############################
# New Prediction
############################


# PEKI YENI BIR MUSTERI GELDIGINDE NE YAPACAĞIZ?

names = pd.Series(["Ali Ahmetoğlu", "Taner Gün", "Berkay", "Polat Konak", "Kamil Atasoy"])
age = pd.Series([18, 43, 34, 50, 40])
total_purchase = pd.Series([5000, 10000, 6000, 30000, 100000])
account_manager = pd.Series([1, 0, 0, 1, 1])
years = pd.Series([20, 10, 3, 8, 30])
num_sites = pd.Series([2, 8, 8, 6, 50])
age_total_purchase = age / total_purchase
segment_label = pd.Series([1, 1, 0, 1, 1])
age_cat_ohe = pd.Series([1, 1, 0, 2, 1])

yeni_musteriler = pd.DataFrame({
    'names': names,
    'age': age,
    'total_purchase': total_purchase,
    'account_manager': account_manager,
    'years': years,
    'num_sites': num_sites,
    "age_total_purchase": age_total_purchase,
    "segment_label": segment_label,
    "age_cat_ohe": age_cat_ohe})

yeni_sdf = spark.createDataFrame(yeni_musteriler)
new_customers = va.transform(yeni_sdf)
new_customers.show(3)

# bi modele soralım bakalım ne diyecek
results = cv_model.transform(new_customers)
results.select("names", "prediction").show()



# scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
# new_customers_final = scaler.fit(new_customers).transform(new_customers)
# results = cv_model.transform(new_customers_final)
# results.select("names", "prediction").show()

# sc.stop()
