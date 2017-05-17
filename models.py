# Extracting Inpatient claims data from datastore to SQL table 
df = sqlContext\
  .read.format('com.databricks.spark.csv')\
  .options(header=True, inferschema=True)\
  .load('/FileStore/tables/h2ciahw41492560976213/Inpatient.csv')
  
df_without_prm = df.filter(df['NCH_PRMRY_PYR_CLM_PD_AMT']==0)
df_without_prm = (df_without_prm.na.fill({'CLM_FROM_DT':25000})).filter(df_without_prm['CLM_FROM_DT']!=25000)
df_without_prm.cache()
sqlContext.registerDataFrameAsTable(df_without_prm, "df_without_prm")

data = sqlContext.sql("select CLM_ID, max(DESYNPUF_ID) as DESYNPUF_ID,MAX(SUBSTRING(ICD9_DGNS_CD_1,0,3)) as ICD9_DGNS_CD_1
,MAX(SUBSTRING(ICD9_DGNS_CD_2,0,3)) as ICD9_DGNS_CD_2,MAX(SUBSTRING(ICD9_PRCDR_CD_1,0,3)) as ICD9_PRCDR_CD_1,
MAX(SUBSTRING(ICD9_PRCDR_CD_2,0,3)) as ICD9_PRCDR_CD_2,int(first(CLM_FROM_DT)/10000) as year,
(first(NCH_BENE_IP_DDCTBL_AMT)+first(NCH_BENE_PTA_COINSRNC_LBLTY_AM)+first(NCH_BENE_BLOOD_DDCTBL_LBLTY_AM)+ ((first(CLM_PASS_THRU_PER_DIEM_AMT) 
* first(CLM_UTLZTN_DAY_CNT) )+first(CLM_PMT_AMT)))as TOTAL_BENEFICIARY_AMT from df_without_prm group by CLM_ID")
sqlContext.registerDataFrameAsTable(data, "data")  

train_data = sqlContext.sql("select * from data where year!=2010")
test_data = sqlContext.sql("select * from data where year=2010")
train_data = train_data.select("ICD9_DGNS_CD_1","ICD9_DGNS_CD_2","ICD9_PRCDR_CD_1","ICD9_PRCDR_CD_2","TOTAL_BENEFICIARY_AMT")
train_data = train_data.na.fill({'ICD9_DGNS_CD_1':"null"})
train_data = train_data.na.fill({'ICD9_DGNS_CD_2':"null"})
train_data = train_data.na.fill({'ICD9_PRCDR_CD_1':"null"})
train_data = train_data.na.fill({'ICD9_PRCDR_CD_2':"null"})

from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="ICD9_DGNS_CD_1", outputCol="ICD9_DGNS_CD_1N")
indexed = indexer.fit(train_data)
train_data = indexed.transform(train_data)
indexer = StringIndexer(inputCol="ICD9_DGNS_CD_2", outputCol="ICD9_DGNS_CD_2N")
indexed = indexer.fit(train_data)
train_data = indexed.transform(train_data)

train_data = train_data.select("ICD9_DGNS_CD_1N","ICD9_DGNS_CD_2N","ICD9_PRCDR_CD_1N","ICD9_PRCDR_CD_2N","TOTAL_BENEFICIARY_AMT")
train_data= train_data.na.fill({'TOTAL_BENEFICIARY_AMT':0})

from pyspark.ml.feature import VectorAssembler
vectorizer = VectorAssembler()
#datasetDF.select(datasetDF['PE'].alias('features')).show()
vectorizer.setInputCols(["ICD9_DGNS_CD_1N","ICD9_DGNS_CD_2N","ICD9_PRCDR_CD_1N","ICD9_PRCDR_CD_2N"])
vectorizer.setOutputCol("features")

from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml import Pipeline

# Create a DecisionTreeRegressor
dt = DecisionTreeRegressor(maxDepth = 8)

dt.setLabelCol("TOTAL_BENEFICIARY_AMT")\
  .setPredictionCol("Predicted_EXP")\
  .setFeaturesCol("features")\
  .setMaxBins(10000)


# Create a Pipeline
dtPipeline = Pipeline()

# Set the stages of the Pipeline
dtPipeline.setStages([vectorizer, dt])
model = dtPipeline.fit(train_data)
train_data_output=model.transform(train_data)

from pyspark.ml.evaluation import RegressionEvaluator

# Create an RMSE evaluator using the label and predicted columns
regEval = RegressionEvaluator(predictionCol="Predicted_EXP", labelCol="TOTAL_BENEFICIARY_AMT", metricName="r2")

# Run the evaluator on the DataFrame
r2 = regEval.evaluate(train_data_output)

print("Root Mean Squared Error: %.2f" % r2)

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# We can reuse the RegressionEvaluator, regEval, to judge the model based on the best Root Mean Squared Error
# Let's create our CrossValidator with 3 fold cross validation
crossval = CrossValidator(estimator=dtPipeline, evaluator=regEval, numFolds=3)

# Let's tune over our dt.maxDepth parameter on the values 2 and 3, create a paramter grid using the ParamGridBuilder
paramGrid = (ParamGridBuilder()
.addGrid(dt.maxDepth, [6,7,8,9])
.build())

# Add the grid to the CrossValidator
crossval.setEstimatorParamMaps(paramGrid)

# Now let's find and return the best model
dtModel = crossval.fit(train_data).bestModel

train_data_output=dtModel.transform(train_data)

#from pyspark.sql import functions as F
#categories = x.select("ICD9_DGNS_CD_1").distinct().rdd.flatMap(lambda x: x).collect()
#exprs = [F.when(F.col("ICD9_DGNS_CD_1") == category, 1).otherwise(0).alias(category) for category in categories]
#train_data.select("CLM_ID", *(exprs)).show()
#categories

