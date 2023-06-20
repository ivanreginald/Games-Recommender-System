from numpy import test
from pyspark.sql import functions as F
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.evaluation import RegressionEvaluator, RankingEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import udf, col, when, expr, collect_list, array, lit
import numpy as np
import implicit
import matplotlib.pyplot as plt
from pyspark.mllib.evaluation import RankingMetrics
import pandas as pd
import time

# start_time = time.time()

spark = SparkSession.builder.appName('Test_Rec').config("spark.seed", 30).getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)

ratings_df = spark.read.csv('steam_rating_reformatted3.csv', inferSchema=True, header=True)
# ratings_df.show()

training_df, validation_df = ratings_df.randomSplit([.8, .2])

iterations = 15
regularization_parameter = 0.3
rank = 100

als = ALS(maxIter=iterations, regParam=regularization_parameter, rank=rank, userCol='user_id', itemCol='game_id', ratingCol='rating', coldStartStrategy='drop')
model = als.fit(training_df)
predictions = model.transform(validation_df)

# end_time = time.time()
# execution_time = end_time - start_time
# print("Execution time: {} seconds".format(execution_time))
# evaluator_rmse = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
# evaluator_mae = RegressionEvaluator(metricName='mae', labelCol='rating', predictionCol='prediction')
#
# rmse = evaluator_rmse.evaluate(predictions)
# mae = evaluator_mae.evaluate(predictions)
# print("RMSE = " + str(rmse))
# print("RMSE = " + str(mae))

# model = ALS(userCol='user_id', itemCol='game_id', ratingCol='rating', nonnegative = True, coldStartStrategy="drop")
# paramGrid = ParamGridBuilder().addGrid(model.regParam, [0.01, 0.1, 0.3]).addGrid(model.rank, [30, 50, 100]).addGrid(model.maxIter, [5, 10, 15]).build()
#
# crossvalidation = CrossValidator(estimator=model,
#                                  estimatorParamMaps=paramGrid,
#                                  evaluator=evaluator,
#                                  numFolds=5)
#
# Best_model = crossvalidation.fit(training_df).bestModel
#
# print(type(Best_model))
# print("Best Model")
# print("rank: ", Best_model._java_obj.parent().getRank())
# print("maxIter: ", Best_model._java_obj.parent().getMaxIter())
# print("regParam: ", Best_model._java_obj.parent().getRegParam())
#
# print("Best RMSE value: ", evaluator.evaluate(Best_model.transform(validation_df)))

# predictions.show(n=50)
# userFactorsDF = model.userFactors
# itemFactorsDF = model.itemFactors
#
# userFactorsDF.show()
# predictions.toPandas().to_csv('mycsv.csv')
# print("a")

predictions = predictions.toPandas()
def precision_recall_at_k(df, k, threshold):
    # Sort the dataframe by user_id and prediction in descending order
    df_sorted = df.sort_values(by=['user_id', 'prediction'], ascending=[True, False])

    relevant_items = df_sorted[df_sorted['rating'] >= threshold].shape[0]

    # Group the dataframe by user_id and select the top k rows for each user
    top_k = df_sorted.groupby('user_id').head(k)

    # Apply threshold to determine relevant items
    top_k['predicted_relevant'] = top_k['prediction'] >= threshold

    recommended_item = top_k[df_sorted['prediction'] >= threshold].shape[0]

    # Calculate True Positives (TP)
    tp = top_k[(top_k['rating'] >= threshold) & top_k['predicted_relevant']].shape[0]

    # Calculate False Positives (FP)
    fp = top_k[(top_k['rating'] < threshold) & top_k['predicted_relevant']].shape[0]

    # Calculate False Negatives (FN)
    fn = top_k[(top_k['rating'] >= threshold) & ~top_k['predicted_relevant']].shape[0]

    # Calculate Precision
    precision = tp / (tp + fp)

    # Calculate Recall
    recall = tp / relevant_items

    return precision, recall


k = 5  # Number of items to consider
precision, recall = precision_recall_at_k(predictions, k, 3.5)
print(f'Precision@{k}: {precision:.2f}')
print(f'Recall@{k}: {recall:.2f}')


