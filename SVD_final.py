import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse.linalg import svds

data = pd.read_csv('steam_users_reformatted3.csv')

# Create a user-item rating matrix

train_data, test_data = train_test_split(data, test_size=0.2)

# Create train and test user-item rating matrices

# # Define the range of k values to try
# k_values = [10, 20, 30]
#
# # Regularization parameter
# tau = [0.01, 0.1, 0.3]
#
# best_rmse = np.inf
# best_tau = 0
# best_k = 0

# # Iterate over different k values
# for k in k_values:
#     for t in tau:
#         # Perform matrix factorization with regularization
#         U, sigma, Vt = svds(train_ratings, k=k)
#
#         sigma = np.diag(sigma)
#
#         sigma_svt = np.maximum(sigma - t, 0)
#
#         # Make predictions on the test data by reconstructing the matrix
#         predicted_ratings = np.dot(np.dot(U, sigma_svt), Vt)
#
#         # Flatten the original and predicted ratings matrices for evaluation
#         test_true_ratings = test_ratings[test_ratings.nonzero()]
#         test_pred_ratings = predicted_ratings[test_ratings.nonzero()]
#
#         # Calculate RMSE
#         rmse = np.sqrt(mean_squared_error(test_true_ratings, test_pred_ratings))
#
#         # Print RMSE for the current k value
#         if rmse < best_rmse:
#             best_tau = t
#             best_rmse = rmse
#             best_k = k
#
#         print("RMSE : " + str(rmse))
#         print("tau: " + str(t))
#         print("k: " + str(k))
#         print("")
#
# print("+++++++++++++++++++++++++")
# print("RMSE : "+ str(best_rmse))
# print("best tau: "+ str(best_tau))
# print("best k: "+ str(best_k))


# Calculate RMSE
# rmse = np.sqrt(mean_squared_error(test_true_ratings, test_pred_ratings))
# mae = mean_absolute_error(test_true_ratings, test_pred_ratings)
#
# print("RMSE :" + str(rmse))
# print("MAE :" + str(mae))


# def calculate_precision_recall(actual_ratings, predicted_ratings, k, tolerance):
#     # Sort the predicted ratings in descending order
#     sorted_indices = np.argsort(predicted_ratings)[::-1]
#     top_k_indices = sorted_indices[:k]
#
#     # Count the number of predicted ratings within the tolerance value
#     num_correct = 0
#     for i in top_k_indices:
#         if abs(predicted_ratings[i] - actual_ratings[i]) <= tolerance:
#             num_correct += 1
#
#     # Calculate precision@k and recall@k
#     precision = num_correct / k
#     recall = num_correct / len(actual_ratings)
#
#     return precision, recall
#
# k = 10
# tolerance = 0.5
#
# precision, recall = calculate_precision_recall(test_true_ratings, test_pred_ratings , k, tolerance)
#
# print("Precision@{}: {:.2f}".format(k, precision))
# print("Recall@{}: {:.2f}".format(k, recall))