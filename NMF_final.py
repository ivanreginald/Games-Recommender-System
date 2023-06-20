import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import time

# start_time = time.time()

# Load data from CSV
data = pd.read_csv('steam_rating_reformatted3.csv')

# data2 = pd.read_csv('steam_games_final.csv')
# data2.set_index(data2.columns[0], inplace=True)
# data2.index = data2.index.astype('int64')

# Create user-item matrix
ratings_matrix = pd.pivot_table(data, values='rating', index='user_id', columns='game_id', fill_value=0)
train_data_mtx, test_data_mtx = train_test_split(ratings_matrix, test_size=0.2, random_state=42)

test_data_df = pd.melt(test_data_mtx.reset_index(), id_vars='user_id', value_name='rating', var_name='game_id')
# test_data_df = test_data_df[test_data_df.iloc[:, 2] != 0]

# Convert ratings matrix to numpy array
ratings_array = np.array(ratings_matrix)

# Split data into training and test sets
train_data = np.array(train_data_mtx)
test_data = np.array(test_data_mtx)


# Tuning hyperparameters: regularization parameter and number of latent factors
# reg_param = [0.01, 0.1, 0.3]
# n_factors = [50, 100, 150]
# itera = [10, 20, 30]
#
#
# best_reg = 0
# best_n = 0
# best_iter = 0
# best_mae = 0
#
# best_rmse = np.inf
# best_model = None
#
# for reg in reg_param:
#     for n in n_factors:
#         for i in itera:
#             # Create an instance of NMF with the current hyperparameters
#             nmf = NMF(n_components=n, alpha_H=reg, max_iter=i, solver='mu')
#
#             # Fit the model to the training data
#             nmf.fit(train_data)
#
#             # Predict ratings for the test data
#             predicted_ratings = nmf.transform(test_data).dot(nmf.components_)
#
#             # Calculate RMSE for non-zero elements
#             non_zero_indices = test_data.nonzero()
#             rmse = np.sqrt(mean_squared_error(test_data[non_zero_indices], predicted_ratings[non_zero_indices]))
#             mae = mean_absolute_error(test_data[non_zero_indices], predicted_ratings[non_zero_indices])
#
#             # Check if current model has lower RMSE
#             if rmse < best_rmse:
#                 best_n = n
#                 best_reg = reg
#                 best_rmse = rmse
#                 best_model = nmf
#                 best_iter = i
#                 best_mae= mae
#
# print("RMSE : "+ str(best_rmse))
# print("best n: "+ str(best_n))
# print("best reg: "+ str(best_reg))
# print("best i: "+ str(best_iter))
# print("best mae: "+ str(best_mae))


nmf = NMF(n_components=150, alpha_H=0.3, max_iter=30, solver='mu')

# Fit the model to the training data
nmf.fit(train_data)
W = nmf.fit_transform(ratings_array)
H = nmf.components_


# new_H = pd.DataFrame(H)
# new_H.to_csv('matrix_item.csv', index=False)

# Predict ratings for the test data
predicted_ratings = nmf.transform(test_data).dot(nmf.components_)

# Calculate RMSE for non-zero elements
non_zero_indices = test_data.nonzero()
rmse = np.sqrt(mean_squared_error(test_data[non_zero_indices], predicted_ratings[non_zero_indices]))
mae = mean_absolute_error(test_data[non_zero_indices], predicted_ratings[non_zero_indices])

print("RMSE: " + str(rmse))
print("MAE: " + str(mae))

rows, cols = np.indices(predicted_ratings.shape)
values = predicted_ratings.ravel()
rows = rows.ravel()
cols = cols.ravel()

new_predicted = {'user': rows, 'game': cols, 'prediction': values}
new_predicted_df = pd.DataFrame(new_predicted)


rows2, cols2 = np.indices(test_data.shape)
values2 = test_data.ravel()
rows2 = rows2.ravel()
cols2 = cols2.ravel()

new_test_data = {'user': rows2, 'game': cols2, 'rating': values2}
new_test_data_df = pd.DataFrame(new_test_data)

new_test_data_df = new_test_data_df[new_test_data_df.iloc[:, 2] != 0]

result = pd.merge(new_test_data_df, new_predicted_df[['user', 'game', 'prediction']], on=['user', 'game'], how='left')


def precision_recall_at_k(df, k, threshold):
    # Sort the dataframe by user_id and prediction in descending order
    df_sorted = df.sort_values(by=['user', 'prediction'], ascending=[True, False])

    relevant_items = df_sorted[df_sorted['rating'] >= threshold].shape[0]

    # Group the dataframe by user_id and select the top k rows for each user
    top_k = df_sorted.groupby('user').head(k)

    recommended_item = top_k[df_sorted['prediction'] >= threshold].shape[0]

    # Apply threshold to determine relevant items
    top_k['predicted_relevant'] = top_k['prediction'] >= threshold

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


k = 5 # Number of items to consider
precision, recall = precision_recall_at_k(result, k, 3.5)
print(f'Precision@{k}: {precision:.2f}')
print(f'Recall@{k}: {recall:.2f}')

print("a")


# nmf = NMF(n_components=100, alpha_H=0.1, max_iter=50, solver='mu')
# W = nmf.fit_transform(ratings_array)
# H = nmf.components_
# end_time = time.time()
# execution_time = end_time - start_time
# print("Execution time: {} seconds".format(execution_time))
#
# # def recommend_similar_games(game_ids, num_recommendations=5):
# #     game_indices = [ratings_matrix.columns.get_loc(game_id) for game_id in game_ids]
# #     game_embeddings = H[:, game_indices]
# #     game_similarities = cosine_similarity(H.T, game_embeddings.T).flatten()
# #     top_indices = np.argsort(game_similarities)[::-1][:num_recommendations]
# #     recommended_games = ratings_matrix.columns[top_indices]
# #     return list(recommended_games.values)
#
# def recommend_similar_games(game_ids, num_recommendations=5):
#     game_indices = [ratings_matrix.columns.get_loc(game_id) for game_id in game_ids]
#     game_embeddings = H[:, game_indices]
#     game_similarities = cosine_similarity(H.T, game_embeddings.T).flatten()
#     top_indices = np.argsort(game_similarities)[::-1][:num_recommendations + len(game_ids)]
#     recommended_games = ratings_matrix.columns[top_indices]
#     recommended_games = [game for game in recommended_games if game not in game_ids][:num_recommendations]
#     return recommended_games
#
# new_H = pd.DataFrame(H)
# new_H.to_csv('matrix_H.csv', index=False)
#
def recommend_similar_games(game_id, num_recommendations=5):
    game_index = ratings_matrix.columns.get_loc(game_id)
    game_embedding = H[:, game_index].reshape(1, -1)  # Reshape to 2D array
    game_similarities = cosine_similarity(game_embedding, H.T).flatten()
    top_indices = np.argsort(game_similarities)[::-1]
    similar_games = ratings_matrix.columns[top_indices]
    similar_games = [game for game in similar_games if game != game_id][:num_recommendations]
    return similar_games
#
# # Example usage
# # game_ids = [10, 30, 9450]
# # recommendations = recommend_similar_games(game_ids, num_recommendations=5)
# # print(f"Top 5 recommended games similar to {game_ids}:")
# # print(recommendations)
#
similar_games_data = pd.DataFrame(columns=['game_id', 'similar_games'])

game_ids = ratings_matrix.columns
for game_id in game_ids:
    similar_games = recommend_similar_games(game_id)
    similar_games_data = similar_games_data.append({'game_id': game_id, 'similar_games': similar_games}, ignore_index=True)

# Save the new DataFrame to a CSV file
similar_games_data.to_csv('similar_games_new.csv', index=False)

print("a")

