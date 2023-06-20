import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

start_time = time.time()

# Step 1: Load the CSV file into a Pandas DataFrame
df = pd.read_csv('steam_rating_reformatted3.csv')

# Step 2: Perform train-test split
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Step 3: Create a user-item matrix
user_ids = df['user_id'].unique()
game_ids = df['game_id'].unique()
num_users = len(user_ids)
num_games = len(game_ids)

# Create a mapping from user_id to user index
user_id_to_index = {user_id: index for index, user_id in enumerate(user_ids)}

# Create a mapping from game_id to game index
game_id_to_index = {game_id: index for index, game_id in enumerate(game_ids)}

train_matrix = np.zeros((num_users, num_games))
for row in train_data.itertuples():
    user_index = user_id_to_index[row[1]]
    game_index = game_id_to_index[row[2]]
    train_matrix[user_index, game_index] = row[3]

# Step 4: Perform Singular Value Decomposition on the training matrix
U, sigma, Vt = svds(train_matrix, k=20)
sigma = np.diag(sigma)
sigma_svt = np.maximum(sigma - 0.1, 0)

# Step 5: Predict ratings for the test data
predicted_ratings = np.dot(np.dot(U, sigma_svt), Vt)
predicted_ratings = np.round(predicted_ratings)

end_time = time.time()
execution_time = end_time - start_time
print("Execution time: {} seconds".format(execution_time))

# Step 6: Extract the test data ratings
test_ratings = []
for row in test_data.itertuples():
    user_index = user_id_to_index[row[1]]
    game_index = game_id_to_index[row[2]]
    test_ratings.append(predicted_ratings[user_index, game_index])

# Step 7: Calculate the Root Mean Square Error (RMSE)
rmse = np.sqrt(mean_squared_error(test_data['rating'], test_ratings))
mae = mean_absolute_error(test_data['rating'], test_ratings)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")


rows, cols = np.indices(predicted_ratings.shape)
values = predicted_ratings.ravel()
rows = rows.ravel()
cols = cols.ravel()

new_predicted = {'user': rows, 'game': cols, 'prediction': values}
new_predicted_df = pd.DataFrame(new_predicted)

test_data_pt = pd.pivot_table(test_data, values='rating', index='user_id', columns='game_id', fill_value=0)
test_data_arr = np.array(test_data_pt)

rows2, cols2 = np.indices(test_data_arr.shape)
values2 = test_data_arr.ravel()
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

print("A")


