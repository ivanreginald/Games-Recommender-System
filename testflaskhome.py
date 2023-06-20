from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import ast

from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder='template', static_folder='static')
steam_games = pd.read_csv('steam_games_final2.csv')

@app.route('/home', methods=['GET', 'POST'])
def home():

    sorted_games = steam_games.sort_values('positive_ratings', ascending=False)
    best5 = sorted_games.head(5)
    best5_dict = best5.to_dict('records')

    return render_template('home.html', data=best5_dict)

@app.route('/game-list/<int:page>', methods=['GET', 'POST'])
def game_list(page):

    sorted_games = steam_games.sort_values('name', ascending=True)
    games = sorted_games.to_dict('records')

    items_per_page = 30
    total_items = steam_games['name'].nunique()
    num_pages = total_items // items_per_page + 1

    start_index = (page - 1) * items_per_page
    end_index = start_index + items_per_page

    games_data = games[start_index:end_index]

    return render_template('game_list.html', data=games_data, page=page, num_pages=num_pages)

@app.route('/game-detail', methods=['GET', 'POST'])
def game_detail():

    df = pd.read_csv('similar_games_new.csv')

    game_detail_data = request.form.get('data')
    data = ast.literal_eval(game_detail_data)
    similar = df[df['game_id'] == data['appid']]
    similar_list = ast.literal_eval(similar['similar_games'].iloc[0])

    similar_game = steam_games[steam_games['appid'].isin(similar_list)]
    similar_game_dict = similar_game.to_dict('records')

    return render_template('game_detail.html', data=data, similar_data=similar_game_dict)

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    return render_template('recommend.html')

@app.route('/recommend/game', methods=['GET', 'POST'])
def recommend_game():
    query = request.args.get('query')  # Retrieve the search query from the request
    game_list = steam_games['name'].to_list()
    # Perform your search logic here, e.g., filtering the book list based on the query
    results = [game for game in game_list if query.lower() in game.lower()]

    return jsonify(results)

@app.route('/recommend-result', methods=['GET', 'POST'])
def recommend_result():
    input1 = request.args.get('game1')
    input2 = request.args.get('game2')
    input3 = request.args.get('game3')

    game_list = []
    if input1 != "": game_list.append(input1)
    if input2 != "": game_list.append(input2)
    if input3 != "": game_list.append(input3)

    steam_games_df = pd.read_csv('steam_games_final2.csv')

    game_df = steam_games_df[steam_games_df['name'].isin(game_list)]
    inputted_game = game_df.to_dict('records')
    game_ids = game_df['appid'].tolist()

    H_df = pd.read_csv('matrix_item.csv')
    H = np.array(H_df)

    steam_games_df.set_index(steam_games_df.columns[0], inplace=True)
    steam_games_df.index = steam_games_df.index.astype('int64')

    def recommend_similar_games(game_ids, num_recommendations=5):
        game_indices = [steam_games_df.index.get_loc(game_id) for game_id in game_ids]
        game_embeddings = H[:, game_indices]
        game_embeddings_avg = np.mean(game_embeddings, axis=1).reshape((-1, 1))
        game_similarities = cosine_similarity(H.T, game_embeddings_avg.T).flatten()
        top_indices = np.argsort(game_similarities)[::-1][:num_recommendations + len(game_ids)]
        recommended_games = steam_games_df.index[top_indices]
        recommended_games = [game for game in recommended_games if game not in game_ids][:num_recommendations]
        return recommended_games

    recommendations = recommend_similar_games(game_ids, num_recommendations=5)

    similar_game = steam_games[steam_games['appid'].isin(recommendations)]
    similar_game_dict = similar_game.to_dict('records')

    return render_template('recommend_result.html', similar_data=similar_game_dict, input_data=inputted_game)

if __name__ == '__main__':
    app.run()
