from flask import Flask, render_template
import pandas as pd

app = Flask(__name__, template_folder='template')
items_per_page = 30
df = pd.read_csv('steam_users_reformatted3.csv')

@app.route('/page/<int:page>')
def paginate(page):
    total_items = df['game_id'].nunique()
    num_pages = total_items // items_per_page + 1

    start_index = (page - 1) * items_per_page
    end_index = start_index + items_per_page

    items = df['game_id'].unique()[start_index:end_index]

    return render_template('index.html', items=items, page=page, num_pages=num_pages)

if __name__ == '__main__':
    app.run()