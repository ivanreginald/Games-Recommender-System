import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('steam_rating_reformatted3.csv')
# Assuming your DataFrame is named 'df' with columns: user_id, game_id, and rating

# Group the DataFrame by rating and calculate the count
rating_counts = df['rating'].value_counts().sort_index()

# Plot the counts
plt.bar(rating_counts.index, rating_counts.values)

# Set labels and title
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Count of Ratings')

# Show the plot
plt.show()