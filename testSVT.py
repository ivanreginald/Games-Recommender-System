import pandas as pd

# Read the CSV file
data = pd.read_csv('steam_games.csv')

# Sort the names based on the review counts in descending order
sorted_names = data.sort_values('positive_ratings', ascending=False)

# Display the top 5 names with the most reviews
top_5_names = sorted_names.head(5)
print(top_5_names)