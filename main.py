from similarity import cosine_similarity
from recommender import genre_vector, build_user_profile, pre_compute_all_genre_vectors, recommend_movies
import numpy as np  # NumPy: fast math library for arrays and numerical operations
import pandas as pd  # Pandas: library for working with spreadsheet-like data (DataFrames)

# Use loader from data package (loads from data/ml-latest-small/...)
from data.load_movielens import load_movies, load_ratings

# Load the MovieLens dataset into pandas DataFrames
# DataFrames are like tables with rows and columns
movies = load_movies()   # Each row is a movie with: movieId, title, genres
ratings = load_ratings() # Each row is a user rating with: userId, movieId, rating, timestamp

# Extract ALL unique genres from the dataset
# Genres in the CSV are pipe-separated: "Comedy|Drama|Action"
# We need a complete list to use as the "axis" for our genre vectors
all_genres = set()  # Set: collection with no duplicates
for genres in movies["genres"]:
    # genres.split("|") converts "Comedy|Drama" → ["Comedy", "Drama"]
    for genre in genres.split("|"):
        all_genres.add(genre)  # Add to set (duplicates auto-removed)

# Convert set to sorted list for consistent ordering
all_genres = sorted(list(all_genres))

#Pre-compute all genre vectors once and not 10k times each recommendation
movie_vectors = pre_compute_all_genre_vectors(movies, all_genres)

print("All genres:") 
print(all_genres)
print("Number of genres:", len(all_genres)) 
# Test: convert the first movie to a genre vector
# .iloc[0] means "integer location 0" = first row (index 0)
first_movie = movies.iloc[0]
vec = genre_vector(first_movie["genres"], all_genres)

print("First movie:", first_movie["title"])
print("Genres:", first_movie["genres"])
print("Genre vector:", vec)  # Shows 1s for the movie's genres, 0s for others

print("\n" + "="*50)
print("MOVIE RECOMMENDER - Get personalized recommendations!")
print("="*50)

# Ask the user for their user ID
user_input = input("\nEnter a user ID (1-610): ")

# Try to convert the input to an integer
# Error handling- what if they type "abc" instead of a number?
try:
    user_id = int(user_input)  # Convert the text they typed to a number
except ValueError:
    # ValueError = they didn't enter a valid number
    print("Error: Please enter a valid number!")
    exit()  # Stop the program

# Check if the user ID actually exists in our ratings data
# If they enter 9999 but we only have users 1-610 we show an error message and exit
unique_user_ids = ratings["userId"].unique()  # Get list of all user IDs in dataset
if user_id not in unique_user_ids:
    print(f"Error: User {user_id} not found! Available users: 1-{max(unique_user_ids)}")
    exit()  # Stop the program

print(f"✓ Found user {user_id}! Building profile...")

# Build the user's profile (what genres do they like?)
user_profile = build_user_profile(user_id, ratings, movies, all_genres)

print(f"\nUser {user_id} Profile Vector:")
print(user_profile) # Shows the user's preferences for each genre (0.0 to 1.0)

# Get movie recommendations based on the user's profile (use precomputed vectors)
recommendations = recommend_movies(user_profile, movies, movie_vectors, top_n=10)
print(f"\nTop 10 Recommendations for User {user_id}:")
for title, score in recommendations:
    # f"..." is an f-string (formatted string literal) - lets you embed {expressions}
    # .4f means "float with 4 decimal places" - e.g., 3.14159 → 3.1416
    print(f"{title}: score={score:.4f}")

