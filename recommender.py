"""Recommender engine: functions for building user profiles and recommending movies."""

import numpy as np
from similarity import cosine_similarity



def genre_vector(genres_str, all_genres):
    """Return a binary vector for `genres_str` against `all_genres`.

    - `genres_str`: pipe-separated genre string from the CSV (e.g. "Comedy|Drama").
    - `all_genres`: ordered list of all genres used for vector positions.
    """
    # Initialize vector with zeros (one position per genre)
    vector = [0] * len(all_genres)
    # Split the movie's genres: "Comedy|Drama" → ["Comedy", "Drama"]
    movie_genres = genres_str.split("|")

    # For each genre in our master list, check if this movie has it
    for idx, genre in enumerate(all_genres):
        # idx: position (0, 1, 2, ...), genre: the actual genre name
        if genre in movie_genres:
            vector[idx] = 1  # Mark 1 if movie has this genre

    return vector



def build_user_profile(user_id, ratings, movies, all_genres, min_rating=4.0):
    """Build a user profile by averaging genre vectors of their highly-rated movies.
    
    A user profile summarizes what genres they like based on movies they rated ≥4.0.
    """
    # Filter ratings: get only this user's ratings
    user_ratings = ratings[ratings["userId"] == user_id]
    # Filter again: keep only movies they rated well (≥ min_rating, default 4.0)
    liked_movies = user_ratings[user_ratings["rating"] >= min_rating]

    # np.zeros() creates an array of 0s (faster than Python lists for math)
    # This will accumulate genre vectors as we loop
    profile_vec = np.zeros(len(all_genres))
    count = 0  # Count how many movies we processed
    
    # .iterrows(): loop through each row in the DataFrame
    for _, row in liked_movies.iterrows():
        movie_id = row["movieId"]
        # Find this movie in the movies DataFrame
        movie_row = movies[movies["movieId"] == movie_id]

        if movie_row.empty:
            continue  # Skip if movie not found
        # .iloc[0] = first (and only) matching row
        genres_str = movie_row.iloc[0]["genres"]
        vec = genre_vector(genres_str, all_genres)

        # += is shorthand for: profile_vec = profile_vec + np.array(vec)
        # We're summing up all the genre vectors
        profile_vec += np.array(vec)
        count += 1

    # Average the summed vectors by dividing by the count
    # This gives us proportions (0.0 to 1.0) for each genre
    if count > 0:
        profile_vec = profile_vec / count

    return profile_vec



def recommend_movies(user_profile, movies, movie_vectors, top_n=5):
    """Recommend top N movies using a pre-computed `movie_vectors` dict.

    The old implementation recalculated every movie's genre vector on the fly
    (shown below) which is slow. We keep that version in a comment for
    learning, and use the faster precomputed approach here.
    """

    # Old (slower) implementation kept here for reference:
    # """
    # def recommend_movies(user_profile, movies, all_genres, top_n=5):
    #     scores = []
    #     for _, movie in movies.iterrows():
    #         movie_vec = genre_vector(movie["genres"], all_genres)
    #         score = cosine_similarity(user_profile, movie_vec)
    #         scores.append((movie["title"], score))
    #     scores.sort(key=lambda x: x[1], reverse=True)
    #     return scores[:top_n]
    # """

    scores = []
    for _, movie in movies.iterrows():
        movie_id = movie["movieId"]
        title = movie["title"]
        vec = movie_vectors.get(movie_id)
        if vec is None:
            # If a precomputed vector is missing, skip or compute on the fly.
            # Here we skip to keep the behavior predictable.
            continue
        score = cosine_similarity(user_profile, vec)
        scores.append((title, score))

    # Sort movies by score (highest first) and take the top N
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]



def pre_compute_all_genre_vectors(movies, all_genres):
    """Pre-compute genre vectors for all movies to speed up recommendations."""
    genre_vecs = {}
    for _, movie in movies.iterrows():
      movie_id = movie["movieId"]
      vec = genre_vector(movie["genres"], all_genres)
      genre_vecs[movie_id] = vec
    return genre_vecs


