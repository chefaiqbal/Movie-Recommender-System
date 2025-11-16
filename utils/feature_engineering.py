"""
Feature Engineering for Matrix Factorization Recommender System

This module computes demographic and genre-based features to enhance
PMF model performance by incorporating user preferences and affinities.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def compute_user_genre_preferences(ratings_df, movies_df):
    """
    Calculate each user's average rating per genre.
    
    This creates a user-genre preference matrix where each entry
    represents how much a user likes a particular genre based on
    their historical ratings.
    
    Args:
        ratings_df (pd.DataFrame): Ratings with UserID, MovieID, Rating
        movies_df (pd.DataFrame): Movies with MovieID, Title, Genres
        
    Returns:
        pd.DataFrame: User-genre preference matrix (users x genres)
    """
    print("\nComputing user-genre preferences...")
    
    # Merge ratings with movie genres
    ratings_with_genres = ratings_df.merge(movies_df[['MovieID', 'Genres']], on='MovieID')
    
    # Split pipe-separated genres into lists
    ratings_with_genres['Genres'] = ratings_with_genres['Genres'].str.split('|')
    
    # Explode genres so each rating-genre combination gets its own row
    ratings_exploded = ratings_with_genres.explode('Genres')
    
    # Calculate average rating per user per genre
    user_genre_prefs = ratings_exploded.groupby(['UserID', 'Genres'])['Rating'].mean()
    user_genre_prefs = user_genre_prefs.unstack(fill_value=0)
    
    print(f"User-genre preferences shape: {user_genre_prefs.shape}")
    print(f"Genres: {list(user_genre_prefs.columns)}")
    
    return user_genre_prefs


def compute_demographic_genre_affinity(ratings_df, users_df, movies_df):
    """
    Calculate how different demographics rate different genres.
    
    This computes affinity scores showing which demographics prefer
    which genres (e.g., Age 18-24 + Action movies = high rating).
    
    Args:
        ratings_df (pd.DataFrame): Ratings with UserID, MovieID, Rating
        users_df (pd.DataFrame): Users with UserID, Gender, Age, Occupation
        movies_df (pd.DataFrame): Movies with MovieID, Genres
        
    Returns:
        dict: Dictionary with 'age_genre', 'occupation_genre', 'gender_genre' DataFrames
    """
    print("\nComputing demographic-genre affinities...")
    
    # Merge all data
    merged = ratings_df.merge(users_df, on='UserID').merge(
        movies_df[['MovieID', 'Genres']], on='MovieID'
    )
    
    # Split genres
    merged['Genres'] = merged['Genres'].str.split('|')
    merged_exploded = merged.explode('Genres')
    
    # Age-Genre affinity
    age_genre = merged_exploded.groupby(['Age', 'Genres'])['Rating'].mean().unstack(fill_value=0)
    print(f"Age-Genre affinity: {age_genre.shape}")
    
    # Occupation-Genre affinity
    occupation_genre = merged_exploded.groupby(['Occupation', 'Genres'])['Rating'].mean().unstack(fill_value=0)
    print(f"Occupation-Genre affinity: {occupation_genre.shape}")
    
    # Gender-Genre affinity
    gender_genre = merged_exploded.groupby(['Gender', 'Genres'])['Rating'].mean().unstack(fill_value=0)
    print(f"Gender-Genre affinity: {gender_genre.shape}")
    
    return {
        'age_genre': age_genre,
        'occupation_genre': occupation_genre,
        'gender_genre': gender_genre
    }


def get_movie_primary_genre(genres_str):
    """
    Extract the primary (first) genre from pipe-separated genres.
    
    Args:
        genres_str (str): Pipe-separated genres (e.g., "Action|Adventure|Sci-Fi")
        
    Returns:
        str: Primary genre
    """
    return genres_str.split('|')[0]


def get_movie_genre_vector(genres_str, all_genres):
    """
    Convert pipe-separated genres into a binary vector.
    
    Args:
        genres_str (str): Pipe-separated genres
        all_genres (list): List of all possible genres
        
    Returns:
        np.ndarray: Binary genre vector
    """
    movie_genres = set(genres_str.split('|'))
    return np.array([1 if genre in movie_genres else 0 for genre in all_genres])


def compute_demographic_bias_for_ratings(ratings_df, users_df, movies_df, affinity_dict):
    """
    Compute demographic bias for each rating based on user demographics and movie genres.
    
    This adds a 'DemographicBias' column to ratings indicating how much
    users with similar demographics typically rate movies in those genres,
    CENTERED AROUND THE GLOBAL MEAN (so bias represents deviation, not absolute rating).
    
    Args:
        ratings_df (pd.DataFrame): Ratings with UserID, MovieID, Rating
        users_df (pd.DataFrame): Users with UserID, Gender, Age, Occupation
        movies_df (pd.DataFrame): Movies with MovieID, Genres
        affinity_dict (dict): Output from compute_demographic_genre_affinity
        
    Returns:
        pd.DataFrame: Ratings with added DemographicBias column
    """
    print("\nComputing demographic bias for each rating...")
    
    # Calculate global mean rating
    global_mean = ratings_df['Rating'].mean()
    print(f"Global mean rating: {global_mean:.4f}")
    
    # Merge data
    ratings_with_demo = ratings_df.merge(
        users_df[['UserID', 'Gender', 'Age', 'Occupation']], on='UserID'
    ).merge(
        movies_df[['MovieID', 'Genres']], on='MovieID'
    )
    
    # Get affinity matrices
    age_genre = affinity_dict['age_genre']
    occupation_genre = affinity_dict['occupation_genre']
    gender_genre = affinity_dict['gender_genre']
    
    # Compute bias for each rating
    demographic_biases = []
    
    for _, row in ratings_with_demo.iterrows():
        genres = row['Genres'].split('|')
        
        # Get affinities for this user's demographics across movie genres
        try:
            age_affinities = [age_genre.loc[row['Age'], genre] if genre in age_genre.columns else global_mean 
                             for genre in genres]
            occ_affinities = [occupation_genre.loc[row['Occupation'], genre] if genre in occupation_genre.columns else global_mean 
                             for genre in genres]
            gender_affinities = [gender_genre.loc[row['Gender'], genre] if genre in gender_genre.columns else global_mean 
                               for genre in genres]
            
            # Average across genres and demographics
            age_affinity = np.mean(age_affinities) if age_affinities else global_mean
            occ_affinity = np.mean(occ_affinities) if occ_affinities else global_mean
            gender_affinity = np.mean(gender_affinities) if gender_affinities else global_mean
            
            # Combine affinities (weighted average)
            combined_affinity = 0.4 * age_affinity + 0.3 * occ_affinity + 0.3 * gender_affinity
            
            # Convert to BIAS by subtracting global mean
            # This makes the bias centered around 0
            combined_bias = combined_affinity - global_mean
            
        except (KeyError, IndexError):
            combined_bias = 0
        
        demographic_biases.append(combined_bias)
    
    ratings_with_demo['DemographicBias'] = demographic_biases
    
    print(f"Demographic bias range: [{min(demographic_biases):.4f}, {max(demographic_biases):.4f}]")
    print(f"Demographic bias mean: {np.mean(demographic_biases):.4f}")
    print(f"Demographic bias std: {np.std(demographic_biases):.4f}")
    
    return ratings_with_demo[['UserID', 'MovieID', 'Rating', 'DemographicBias']]


def extract_all_genres(movies_df):
    """
    Extract unique list of all genres from movies dataset.
    
    Args:
        movies_df (pd.DataFrame): Movies with Genres column
        
    Returns:
        list: Sorted list of unique genres
    """
    all_genres = set()
    for genres_str in movies_df['Genres']:
        all_genres.update(genres_str.split('|'))
    
    return sorted(list(all_genres))


def create_genre_matrix(movies_df):
    """
    Create a movie-genre binary matrix.
    
    Args:
        movies_df (pd.DataFrame): Movies with MovieID and Genres
        
    Returns:
        pd.DataFrame: Binary matrix (movies x genres)
    """
    all_genres = extract_all_genres(movies_df)
    
    genre_matrix = pd.DataFrame(
        index=movies_df['MovieID'],
        columns=all_genres,
        dtype=int
    )
    
    for idx, row in movies_df.iterrows():
        movie_id = row['MovieID']
        genres = row['Genres'].split('|')
        for genre in all_genres:
            genre_matrix.loc[movie_id, genre] = 1 if genre in genres else 0
    
    return genre_matrix


def normalize_demographic_features(users_df):
    """
    Normalize demographic features to [0, 1] range.
    
    Args:
        users_df (pd.DataFrame): Users with Age, Gender, Occupation
        
    Returns:
        pd.DataFrame: Users with normalized features
    """
    users_normalized = users_df.copy()
    
    # Normalize age (already categorical: 1, 18, 25, 35, 45, 50, 56)
    # Map to normalized values
    age_mapping = {1: 0.0, 18: 0.3, 25: 0.4, 35: 0.6, 45: 0.75, 50: 0.85, 56: 1.0}
    users_normalized['AgeNorm'] = users_normalized['Age'].map(age_mapping)
    
    # Gender to binary (M=1, F=0)
    users_normalized['GenderBinary'] = (users_normalized['Gender'] == 'M').astype(int)
    
    # Normalize occupation (0-20 range)
    users_normalized['OccupationNorm'] = users_normalized['Occupation'] / 20.0
    
    return users_normalized


def compute_user_feature_matrix(users_df):
    """
    Create a feature matrix for users combining demographics.
    
    Args:
        users_df (pd.DataFrame): Users with demographic info
        
    Returns:
        pd.DataFrame: User feature matrix
    """
    users_norm = normalize_demographic_features(users_df)
    
    feature_matrix = pd.DataFrame({
        'UserID': users_norm['UserID'],
        'Age': users_norm['AgeNorm'],
        'Gender': users_norm['GenderBinary'],
        'Occupation': users_norm['OccupationNorm']
    }).set_index('UserID')
    
    return feature_matrix


if __name__ == "__main__":
    # Test the feature engineering functions
    from data_loader import load_ratings, load_users, load_movies
    
    print("Testing Feature Engineering...")
    
    ratings = load_ratings('data/ratings.dat')
    users = load_users('data/users.dat')
    movies = load_movies('data/movies.dat')
    
    # Test user-genre preferences
    user_genre_prefs = compute_user_genre_preferences(ratings, movies)
    print(f"\nUser-genre preferences:\n{user_genre_prefs.head()}")
    
    # Test demographic-genre affinity
    affinity = compute_demographic_genre_affinity(ratings, users, movies)
    print(f"\nAge-genre affinity:\n{affinity['age_genre'].head()}")
    
    # Test all genres extraction
    genres = extract_all_genres(movies)
    print(f"\nAll genres ({len(genres)}): {genres}")
