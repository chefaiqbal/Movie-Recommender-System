"""
Data loading utilities for MovieLens dataset.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_ratings(data_path='data/ratings.dat'):
    """
    Load ratings data from MovieLens dataset.
    
    Args:
        data_path (str): Path to ratings.dat file
        
    Returns:
        pd.DataFrame: DataFrame with columns [UserID, MovieID, Rating, Timestamp]
    """
    ratings = pd.read_csv(
        data_path,
        sep='::',
        engine='python',
        names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
        encoding='latin-1'
    )
    
    print(f"Loaded {len(ratings)} ratings")
    print(f"Number of users: {ratings['UserID'].nunique()}")
    print(f"Number of movies: {ratings['MovieID'].nunique()}")
    print(f"Rating range: {ratings['Rating'].min()} - {ratings['Rating'].max()}")
    
    return ratings


def load_users(data_path='data/users.dat'):
    """
    Load users data from MovieLens dataset.
    
    Args:
        data_path (str): Path to users.dat file
        
    Returns:
        pd.DataFrame: DataFrame with columns [UserID, Gender, Age, Occupation, Zip-code]
    """
    users = pd.read_csv(
        data_path,
        sep='::',
        engine='python',
        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'],
        encoding='latin-1'
    )
    
    print(f"Loaded {len(users)} users")
    
    return users


def load_movies(data_path='data/movies.dat'):
    """
    Load movies data from MovieLens dataset.
    
    Args:
        data_path (str): Path to movies.dat file
        
    Returns:
        pd.DataFrame: DataFrame with columns [MovieID, Title, Genres]
    """
    movies = pd.read_csv(
        data_path,
        sep='::',
        engine='python',
        names=['MovieID', 'Title', 'Genres'],
        encoding='latin-1'
    )
    
    print(f"Loaded {len(movies)} movies")
    
    return movies


def preprocess_data(ratings, users, movies):
    """
    Preprocess the data by removing null values and merging datasets.
    
    Args:
        ratings (pd.DataFrame): Ratings dataframe
        users (pd.DataFrame): Users dataframe
        movies (pd.DataFrame): Movies dataframe
        
    Returns:
        pd.DataFrame: Cleaned and merged dataframe
    """
    # Check for null values
    print("\nChecking for null values...")
    print(f"Ratings nulls: {ratings.isnull().sum().sum()}")
    print(f"Users nulls: {users.isnull().sum().sum()}")
    print(f"Movies nulls: {movies.isnull().sum().sum()}")
    
    # Remove any null values
    ratings = ratings.dropna()
    users = users.dropna()
    movies = movies.dropna()
    
    print(f"\nAfter removing nulls:")
    print(f"Ratings: {len(ratings)}")
    print(f"Users: {len(users)}")
    print(f"Movies: {len(movies)}")
    
    return ratings, users, movies


def get_dataset_stats(ratings):
    """
    Print statistics about the dataset.
    
    Args:
        ratings (pd.DataFrame): Ratings dataframe
    """
    print("\n=== Dataset Statistics ===")
    print(f"Total ratings: {len(ratings)}")
    print(f"Unique users: {ratings['UserID'].nunique()}")
    print(f"Unique movies: {ratings['MovieID'].nunique()}")
    print(f"Sparsity: {1 - len(ratings) / (ratings['UserID'].nunique() * ratings['MovieID'].nunique()):.4f}")
    print(f"\nRating distribution:")
    print(ratings['Rating'].value_counts().sort_index())


def filter_sparse_users_items(ratings, min_user_ratings=25, min_item_ratings=10):
    """
    Remove users and items with too few ratings to reduce sparsity.
    
    This helps the model by focusing on users/items with sufficient data
    and can improve RMSE by removing noisy/sparse data points.
    
    Args:
        ratings (pd.DataFrame): Ratings dataframe
        min_user_ratings (int): Minimum ratings per user
        min_item_ratings (int): Minimum ratings per item
        
    Returns:
        pd.DataFrame: Filtered ratings dataframe
    """
    print(f"\n[Sparsity Filtering]")
    print(f"Original: {len(ratings)} ratings, {ratings['UserID'].nunique()} users, {ratings['MovieID'].nunique()} movies")
    
    # Iteratively remove sparse users and items
    prev_len = 0
    iteration = 0
    
    while len(ratings) != prev_len and iteration < 5:
        iteration += 1
        prev_len = len(ratings)
        
        # Remove users with too few ratings
        user_counts = ratings['UserID'].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        ratings = ratings[ratings['UserID'].isin(valid_users)]
        
        # Remove items with too few ratings
        item_counts = ratings['MovieID'].value_counts()
        valid_items = item_counts[item_counts >= min_item_ratings].index
        ratings = ratings[ratings['MovieID'].isin(valid_items)]
    
    print(f"After filtering (iter={iteration}): {len(ratings)} ratings, {ratings['UserID'].nunique()} users, {ratings['MovieID'].nunique()} movies")
    print(f"Sparsity: {1 - len(ratings) / (ratings['UserID'].nunique() * ratings['MovieID'].nunique()):.4f}")
    
    return ratings
