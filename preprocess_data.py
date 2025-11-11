"""
Data preprocessing script for Matrix Factorization Recommender System.

This script:
1. Loads MovieLens dataset
2. Preprocesses data (removes nulls)
3. Creates user-item matrix
4. Splits into train/test sets (random_state=42)
5. Normalizes the matrix
6. Saves the processed data
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.data_loader import (
    load_ratings, 
    load_users, 
    load_movies, 
    preprocess_data,
    get_dataset_stats
)
from utils.matrix_creation import (
    create_user_item_matrix,
    split_train_test,
    normalize_matrix,
    save_matrix
)


def main():
    """Main preprocessing pipeline."""
    
    print("="*60)
    print("Matrix Factorization - Data Preprocessing")
    print("="*60)
    
    # Step 1: Load data
    print("\n[Step 1] Loading MovieLens Dataset...")
    ratings = load_ratings('data/ratings.dat')
    users = load_users('data/users.dat')
    movies = load_movies('data/movies.dat')
    
    # Step 2: Preprocess data (remove nulls)
    print("\n[Step 2] Preprocessing Data...")
    ratings, users, movies = preprocess_data(ratings, users, movies)
    
    # Display dataset statistics
    get_dataset_stats(ratings)
    
    # Step 3: Split data into train/test sets
    print("\n[Step 3] Splitting Data into Train/Test Sets...")
    train_ratings, test_ratings = split_train_test(
        ratings, 
        test_size=0.2, 
        random_state=42
    )
    
    # Save train/test splits for later use
    train_ratings.to_csv('data/processed/train_ratings.csv', index=False)
    test_ratings.to_csv('data/processed/test_ratings.csv', index=False)
    print("Train/test splits saved to data/processed/")
    
    # Step 4: Create user-item matrix from training data
    print("\n[Step 4] Creating User-Item Matrix...")
    user_item_matrix = create_user_item_matrix(train_ratings)
    
    # Step 5: Normalize the matrix
    print("\n[Step 5] Normalizing User-Item Matrix...")
    normalized_matrix, norm_params = normalize_matrix(
        user_item_matrix, 
        method='mean_centering'
    )
    
    # Save normalization parameters
    np.save('data/processed/normalization_params.npy', norm_params)
    print("Normalization parameters saved to data/processed/normalization_params.npy")
    
    # Step 6: Save the normalized matrix
    print("\n[Step 6] Saving Processed Data...")
    save_matrix(normalized_matrix, 'data/processed/user_item_matrix.csv')
    
    # Also save the original (non-normalized) matrix for reference
    save_matrix(user_item_matrix, 'data/processed/user_item_matrix_original.csv')
    
    # Save movies and users data for later use
    movies.to_csv('data/processed/movies.csv', index=False)
    users.to_csv('data/processed/users.csv', index=False)
    print("Movies and users data saved to data/processed/")
    
    # Print summary
    print("\n" + "="*60)
    print("Preprocessing Complete!")
    print("="*60)
    print(f"✓ User-item matrix shape: {user_item_matrix.shape}")
    print(f"✓ Normalized matrix saved: data/processed/user_item_matrix.csv")
    print(f"✓ Original matrix saved: data/processed/user_item_matrix_original.csv")
    print(f"✓ Train ratings: {len(train_ratings)}")
    print(f"✓ Test ratings: {len(test_ratings)}")
    print(f"✓ Normalization method: mean_centering")
    print("="*60)


if __name__ == "__main__":
    main()
