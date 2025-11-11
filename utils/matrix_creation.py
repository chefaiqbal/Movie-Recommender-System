"""
User-item matrix creation and preprocessing utilities.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def create_user_item_matrix(ratings):
    """
    Create a user-item interaction matrix from ratings data.
    
    Args:
        ratings (pd.DataFrame): DataFrame with columns [UserID, MovieID, Rating, Timestamp]
        
    Returns:
        pd.DataFrame: User-item matrix with users as rows and movies as columns
    """
    print("\nCreating user-item matrix...")
    
    # Create pivot table
    user_item_matrix = ratings.pivot_table(
        index='UserID',
        columns='MovieID',
        values='Rating',
        fill_value=0
    )
    
    print(f"Matrix shape: {user_item_matrix.shape}")
    print(f"Users (rows): {user_item_matrix.shape[0]}")
    print(f"Movies (columns): {user_item_matrix.shape[1]}")
    print(f"Sparsity: {(user_item_matrix == 0).sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1]):.4f}")
    
    return user_item_matrix


def split_train_test(ratings, test_size=0.2, random_state=42):
    """
    Split ratings data into train and test sets.
    
    Args:
        ratings (pd.DataFrame): DataFrame with columns [UserID, MovieID, Rating, Timestamp]
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_ratings, test_ratings)
    """
    print(f"\nSplitting data into train/test sets (test_size={test_size}, random_state={random_state})...")
    
    train_ratings, test_ratings = train_test_split(
        ratings,
        test_size=test_size,
        random_state=random_state,
        stratify=None
    )
    
    print(f"Train set: {len(train_ratings)} ratings")
    print(f"Test set: {len(test_ratings)} ratings")
    
    return train_ratings, test_ratings


def normalize_matrix(matrix, method='mean_centering'):
    """
    Normalize the user-item matrix.
    
    Args:
        matrix (pd.DataFrame or np.ndarray): User-item matrix
        method (str): Normalization method ('mean_centering', 'min_max', or 'z_score')
        
    Returns:
        tuple: (normalized_matrix, normalization_params)
    """
    print(f"\nNormalizing matrix using {method}...")
    
    if isinstance(matrix, pd.DataFrame):
        matrix_values = matrix.values
        is_dataframe = True
        index = matrix.index
        columns = matrix.columns
    else:
        matrix_values = matrix
        is_dataframe = False
    
    # Create a copy to avoid modifying original
    normalized = matrix_values.copy()
    
    # Get mask of non-zero entries (actual ratings)
    mask = normalized != 0
    
    if method == 'mean_centering':
        # Compute mean rating per user (only for rated items)
        user_means = np.zeros(normalized.shape[0])
        for i in range(normalized.shape[0]):
            rated_items = normalized[i, :][mask[i, :]]
            if len(rated_items) > 0:
                user_means[i] = rated_items.mean()
        
        # Center the ratings
        for i in range(normalized.shape[0]):
            normalized[i, mask[i, :]] -= user_means[i]
        
        params = {'method': method, 'user_means': user_means}
        
    elif method == 'min_max':
        # Min-max normalization to [0, 1]
        min_rating = normalized[mask].min()
        max_rating = normalized[mask].max()
        
        normalized[mask] = (normalized[mask] - min_rating) / (max_rating - min_rating)
        
        params = {'method': method, 'min': min_rating, 'max': max_rating}
        
    elif method == 'z_score':
        # Z-score normalization
        mean_rating = normalized[mask].mean()
        std_rating = normalized[mask].std()
        
        normalized[mask] = (normalized[mask] - mean_rating) / std_rating
        
        params = {'method': method, 'mean': mean_rating, 'std': std_rating}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if is_dataframe:
        normalized = pd.DataFrame(normalized, index=index, columns=columns)
    
    print(f"Normalization complete.")
    
    return normalized, params


def denormalize_predictions(predictions, normalization_params):
    """
    Denormalize predictions back to original rating scale.
    
    Args:
        predictions (np.ndarray): Normalized predictions
        normalization_params (dict): Parameters used for normalization
        
    Returns:
        np.ndarray: Denormalized predictions
    """
    method = normalization_params['method']
    denormalized = predictions.copy()
    
    if method == 'mean_centering':
        user_means = normalization_params['user_means']
        for i in range(denormalized.shape[0]):
            denormalized[i, :] += user_means[i]
            
    elif method == 'min_max':
        min_rating = normalization_params['min']
        max_rating = normalization_params['max']
        denormalized = denormalized * (max_rating - min_rating) + min_rating
        
    elif method == 'z_score':
        mean_rating = normalization_params['mean']
        std_rating = normalization_params['std']
        denormalized = denormalized * std_rating + mean_rating
    
    # Clip to valid rating range (typically 1-5 for MovieLens)
    denormalized = np.clip(denormalized, 1, 5)
    
    return denormalized


def save_matrix(matrix, filepath):
    """
    Save user-item matrix to CSV file.
    
    Args:
        matrix (pd.DataFrame): Matrix to save
        filepath (str): Path to save the file
    """
    print(f"\nSaving matrix to {filepath}...")
    matrix.to_csv(filepath)
    print(f"Matrix saved successfully.")


def load_matrix(filepath):
    """
    Load user-item matrix from CSV file.
    
    Args:
        filepath (str): Path to the file
        
    Returns:
        pd.DataFrame: Loaded matrix
    """
    print(f"\nLoading matrix from {filepath}...")
    matrix = pd.read_csv(filepath, index_col=0)
    print(f"Matrix loaded. Shape: {matrix.shape}")
    return matrix
