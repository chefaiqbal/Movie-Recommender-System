"""
Singular Value Decomposition (SVD) Model for Collaborative Filtering.

This module implements matrix factorization using SVD to predict user ratings
for movies in the MovieLens dataset.
"""

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
import json
from pathlib import Path


class SVDModel:
    """
    SVD-based collaborative filtering model.
    
    Uses Singular Value Decomposition to factorize the user-item matrix
    into three matrices: U, Sigma, and V^T, where:
    - U: User features matrix
    - Sigma: Diagonal matrix of singular values
    - V^T: Item (movie) features matrix
    
    The predicted ratings are reconstructed as: U @ Sigma @ V^T
    """
    
    def __init__(self, n_factors=50):
        """
        Initialize SVD model.
        
        Args:
            n_factors (int): Number of latent factors (singular values to keep)
        """
        self.n_factors = n_factors
        self.U = None
        self.sigma = None
        self.Vt = None
        self.predictions = None
        self.user_means = None
        
    def fit(self, user_item_matrix, user_means=None, original_matrix=None):
        """
        Train the SVD model on the user-item matrix.
        
        Args:
            user_item_matrix (pd.DataFrame or np.ndarray): User-item rating matrix (normalized)
            user_means (np.ndarray): Mean ratings per user (for denormalization)
            original_matrix (pd.DataFrame or np.ndarray): Original matrix for computing biases
        """
        print(f"\nTraining SVD model with {self.n_factors} latent factors...")
        
        if isinstance(user_item_matrix, pd.DataFrame):
            matrix = user_item_matrix.values
            self.user_ids = user_item_matrix.index
            self.item_ids = user_item_matrix.columns
        else:
            matrix = user_item_matrix
            self.user_ids = None
            self.item_ids = None
        
        # Compute biases from original matrix if provided
        if original_matrix is not None:
            if isinstance(original_matrix, pd.DataFrame):
                orig_matrix = original_matrix.values
            else:
                orig_matrix = original_matrix
            
            # Compute global mean, user biases, and item biases
            mask = orig_matrix != 0
            self.global_mean = orig_matrix[mask].mean()
            
            # User biases
            self.user_bias = np.zeros(orig_matrix.shape[0])
            for i in range(orig_matrix.shape[0]):
                user_ratings = orig_matrix[i, mask[i, :]]
                if len(user_ratings) > 0:
                    self.user_bias[i] = user_ratings.mean() - self.global_mean
            
            # Item biases
            self.item_bias = np.zeros(orig_matrix.shape[1])
            for j in range(orig_matrix.shape[1]):
                item_ratings = orig_matrix[mask[:, j], j]
                if len(item_ratings) > 0:
                    self.item_bias[j] = item_ratings.mean() - self.global_mean
            
            print(f"Global mean: {self.global_mean:.4f}")
            print(f"User bias range: [{self.user_bias.min():.4f}, {self.user_bias.max():.4f}]")
            print(f"Item bias range: [{self.item_bias.min():.4f}, {self.item_bias.max():.4f}]")
        else:
            self.global_mean = None
            self.user_bias = None
            self.item_bias = None
        
        self.user_means = user_means
        
        # Perform SVD
        # svds returns singular values in ascending order, so we reverse them
        U, sigma, Vt = svds(matrix, k=self.n_factors)
        
        # Reverse to get descending order (largest singular values first)
        self.U = U[:, ::-1]
        self.sigma = sigma[::-1]
        self.Vt = Vt[::-1, :]
        
        print(f"SVD decomposition complete.")
        print(f"U shape: {self.U.shape}")
        print(f"Sigma shape: {self.sigma.shape}")
        print(f"Vt shape: {self.Vt.shape}")
        
        # Reconstruct the full prediction matrix
        sigma_matrix = np.diag(self.sigma)
        self.predictions = self.U @ sigma_matrix @ self.Vt
        
        print(f"Predictions matrix shape: {self.predictions.shape}")
        
    def predict(self, user_idx=None, item_idx=None):
        """
        Get predictions for specific users/items or the full matrix.
        
        Args:
            user_idx (int or array-like, optional): User index/indices
            item_idx (int or array-like, optional): Item index/indices
            
        Returns:
            np.ndarray: Predicted ratings
        """
        if self.predictions is None:
            raise ValueError("Model must be trained before making predictions")
        
        if user_idx is None and item_idx is None:
            return self.predictions
        elif user_idx is not None and item_idx is not None:
            return self.predictions[user_idx, item_idx]
        elif user_idx is not None:
            return self.predictions[user_idx, :]
        else:
            return self.predictions[:, item_idx]
    
    def denormalize_predictions(self):
        """
        Denormalize predictions by adding back biases or user means.
        
        Returns:
            np.ndarray: Denormalized predictions
        """
        denormalized = self.predictions.copy()
        
        # If we have bias terms, use them
        if self.global_mean is not None and self.user_bias is not None and self.item_bias is not None:
            for i in range(denormalized.shape[0]):
                for j in range(denormalized.shape[1]):
                    denormalized[i, j] = (self.global_mean + 
                                         self.user_bias[i] + 
                                         self.item_bias[j] + 
                                         denormalized[i, j])
        # Otherwise use user means if available
        elif self.user_means is not None:
            for i in range(denormalized.shape[0]):
                denormalized[i, :] += self.user_means[i]
        
        # Clip to valid rating range (1-5)
        denormalized = np.clip(denormalized, 1, 5)
        
        return denormalized
    
    def evaluate(self, test_data, user_item_matrix):
        """
        Evaluate model on test data using RMSE.
        
        Args:
            test_data (pd.DataFrame): Test ratings with columns [UserID, MovieID, Rating]
            user_item_matrix (pd.DataFrame): User-item matrix (for indexing)
            
        Returns:
            float: RMSE on test set
        """
        print("\nEvaluating SVD model on test set...")
        
        # Get denormalized predictions
        predictions_denorm = self.denormalize_predictions()
        
        # Create a DataFrame for easier lookup
        if self.user_ids is not None and self.item_ids is not None:
            pred_df = pd.DataFrame(
                predictions_denorm,
                index=self.user_ids,
                columns=self.item_ids
            )
        else:
            pred_df = pd.DataFrame(
                predictions_denorm,
                index=user_item_matrix.index,
                columns=user_item_matrix.columns
            )
        
        # Convert column names to int if they're strings
        if pred_df.columns.dtype == 'object':
            pred_df.columns = pred_df.columns.astype(int)
        
        # Get predictions for test set
        actual_ratings = []
        predicted_ratings = []
        
        for _, row in test_data.iterrows():
            user_id = int(row['UserID'])
            movie_id = int(row['MovieID'])
            actual_rating = row['Rating']
            
            # Check if user and movie exist in our matrix
            if user_id in pred_df.index and movie_id in pred_df.columns:
                predicted_rating = pred_df.loc[user_id, movie_id]
                actual_ratings.append(actual_rating)
                predicted_ratings.append(predicted_rating)
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
        
        print(f"Test set size: {len(test_data)}")
        print(f"Evaluated on: {len(actual_ratings)} ratings")
        print(f"RMSE: {rmse:.4f}")
        
        return rmse
    
    def save_predictions(self, filepath):
        """
        Save the full prediction matrix.
        
        Args:
            filepath (str): Path to save the predictions
        """
        predictions_denorm = self.denormalize_predictions()
        np.save(filepath, predictions_denorm)
        print(f"\nPredictions saved to {filepath}")
    
    def save_model(self, directory):
        """
        Save the SVD model components.
        
        Args:
            directory (str): Directory to save model components
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        np.save(f"{directory}/U.npy", self.U)
        np.save(f"{directory}/sigma.npy", self.sigma)
        np.save(f"{directory}/Vt.npy", self.Vt)
        
        if self.user_ids is not None:
            np.save(f"{directory}/user_ids.npy", self.user_ids)
        if self.item_ids is not None:
            np.save(f"{directory}/item_ids.npy", self.item_ids)
        
        print(f"\nModel components saved to {directory}/")
    
    def load_model(self, directory):
        """
        Load the SVD model components.
        
        Args:
            directory (str): Directory containing model components
        """
        self.U = np.load(f"{directory}/U.npy")
        self.sigma = np.load(f"{directory}/sigma.npy")
        self.Vt = np.load(f"{directory}/Vt.npy")
        
        if Path(f"{directory}/user_ids.npy").exists():
            self.user_ids = np.load(f"{directory}/user_ids.npy", allow_pickle=True)
        if Path(f"{directory}/item_ids.npy").exists():
            self.item_ids = np.load(f"{directory}/item_ids.npy", allow_pickle=True)
        
        # Reconstruct predictions
        sigma_matrix = np.diag(self.sigma)
        self.predictions = self.U @ sigma_matrix @ self.Vt
        
        print(f"\nModel loaded from {directory}/")


def save_metrics(metrics, filepath='reports/model_metrics.json'):
    """
    Save or update model metrics to JSON file.
    
    Args:
        metrics (dict): Dictionary of metrics to save
        filepath (str): Path to metrics file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing metrics if file exists
    if Path(filepath).exists():
        with open(filepath, 'r') as f:
            existing_metrics = json.load(f)
        existing_metrics.update(metrics)
        metrics = existing_metrics
    
    # Save metrics
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to {filepath}")
