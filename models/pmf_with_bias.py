"""
Probabilistic Matrix Factorization with Bias Terms

Enhanced PMF model that includes global mean, user bias, and item bias
similar to the SVD approach, which should improve performance.

Now with demographic and genre-based feature enhancement for better RMSE.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


class PMFWithBias:
    """
    Probabilistic Matrix Factorization with explicit bias terms.
    
    Prediction: r_ui = μ + b_u + b_i + U[u] @ V[i]
    where:
        μ = global mean
        b_u = user bias
        b_i = item bias
        U[u] @ V[i] = latent factor interaction
    """
    
    def __init__(
        self,
        n_factors=50,
        learning_rate=0.005,
        regularization=0.02,
        n_epochs=100,
        early_stopping=True,
        patience=5,
        random_state=None,
        use_demographic_bias=True,
        demographic_weight=0.15
    ):
        """
        Initialize PMF with bias model.
        
        Args:
            n_factors (int): Number of latent factors
            learning_rate (float): Learning rate for SGD
            regularization (float): L2 regularization strength
            n_epochs (int): Maximum number of training epochs
            early_stopping (bool): Whether to use early stopping
            patience (int): Epochs to wait before early stopping
            random_state (int): Random seed for reproducibility
            use_demographic_bias (bool): Whether to use demographic+genre features
            demographic_weight (float): Weight for demographic bias contribution
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.random_state = random_state
        self.use_demographic_bias = use_demographic_bias
        self.demographic_weight = demographic_weight
        
        # Model parameters (initialized in fit)
        self.global_mean = None
        self.user_bias = None
        self.item_bias = None
        self.U = None  # User latent factors
        self.V = None  # Item latent factors
        
        # Demographic features
        self.demographic_biases = None
        self.user_id_to_idx = None
        self.movie_id_to_idx = None
        
        # Training history
        self.train_mse_history = []
        self.test_mse_history = []
    
    def fit(self, R_train, R_test=None, users_df=None, movies_df=None, 
            ratings_train_df=None, verbose=True):
        """
        Train the PMF model with bias terms using SGD.
        
        Args:
            R_train (np.ndarray): Training user-item matrix (n_users x n_items)
            R_test (np.ndarray): Test user-item matrix (optional, for early stopping)
            users_df (pd.DataFrame): User demographics (for demographic bias)
            movies_df (pd.DataFrame): Movie genres (for demographic bias)
            ratings_train_df (pd.DataFrame): Training ratings DataFrame (for mapping)
            verbose (bool): Whether to print training progress
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_users, n_items = R_train.shape
        
        # Calculate global mean from observed ratings
        observed_mask = ~np.isnan(R_train)
        self.global_mean = np.nanmean(R_train)
        
        # Compute demographic biases if enabled and data provided
        if self.use_demographic_bias and users_df is not None and movies_df is not None and ratings_train_df is not None:
            print("\n[Feature Engineering] Computing demographic-genre biases...")
            from utils.feature_engineering import (
                compute_demographic_genre_affinity,
                compute_demographic_bias_for_ratings
            )
            
            # Compute demographic-genre affinity
            affinity_dict = compute_demographic_genre_affinity(
                ratings_train_df, users_df, movies_df
            )
            
            # Compute bias for each training rating
            ratings_with_bias = compute_demographic_bias_for_ratings(
                ratings_train_df, users_df, movies_df, affinity_dict
            )
            
            # Create demographic bias matrix
            self.demographic_biases = np.zeros((n_users, n_items))
            
            # Map ratings to matrix positions
            if hasattr(self, 'user_id_to_idx') and self.user_id_to_idx is not None:
                for _, row in ratings_with_bias.iterrows():
                    user_idx = self.user_id_to_idx.get(int(row['UserID']))
                    movie_idx = self.movie_id_to_idx.get(int(row['MovieID']))
                    if user_idx is not None and movie_idx is not None:
                        self.demographic_biases[user_idx, movie_idx] = row['DemographicBias']
            
            print(f"Demographic biases computed successfully")
        
        # Initialize bias terms
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        
        # Initialize latent factors with small random values
        self.U = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.V = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        # Get coordinates of observed ratings in training set
        train_users, train_items = np.where(observed_mask)
        train_ratings = R_train[observed_mask]
        
        # Early stopping setup
        best_test_mse = float('inf')
        patience_counter = 0
        best_params = None
        
        # Training loop
        for epoch in range(self.n_epochs):
            # Shuffle training samples
            indices = np.random.permutation(len(train_users))
            
            # SGD updates
            for idx in indices:
                u = train_users[idx]
                i = train_items[idx]
                r_ui = train_ratings[idx]
                
                # Compute prediction
                pred = self.global_mean + self.user_bias[u] + self.item_bias[i]
                pred += np.dot(self.U[u], self.V[i])
                
                # Add demographic bias if available
                if self.use_demographic_bias and self.demographic_biases is not None:
                    pred += self.demographic_weight * self.demographic_biases[u, i]
                
                # Clip prediction to valid range [1, 5]
                pred = np.clip(pred, 1.0, 5.0)
                
                # Compute error
                error = r_ui - pred
                
                # Update biases
                self.user_bias[u] += self.learning_rate * (error - self.regularization * self.user_bias[u])
                self.item_bias[i] += self.learning_rate * (error - self.regularization * self.item_bias[i])
                
                # Update latent factors
                U_u = self.U[u].copy()
                self.U[u] += self.learning_rate * (error * self.V[i] - self.regularization * self.U[u])
                self.V[i] += self.learning_rate * (error * U_u - self.regularization * self.V[i])
            
            # Compute training MSE
            train_mse = self._compute_mse(R_train)
            self.train_mse_history.append(train_mse)
            
            # Compute test MSE if provided
            if R_test is not None:
                test_mse = self._compute_mse(R_test)
                self.test_mse_history.append(test_mse)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1:3d}: Train MSE = {train_mse:.4f}, Test MSE = {test_mse:.4f}")
                
                # Early stopping check
                if self.early_stopping:
                    if test_mse < best_test_mse:
                        best_test_mse = test_mse
                        patience_counter = 0
                        # Save best parameters
                        best_params = {
                            'U': self.U.copy(),
                            'V': self.V.copy(),
                            'user_bias': self.user_bias.copy(),
                            'item_bias': self.item_bias.copy()
                        }
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            if verbose:
                                print(f"\nEarly stopping at epoch {epoch + 1}")
                                print(f"Best test MSE: {best_test_mse:.4f}")
                            # Restore best parameters
                            if best_params is not None:
                                self.U = best_params['U']
                                self.V = best_params['V']
                                self.user_bias = best_params['user_bias']
                                self.item_bias = best_params['item_bias']
                            break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1:3d}: Train MSE = {train_mse:.4f}")
    
    def predict(self):
        """
        Generate predictions for all user-item pairs.
        
        Returns:
            np.ndarray: Predicted ratings matrix (n_users x n_items)
        """
        # Compute: μ + b_u + b_i + U @ V^T
        predictions = self.global_mean + self.user_bias[:, np.newaxis] + self.item_bias[np.newaxis, :]
        predictions += np.dot(self.U, self.V.T)
        
        # Add demographic bias if available
        if self.use_demographic_bias and self.demographic_biases is not None:
            predictions += self.demographic_weight * self.demographic_biases
        
        # Clip to valid rating range [1, 5]
        predictions = np.clip(predictions, 1.0, 5.0)
        
        return predictions
    
    def evaluate(self, R_test):
        """
        Evaluate model on test set.
        
        Args:
            R_test (np.ndarray): Test user-item matrix
            
        Returns:
            dict: Dictionary containing RMSE and MSE metrics
        """
        predictions = self.predict()
        
        # Get observed test ratings
        test_mask = ~np.isnan(R_test)
        actual = R_test[test_mask]
        pred = predictions[test_mask]
        
        # Calculate metrics
        mse = mean_squared_error(actual, pred)
        rmse = np.sqrt(mse)
        
        return {
            'RMSE': rmse,
            'MSE': mse
        }
    
    def _compute_mse(self, R):
        """
        Compute MSE on observed ratings in matrix R.
        
        Args:
            R (np.ndarray): User-item rating matrix
            
        Returns:
            float: Mean squared error
        """
        predictions = self.predict()
        mask = ~np.isnan(R)
        actual = R[mask]
        pred = predictions[mask]
        
        return mean_squared_error(actual, pred)
