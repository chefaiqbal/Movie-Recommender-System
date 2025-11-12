"""
Probabilistic Matrix Factorization (PMF) Model

This module implements the PMF algorithm using gradient descent with
L2 regularization to learn latent factor representations of users and items.
"""

import numpy as np
from sklearn.metrics import mean_squared_error


class PMFModel:
    """
    Probabilistic Matrix Factorization model for collaborative filtering.
    
    The model learns latent factor matrices U (users) and V (items) by
    minimizing the regularized squared error between observed ratings
    and predicted ratings.
    
    Attributes:
        n_factors (int): Number of latent factors
        learning_rate (float): Learning rate for gradient descent
        regularization (float): L2 regularization parameter
        n_epochs (int): Number of training iterations
        U (np.ndarray): User latent factor matrix (n_users × n_factors)
        V (np.ndarray): Item latent factor matrix (n_items × n_factors)
        train_mse_history (list): MSE on training set per epoch
        test_mse_history (list): MSE on test set per epoch
    """
    
    def __init__(self, n_factors=50, learning_rate=0.005, 
                 regularization=0.02, n_epochs=100, random_state=42,
                 early_stopping=True, patience=5):
        """
        Initialize PMF model with hyperparameters.
        
        Args:
            n_factors (int): Number of latent factors
            learning_rate (float): Learning rate for gradient descent
            regularization (float): L2 regularization strength
            n_epochs (int): Number of training epochs
            random_state (int): Random seed for reproducibility
            early_stopping (bool): Stop training when test error increases
            patience (int): Epochs to wait before early stopping
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.patience = patience
        
        self.U = None  # User latent factors
        self.V = None  # Item latent factors
        
        self.train_mse_history = []
        self.test_mse_history = []
        self.best_U = None
        self.best_V = None
        self.best_epoch = 0
        
        np.random.seed(random_state)
    
    def _initialize_factors(self, n_users, n_items):
        """
        Initialize latent factor matrices with small random values.
        
        Args:
            n_users (int): Number of users
            n_items (int): Number of items
        """
        # Initialize with small random values (Gaussian distribution)
        self.U = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.V = np.random.normal(0, 0.1, (n_items, self.n_factors))
    
    def _get_rating_indices(self, R):
        """
        Get indices of observed ratings in the matrix.
        
        Args:
            R (np.ndarray): User-item rating matrix
            
        Returns:
            tuple: Arrays of user indices and item indices for observed ratings
        """
        return np.where(~np.isnan(R))
    
    def _compute_mse(self, R, user_indices, item_indices):
        """
        Compute Mean Squared Error for observed ratings.
        
        Args:
            R (np.ndarray): True ratings matrix
            user_indices (np.ndarray): User indices for observed ratings
            item_indices (np.ndarray): Item indices for observed ratings
            
        Returns:
            float: Mean squared error
        """
        predictions = np.sum(self.U[user_indices] * self.V[item_indices], axis=1)
        true_ratings = R[user_indices, item_indices]
        return mean_squared_error(true_ratings, predictions)
    
    def fit(self, R_train, R_test=None, verbose=True):
        """
        Train the PMF model using gradient descent.
        
        Args:
            R_train (np.ndarray): Training ratings matrix (NaN for missing)
            R_test (np.ndarray): Test ratings matrix (NaN for missing), optional
            verbose (bool): Print progress during training
            
        Returns:
            self: Trained model
        """
        n_users, n_items = R_train.shape
        self._initialize_factors(n_users, n_items)
        
        # Get indices of observed ratings
        train_user_idx, train_item_idx = self._get_rating_indices(R_train)
        if R_test is not None:
            test_user_idx, test_item_idx = self._get_rating_indices(R_test)
        
        best_test_mse = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.n_epochs):
            # Shuffle training data for SGD
            shuffle_idx = np.random.permutation(len(train_user_idx))
            
            # Gradient descent on each observed rating
            for idx in shuffle_idx:
                u = train_user_idx[idx]
                i = train_item_idx[idx]
                
                # Compute prediction error
                prediction = np.dot(self.U[u, :], self.V[i, :])
                error = R_train[u, i] - prediction
                
                # Update latent factors with gradient descent
                U_gradient = -2 * error * self.V[i, :] + 2 * self.regularization * self.U[u, :]
                V_gradient = -2 * error * self.U[u, :] + 2 * self.regularization * self.V[i, :]
                
                self.U[u, :] -= self.learning_rate * U_gradient
                self.V[i, :] -= self.learning_rate * V_gradient
            
            # Compute MSE for this epoch
            train_mse = self._compute_mse(R_train, train_user_idx, train_item_idx)
            self.train_mse_history.append(train_mse)
            
            if R_test is not None:
                test_mse = self._compute_mse(R_test, test_user_idx, test_item_idx)
                self.test_mse_history.append(test_mse)
                
                # Early stopping logic
                if test_mse < best_test_mse:
                    best_test_mse = test_mse
                    self.best_U = self.U.copy()
                    self.best_V = self.V.copy()
                    self.best_epoch = epoch + 1
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{self.n_epochs} - "
                          f"Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
                
                # Early stopping
                if self.early_stopping and patience_counter >= self.patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                        print(f"Best test MSE: {best_test_mse:.4f} at epoch {self.best_epoch}")
                    # Restore best model
                    self.U = self.best_U
                    self.V = self.best_V
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{self.n_epochs} - Train MSE: {train_mse:.4f}")
        
        # If no early stopping occurred, save final model as best
        if self.best_U is None:
            self.best_U = self.U.copy()
            self.best_V = self.V.copy()
            self.best_epoch = self.n_epochs
        
        return self
    
    def predict(self, user_indices=None, item_indices=None):
        """
        Generate predictions for user-item pairs.
        
        Args:
            user_indices (np.ndarray): User indices (if None, predict all)
            item_indices (np.ndarray): Item indices (if None, predict all)
            
        Returns:
            np.ndarray: Predicted ratings
        """
        if user_indices is None and item_indices is None:
            # Predict full matrix
            return np.dot(self.U, self.V.T)
        else:
            # Predict specific user-item pairs
            return np.sum(self.U[user_indices] * self.V[item_indices], axis=1)
    
    def evaluate(self, R_test):
        """
        Evaluate model on test set.
        
        Args:
            R_test (np.ndarray): Test ratings matrix (NaN for missing)
            
        Returns:
            dict: Dictionary with RMSE metric
        """
        test_user_idx, test_item_idx = self._get_rating_indices(R_test)
        mse = self._compute_mse(R_test, test_user_idx, test_item_idx)
        rmse = np.sqrt(mse)
        
        return {"RMSE": rmse, "MSE": mse}
