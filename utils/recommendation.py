"""
Recommendation System Module

Generates movie recommendations using trained SVD and PMF models.
Provides utilities to get top-N recommendations for any user.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional


class RecommendationSystem:
    """
    Movie recommendation system using SVD and PMF models.
    
    Attributes:
        svd_predictions (np.ndarray): SVD predicted ratings matrix
        pmf_predictions (np.ndarray): PMF predicted ratings matrix
        user_ids (np.ndarray): User ID mapping
        item_ids (np.ndarray): Movie ID mapping
        movies_df (pd.DataFrame): Movie metadata (titles, genres)
        train_ratings (pd.DataFrame): Training ratings for filtering watched movies
    """
    
    def __init__(self, data_dir: str = "data/processed", models_dir: str = "reports"):
        """
        Initialize recommendation system by loading models and data.
        
        Args:
            data_dir (str): Directory containing processed data
            models_dir (str): Directory containing saved models and predictions
        """
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        
        print("Loading recommendation system...")
        self._load_models()
        self._load_data()
        print("✓ Recommendation system ready!")
    
    def _load_models(self):
        """Load trained model predictions."""
        # Load SVD predictions
        svd_path = self.models_dir / "svd_predictions.npy"
        if not svd_path.exists():
            raise FileNotFoundError(f"SVD predictions not found at {svd_path}")
        self.svd_predictions = np.load(svd_path)
        
        # Load PMF predictions
        pmf_path = self.models_dir / "pmf_predictions.npy"
        if not pmf_path.exists():
            raise FileNotFoundError(f"PMF predictions not found at {pmf_path}")
        self.pmf_predictions = np.load(pmf_path)
        
        # Load user and item IDs
        svd_model_dir = self.models_dir / "svd_model"
        self.user_ids = np.load(svd_model_dir / "user_ids.npy")
        self.item_ids = np.load(svd_model_dir / "item_ids.npy")
        
        print(f"  ✓ Loaded SVD predictions: {self.svd_predictions.shape}")
        print(f"  ✓ Loaded PMF predictions: {self.pmf_predictions.shape}")
    
    def _load_data(self):
        """Load movie metadata and training ratings."""
        # Load movies
        movies_path = self.data_dir / "movies.csv"
        if not movies_path.exists():
            raise FileNotFoundError(f"Movies data not found at {movies_path}")
        self.movies_df = pd.read_csv(movies_path)
        
        # Load training ratings (to filter out already watched movies)
        train_path = self.data_dir / "train_ratings.csv"
        if not train_path.exists():
            raise FileNotFoundError(f"Training ratings not found at {train_path}")
        self.train_ratings = pd.read_csv(train_path)
        
        print(f"  ✓ Loaded {len(self.movies_df)} movies")
        print(f"  ✓ Loaded {len(self.train_ratings):,} training ratings")
    
    def get_user_index(self, user_id: int) -> Optional[int]:
        """
        Get matrix index for a user ID.
        
        Args:
            user_id (int): Original user ID
            
        Returns:
            int: Matrix index, or None if user not found
        """
        idx = np.where(self.user_ids == user_id)[0]
        return int(idx[0]) if len(idx) > 0 else None
    
    def get_movie_index(self, movie_id: int) -> Optional[int]:
        """
        Get matrix index for a movie ID.
        
        Args:
            movie_id (int): Original movie ID
            
        Returns:
            int: Matrix index, or None if movie not found
        """
        idx = np.where(self.item_ids == movie_id)[0]
        return int(idx[0]) if len(idx) > 0 else None
    
    def get_watched_movies(self, user_id: int) -> set:
        """
        Get set of movie IDs that user has already watched.
        
        Args:
            user_id (int): User ID
            
        Returns:
            set: Set of movie IDs user has rated
        """
        user_ratings = self.train_ratings[self.train_ratings['UserID'] == user_id]
        return set(user_ratings['MovieID'].values)
    
    def generate_recommendations(
        self,
        user_id: int,
        model: str = 'pmf',
        top_n: int = 10,
        exclude_watched: bool = True
    ) -> pd.DataFrame:
        """
        Generate top-N movie recommendations for a user.
        
        Args:
            user_id (int): User ID to generate recommendations for
            model (str): Model to use - 'svd' or 'pmf' (default: 'pmf')
            top_n (int): Number of recommendations to return (default: 10)
            exclude_watched (bool): Whether to exclude already watched movies (default: True)
            
        Returns:
            pd.DataFrame: Top-N recommendations with columns:
                - MovieID: Movie ID
                - Title: Movie title
                - Genres: Movie genres
                - PredictedRating: Model's predicted rating
                - Rank: Recommendation rank (1 = best)
        
        Raises:
            ValueError: If user_id not found or model not recognized
        """
        # Validate user
        user_idx = self.get_user_index(user_id)
        if user_idx is None:
            raise ValueError(f"User ID {user_id} not found in the dataset")
        
        # Select model predictions
        if model.lower() == 'svd':
            predictions = self.svd_predictions[user_idx, :]
        elif model.lower() == 'pmf':
            predictions = self.pmf_predictions[user_idx, :]
        else:
            raise ValueError(f"Model must be 'svd' or 'pmf', got '{model}'")
        
        # Create recommendations DataFrame
        recommendations = pd.DataFrame({
            'MovieID': self.item_ids,
            'PredictedRating': predictions
        })
        
        # Exclude watched movies if requested
        if exclude_watched:
            watched = self.get_watched_movies(user_id)
            recommendations = recommendations[~recommendations['MovieID'].isin(watched)]
        
        # Sort by predicted rating (descending)
        recommendations = recommendations.sort_values('PredictedRating', ascending=False)
        
        # Get top-N
        recommendations = recommendations.head(top_n).reset_index(drop=True)
        
        # Add rank
        recommendations['Rank'] = range(1, len(recommendations) + 1)
        
        # Merge with movie metadata
        recommendations = recommendations.merge(
            self.movies_df[['MovieID', 'Title', 'Genres']],
            on='MovieID',
            how='left'
        )
        
        # Reorder columns
        recommendations = recommendations[['Rank', 'MovieID', 'Title', 'Genres', 'PredictedRating']]
        
        return recommendations
    
    def get_top_rated_movies(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        """
        Get user's top-rated movies from training data.
        
        Args:
            user_id (int): User ID
            top_n (int): Number of top movies to return (default: 10)
            
        Returns:
            pd.DataFrame: User's top-rated movies with titles and genres
        """
        # Get user's ratings
        user_ratings = self.train_ratings[self.train_ratings['UserID'] == user_id].copy()
        
        if len(user_ratings) == 0:
            return pd.DataFrame(columns=['Rank', 'MovieID', 'Title', 'Genres', 'Rating'])
        
        # Sort by rating
        user_ratings = user_ratings.sort_values('Rating', ascending=False)
        
        # Get top-N
        user_ratings = user_ratings.head(top_n).reset_index(drop=True)
        
        # Add rank
        user_ratings['Rank'] = range(1, len(user_ratings) + 1)
        
        # Merge with movie metadata
        user_ratings = user_ratings.merge(
            self.movies_df[['MovieID', 'Title', 'Genres']],
            on='MovieID',
            how='left'
        )
        
        # Reorder columns
        user_ratings = user_ratings[['Rank', 'MovieID', 'Title', 'Genres', 'Rating']]
        
        return user_ratings
    
    def compare_models(self, user_id: int, top_n: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Compare recommendations from SVD and PMF models for a user.
        
        Args:
            user_id (int): User ID
            top_n (int): Number of recommendations per model (default: 10)
            
        Returns:
            dict: Dictionary with keys 'svd', 'pmf', and 'top_rated' containing DataFrames
        """
        return {
            'top_rated': self.get_top_rated_movies(user_id, top_n),
            'svd': self.generate_recommendations(user_id, model='svd', top_n=top_n),
            'pmf': self.generate_recommendations(user_id, model='pmf', top_n=top_n)
        }
    
    def save_user_recommendations(
        self,
        user_id: int,
        model: str = 'pmf',
        top_n: int = 10,
        output_dir: str = "reports"
    ) -> Path:
        """
        Generate and save recommendations to CSV file.
        
        Args:
            user_id (int): User ID
            model (str): Model to use - 'svd' or 'pmf'
            top_n (int): Number of recommendations
            output_dir (str): Directory to save recommendations
            
        Returns:
            Path: Path to saved CSV file
        """
        recommendations = self.generate_recommendations(user_id, model, top_n)
        
        output_path = Path(output_dir) / f"user_{user_id}_recommendations.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        recommendations.to_csv(output_path, index=False)
        
        return output_path


def generate_recommendations(
    user_id: int,
    model: str = 'pmf',
    top_n: int = 10
) -> pd.DataFrame:
    """
    Convenience function to generate recommendations.
    
    This is the main interface required by project specifications.
    
    Args:
        user_id (int): User ID to generate recommendations for
        model (str): Model to use - 'svd' or 'pmf' (default: 'pmf')
        top_n (int): Number of recommendations to return (default: 10)
        
    Returns:
        pd.DataFrame: Top-N recommendations
        
    Example:
        >>> recs = generate_recommendations(user_id=42, model='pmf', top_n=10)
        >>> print(recs)
    """
    rec_system = RecommendationSystem()
    return rec_system.generate_recommendations(user_id, model, top_n)


if __name__ == "__main__":
    # Demo usage
    print("="*60)
    print("Recommendation System Demo")
    print("="*60)
    print()
    
    # Initialize system
    rec_system = RecommendationSystem()
    print()
    
    # Test with a sample user
    test_user_id = 1
    print(f"Testing with User {test_user_id}")
    print("-" * 60)
    
    # Get top-rated movies
    print(f"\nUser {test_user_id}'s Top-Rated Movies:")
    top_rated = rec_system.get_top_rated_movies(test_user_id, top_n=5)
    print(top_rated.to_string(index=False))
    
    # Get SVD recommendations
    print(f"\n\nSVD Recommendations for User {test_user_id}:")
    svd_recs = rec_system.generate_recommendations(test_user_id, model='svd', top_n=5)
    print(svd_recs.to_string(index=False))
    
    # Get PMF recommendations
    print(f"\n\nPMF Recommendations for User {test_user_id}:")
    pmf_recs = rec_system.generate_recommendations(test_user_id, model='pmf', top_n=5)
    print(pmf_recs.to_string(index=False))
    
    # Save recommendations
    print("\n\nSaving recommendations...")
    output_path = rec_system.save_user_recommendations(test_user_id, model='pmf', top_n=10)
    print(f"✓ Saved to: {output_path}")
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
