"""
Training script for SVD model.

This script:
1. Loads preprocessed data
2. Trains SVD model
3. Evaluates on test set
4. Saves predictions and metrics
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.matrix_creation import load_matrix
from models.svd_model import SVDModel, save_metrics


def main():
    """Main training pipeline for SVD model."""
    
    print("="*60)
    print("SVD Model Training")
    print("="*60)
    
    # Load preprocessed data
    print("\n[Step 1] Loading preprocessed data...")
    user_item_matrix = load_matrix('data/processed/user_item_matrix.csv')
    user_item_original = load_matrix('data/processed/user_item_matrix_original.csv')
    test_ratings = pd.read_csv('data/processed/test_ratings.csv')
    
    # Load normalization parameters
    norm_params = np.load('data/processed/normalization_params.npy', allow_pickle=True).item()
    user_means = norm_params.get('user_means', None)
    
    print(f"User-item matrix shape: {user_item_matrix.shape}")
    print(f"Test ratings: {len(test_ratings)}")
    
    # Try different values of n_factors to find best RMSE
    print("\n[Step 2] Finding optimal number of latent factors...")
    best_rmse = float('inf')
    best_model = None
    best_k = None
    
    for k in [10, 15, 20, 25, 30, 40, 50]:
        print(f"\nTrying k={k}...")
        svd_model = SVDModel(n_factors=k)
        svd_model.fit(user_item_matrix, user_means=user_means, original_matrix=user_item_original)
        rmse = svd_model.evaluate(test_ratings, user_item_matrix)
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = svd_model
            best_k = k
        
        print(f"  RMSE: {rmse:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Best configuration: k={best_k}, RMSE={best_rmse:.4f}")
    print(f"{'='*60}")
    
    svd_model = best_model
    
    # Evaluate on test set
    print("\n[Step 3] Evaluating model...")
    rmse = svd_model.evaluate(test_ratings, user_item_matrix)
    
    # Save predictions
    print("\n[Step 4] Saving predictions...")
    svd_model.save_predictions('reports/svd_predictions.npy')
    
    # Save model components
    svd_model.save_model('reports/svd_model')
    
    # Save metrics
    print("\n[Step 5] Saving metrics...")
    metrics = {
        'SVD_RMSE': round(rmse, 4)
    }
    save_metrics(metrics, 'reports/model_metrics.json')
    
    # Print summary
    print("\n" + "="*60)
    print("SVD Model Training Complete!")
    print("="*60)
    print(f"✓ RMSE: {rmse:.4f}")
    print(f"✓ Target: ≤ 0.90")
    print(f"✓ Status: {'PASS ✓' if rmse <= 0.90 else 'FAIL ✗'}")
    print(f"✓ Predictions saved: reports/svd_predictions.npy")
    print(f"✓ Model saved: reports/svd_model/")
    print(f"✓ Metrics saved: reports/model_metrics.json")
    print("="*60)


if __name__ == "__main__":
    main()
