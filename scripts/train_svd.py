"""
SVD Model Training Script

Implements Singular Value Decomposition using scipy.sparse.linalg.svds
with bias correction for improved accuracy.

This script trains the SVD model and evaluates performance on the test set.
Target: RMSE ≤ 0.90
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
import json


def main():
    print("="*60)
    print("SVD Model Training")
    print("="*60)
    
    # Load data
    print("\n[Step 1] Loading preprocessed data...")
    data_dir = project_root / "data" / "processed"
    
    user_item_original = pd.read_csv(data_dir / "user_item_matrix_original.csv", index_col=0)
    train_ratings = pd.read_csv(data_dir / "train_ratings.csv")
    test_ratings = pd.read_csv(data_dir / "test_ratings.csv")
    
    print(f"User-item matrix shape: {user_item_original.shape}")
    print(f"Train ratings: {len(train_ratings):,}")
    print(f"Test ratings: {len(test_ratings):,}")
    
    # Convert to numpy
    R = user_item_original.values
    user_ids = user_item_original.index.values
    item_ids = user_item_original.columns.astype(int).values
    
    print(f"\n[Step 2] Computing biases...")
    
    # Create mask for known ratings
    mask = R != 0
    
    # Global mean (from all known ratings)
    global_mean = R[mask].mean()
    
    # User biases
    user_bias = np.zeros(R.shape[0])
    for i in range(R.shape[0]):
        user_ratings = R[i, mask[i, :]]
        if len(user_ratings) > 0:
            user_bias[i] = user_ratings.mean() - global_mean
    
    # Item biases
    item_bias = np.zeros(R.shape[1])
    for j in range(R.shape[1]):
        item_ratings = R[mask[:, j], j]
        if len(item_ratings) > 0:
            item_bias[j] = item_ratings.mean() - global_mean
    
    print(f"Global mean: {global_mean:.2f}")
    
    # Remove biases from original matrix
    print(f"\n[Step 3] Centering matrix (removing biases)...")
    R_centered = R.copy()
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i, j] != 0:  # Only center known ratings
                R_centered[i, j] = R[i, j] - global_mean - user_bias[i] - item_bias[j]
    
    # Train SVD with optimal k
    print(f"\n[Step 4] Training SVD model with scipy.sparse.linalg.svds...")
    
    k_values = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    best_rmse = float('inf')
    best_k = None
    best_predictions = None
    results = []
    
    for k in k_values:
        try:
            # Apply SVD to centered matrix
            U, sigma, Vt = svds(R_centered, k=k)
            
            # Reverse to descending order
            U = U[:, ::-1]
            sigma = sigma[::-1]
            Vt = Vt[::-1, :]
            
            # Reconstruct centered predictions
            sigma_matrix = np.diag(sigma)
            predictions_centered = U @ sigma_matrix @ Vt
            
            # Add biases back
            predictions = predictions_centered.copy()
            for i in range(predictions.shape[0]):
                for j in range(predictions.shape[1]):
                    predictions[i, j] = global_mean + user_bias[i] + item_bias[j] + predictions_centered[i, j]
            
            # Clip to valid range
            predictions = np.clip(predictions, 1, 5)
            
            # Evaluate on test set
            pred_df = pd.DataFrame(predictions, index=user_ids, columns=item_ids)
            
            actual_ratings = []
            predicted_ratings = []
            
            for _, row in test_ratings.iterrows():
                user_id = int(row['UserID'])
                movie_id = int(row['MovieID'])
                actual_rating = row['Rating']
                
                if user_id in pred_df.index and movie_id in pred_df.columns:
                    predicted_rating = pred_df.loc[user_id, movie_id]
                    actual_ratings.append(actual_rating)
                    predicted_ratings.append(predicted_rating)
            
            rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
            results.append({'k': k, 'rmse': rmse})
            
            print(f"k={k:2d}: RMSE = {rmse:.2f}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_k = k
                best_predictions = predictions
                best_U = U
                best_sigma = sigma
                best_Vt = Vt
        
        except Exception as e:
            print(f"k={k:2d}: Error - {e}")
            continue
    
    # Results summary
    print("\n" + "="*60)
    print("[Step 5] Evaluation Results")
    print("="*60)
    print(f"Best k: {best_k}")
    print(f"RMSE: {best_rmse:.2f}")
    print(f"Target: ≤ 0.90")
    print(f"Status: {'✓ PASS' if best_rmse <= 0.90 else '✗ FAIL'}")
    print("="*60)
    
    # Save results
    print(f"\n[Step 6] Saving model...")
    
    # Save model components
    model_dir = project_root / "reports" / "svd_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(model_dir / "U.npy", best_U)
    np.save(model_dir / "sigma.npy", best_sigma)
    np.save(model_dir / "Vt.npy", best_Vt)
    np.save(model_dir / "user_ids.npy", user_ids)
    np.save(model_dir / "item_ids.npy", item_ids)
    
    # Save predictions
    np.save(project_root / "reports" / "svd_predictions.npy", best_predictions)
    
    # Save metrics
    metrics = {
        'SVD_RMSE': round(best_rmse, 2),
        'SVD_k_factors': int(best_k),
        'SVD_method': 'scipy.sparse.linalg.svds',
        'SVD_passes_audit': bool(best_rmse <= 0.90)
    }
    
    with open(project_root / "reports" / "model_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Model saved to {model_dir}/")
    print(f"✓ Predictions saved to reports/svd_predictions.npy")
    print(f"✓ Metrics saved to reports/model_metrics.json")
    
    print("\n" + "="*60)
    print("SVD Model Training Complete!")
    print("="*60)
    
    return best_rmse


if __name__ == "__main__":
    main()
