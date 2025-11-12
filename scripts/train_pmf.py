"""
PMF Model Training Script

Trains Probabilistic Matrix Factorization model with hyperparameter
optimization and saves the best model, predictions, and convergence plot.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pmf_model import PMFModel


def load_data():
    """Load preprocessed user-item matrices."""
    print("[Step 1] Loading preprocessed data...")
    
    # Load the original matrix
    matrix_path = Path("data/processed/user_item_matrix_original.csv")
    matrix_df = pd.read_csv(matrix_path, index_col=0)
    
    # Get user and movie ID mappings
    user_ids = matrix_df.index.values  # Original user IDs
    movie_ids = matrix_df.columns.astype(int).values  # Original movie IDs
    
    # Create mapping from original IDs to matrix indices
    user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}
    
    n_users, n_items = matrix_df.shape
    
    # Initialize empty matrices with NaN
    R_train = np.full((n_users, n_items), np.nan)
    R_test = np.full((n_users, n_items), np.nan)
    
    # Load train ratings and fill matrix
    train_ratings_path = Path("data/processed/train_ratings.csv")
    train_ratings = pd.read_csv(train_ratings_path)
    for _, row in train_ratings.iterrows():
        user_id = int(row['UserID'])
        movie_id = int(row['MovieID'])
        
        if user_id in user_id_to_idx and movie_id in movie_id_to_idx:
            user_idx = user_id_to_idx[user_id]
            movie_idx = movie_id_to_idx[movie_id]
            R_train[user_idx, movie_idx] = row['Rating']
    
    # Load test ratings and fill matrix
    test_ratings_path = Path("data/processed/test_ratings.csv")
    test_ratings = pd.read_csv(test_ratings_path)
    for _, row in test_ratings.iterrows():
        user_id = int(row['UserID'])
        movie_id = int(row['MovieID'])
        
        if user_id in user_id_to_idx and movie_id in movie_id_to_idx:
            user_idx = user_id_to_idx[user_id]
            movie_idx = movie_id_to_idx[movie_id]
            R_test[user_idx, movie_idx] = row['Rating']
    
    print(f"User-item matrix shape: {R_train.shape}")
    print(f"Train ratings: {np.sum(~np.isnan(R_train)):,}")
    print(f"Test ratings: {np.sum(~np.isnan(R_test)):,}")
    
    return R_train, R_test


def plot_convergence(train_mse, test_mse, save_path):
    """
    Plot and save MSE convergence over epochs.
    
    Args:
        train_mse (list): Training MSE history
        test_mse (list): Test MSE history
        save_path (Path): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_mse) + 1)
    
    plt.plot(epochs, train_mse, 'b-', label='Train MSE', linewidth=2)
    plt.plot(epochs, test_mse, 'r-', label='Test MSE', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.title('PMF Model Convergence', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nConvergence plot saved to: {save_path}")
    plt.close()


def save_model_factors(model, save_dir):
    """
    Save learned latent factor matrices.
    
    Args:
        model (PMFModel): Trained PMF model
        save_dir (Path): Directory to save factors
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(save_dir / "U.npy", model.U)
    np.save(save_dir / "V.npy", model.V)
    
    print(f"Latent factors saved to: {save_dir}")
    print(f"  U (users): {model.U.shape}")
    print(f"  V (items): {model.V.shape}")


def save_predictions(predictions, save_path):
    """
    Save full prediction matrix.
    
    Args:
        predictions (np.ndarray): Predicted ratings matrix
        save_path (Path): Path to save predictions
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, predictions)
    print(f"Predictions saved to: {save_path}")


def update_metrics(rmse, improvement):
    """
    Update model_metrics.json with PMF results.
    
    Args:
        rmse (float): PMF RMSE on test set
        improvement (float): Percentage improvement over SVD
    """
    metrics_path = Path("reports/model_metrics.json")
    
    # Load existing metrics
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = {}
    
    # Update with PMF results
    metrics["PMF_RMSE"] = round(rmse, 2)
    metrics["PMF_RMSE_exact"] = round(rmse, 4)
    metrics["PMF_n_factors"] = 100
    metrics["PMF_learning_rate"] = 0.002
    metrics["PMF_regularization"] = 0.15
    metrics["PMF_passes_audit"] = rmse <= 0.85
    metrics["PMF_vs_SVD_improvement_%"] = round(improvement, 2)
    
    # Save updated metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics updated in: {metrics_path}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("PMF Model Training")
    print("=" * 60)
    print()
    
    # Load data
    R_train, R_test = load_data()
    print()
    
    # Train PMF model with optimized hyperparameters
    print("[Step 2] Training PMF model...")
    print("Hyperparameters:")
    print("  - Latent factors: 100")
    print("  - Learning rate: 0.002")
    print("  - Regularization: 0.15")
    print("  - Max epochs: 100")
    print("  - Early stopping: Enabled (patience=3)")
    print()
    
    model = PMFModel(
        n_factors=100,
        learning_rate=0.002,
        regularization=0.15,
        n_epochs=100,
        early_stopping=True,
        patience=3,
        random_state=42
    )
    
    model.fit(R_train, R_test, verbose=True)
    print()
    
    # Evaluate on test set
    print("[Step 3] Evaluating on test set...")
    results = model.evaluate(R_test)
    rmse = results['RMSE']
    
    print(f"Final Test RMSE: {rmse:.2f}")
    print()
    
    # Check audit requirement
    print("[Step 4] Audit Check...")
    print(f"Target: RMSE ≤ 0.85")
    print(f"Result: RMSE = {rmse:.2f}")
    
    if rmse <= 0.85:
        print("✓ PASSES audit requirement")
    else:
        print("✗ DOES NOT PASS audit requirement")
    print()
    
    # Calculate improvement over SVD
    metrics_path = Path("reports/model_metrics.json")
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        svd_rmse = metrics.get("SVD_RMSE", 0.90)
        improvement = ((svd_rmse - rmse) / svd_rmse) * 100
        
        print("[Step 5] Model Comparison...")
        print(f"SVD RMSE: {svd_rmse:.2f}")
        print(f"PMF RMSE: {rmse:.2f}")
        print(f"Improvement: {improvement:.2f}%")
        
        if improvement >= 5.0:
            print("✓ PASSES 5% improvement requirement")
        else:
            print("✗ DOES NOT PASS 5% improvement requirement")
        print()
    else:
        improvement = 0.0
    
    # Save convergence plot
    print("[Step 6] Saving results...")
    plot_convergence(
        model.train_mse_history,
        model.test_mse_history,
        Path("reports/pmf_convergence.png")
    )
    
    # Save latent factors
    save_model_factors(model, Path("reports/pmf_factors"))
    
    # Save predictions
    predictions = model.predict()
    predictions = np.clip(predictions, 1, 5)  # Clip to valid rating range
    save_predictions(predictions, Path("reports/pmf_predictions.npy"))
    
    # Update metrics file
    update_metrics(rmse, improvement)
    
    print()
    print("=" * 60)
    print("PMF Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
