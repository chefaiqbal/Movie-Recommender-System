"""
Generate Model Comparison Visualizations

Creates all required visualization plots for the project:
- predicted_vs_actual.png: Scatter plot comparing SVD and PMF predictions
- rmse_comparison.png: Bar chart comparing model performance
- user_comparison.png: SVD vs PMF predictions for a specific user
- top_recommendations.png: Histogram of top recommended movies
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


def load_data():
    """Load test data and model predictions."""
    print("[Step 1] Loading data and predictions...")
    
    # Load test ratings
    test_df = pd.read_csv("data/processed/test_ratings.csv")
    
    # Load predictions
    svd_predictions = np.load("reports/svd_predictions.npy")
    pmf_predictions = np.load("reports/pmf_predictions.npy")
    
    # Load user/movie ID mappings
    user_ids = np.load("reports/svd_model/user_ids.npy")
    item_ids = np.load("reports/svd_model/item_ids.npy")
    
    # Create mapping dictionaries
    user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
    
    print(f"  ✓ Loaded {len(test_df):,} test ratings")
    print(f"  ✓ SVD predictions: {svd_predictions.shape}")
    print(f"  ✓ PMF predictions: {pmf_predictions.shape}")
    
    return test_df, svd_predictions, pmf_predictions, user_id_to_idx, item_id_to_idx


def extract_predictions(test_df, svd_pred, pmf_pred, user_map, item_map):
    """Extract predicted ratings for test set."""
    print("\n[Step 2] Extracting predictions for test set...")
    
    actual_ratings = []
    svd_ratings = []
    pmf_ratings = []
    
    for _, row in test_df.iterrows():
        user_id = int(row['UserID'])
        movie_id = int(row['MovieID'])
        actual = row['Rating']
        
        if user_id in user_map and movie_id in item_map:
            user_idx = user_map[user_id]
            movie_idx = item_map[movie_id]
            
            actual_ratings.append(actual)
            svd_ratings.append(svd_pred[user_idx, movie_idx])
            pmf_ratings.append(pmf_pred[user_idx, movie_idx])
    
    print(f"  ✓ Extracted {len(actual_ratings):,} predictions")
    
    return np.array(actual_ratings), np.array(svd_ratings), np.array(pmf_ratings)


def plot_predicted_vs_actual(actual, svd_pred, pmf_pred, save_path):
    """
    Create scatter plot comparing predicted vs actual ratings.
    
    Shows both SVD and PMF predictions on the same plot.
    """
    print("\n[Step 3] Creating predicted vs actual scatter plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sample data for visualization (to avoid overcrowding)
    sample_size = min(5000, len(actual))
    sample_idx = np.random.choice(len(actual), sample_size, replace=False)
    
    # SVD plot
    ax1 = axes[0]
    ax1.scatter(actual[sample_idx], svd_pred[sample_idx], 
                alpha=0.3, s=10, c='#3498db', edgecolors='none')
    ax1.plot([1, 5], [1, 5], 'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Rating', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Rating', fontsize=12, fontweight='bold')
    ax1.set_title('SVD Model: Predicted vs Actual Ratings', fontsize=14, fontweight='bold')
    ax1.set_xlim(0.5, 5.5)
    ax1.set_ylim(0.5, 5.5)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Calculate and display RMSE
    svd_rmse = np.sqrt(mean_squared_error(actual, svd_pred))
    ax1.text(0.05, 0.95, f'RMSE = {svd_rmse:.4f}', 
             transform=ax1.transAxes, fontsize=11, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # PMF plot
    ax2 = axes[1]
    ax2.scatter(actual[sample_idx], pmf_pred[sample_idx], 
                alpha=0.3, s=10, c='#2ecc71', edgecolors='none')
    ax2.plot([1, 5], [1, 5], 'r--', linewidth=2, label='Perfect Prediction')
    ax2.set_xlabel('Actual Rating', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Predicted Rating', fontsize=12, fontweight='bold')
    ax2.set_title('PMF Model: Predicted vs Actual Ratings', fontsize=14, fontweight='bold')
    ax2.set_xlim(0.5, 5.5)
    ax2.set_ylim(0.5, 5.5)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Calculate and display RMSE
    pmf_rmse = np.sqrt(mean_squared_error(actual, pmf_pred))
    ax2.text(0.05, 0.95, f'RMSE = {pmf_rmse:.4f}', 
             transform=ax2.transAxes, fontsize=11, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {save_path}")
    plt.close()


def plot_rmse_comparison(actual, svd_pred, pmf_pred, save_path):
    """
    Create bar chart comparing RMSE of SVD and PMF models.
    """
    print("\n[Step 4] Creating RMSE comparison bar chart...")
    
    # Calculate RMSE for both models
    svd_rmse = np.sqrt(mean_squared_error(actual, svd_pred))
    pmf_rmse = np.sqrt(mean_squared_error(actual, pmf_pred))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['SVD', 'PMF with Biases']
    rmse_values = [svd_rmse, pmf_rmse]
    colors = ['#3498db', '#2ecc71']
    
    bars = ax.bar(models, rmse_values, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=2, width=0.5)
    
    # Add target lines
    ax.axhline(y=0.90, color='orange', linestyle='--', linewidth=2, 
               label='SVD Target (≤ 0.90)', zorder=0)
    ax.axhline(y=0.85, color='red', linestyle='--', linewidth=2, 
               label='PMF Target (≤ 0.85)', zorder=0)
    
    # Add value labels on bars
    for bar, value in zip(bars, rmse_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}\n({value:.2f})',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    # Add improvement percentage
    improvement = ((svd_rmse - pmf_rmse) / svd_rmse) * 100
    ax.text(0.5, max(rmse_values) * 0.5, 
            f'PMF Improvement: {improvement:.2f}%',
            ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    ax.set_ylabel('RMSE (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0.7, max(rmse_values) * 1.15)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add pass/fail indicators
    svd_status = '✓ PASS' if svd_rmse <= 0.90 else '✗ FAIL'
    pmf_status = '✓ PASS' if pmf_rmse <= 0.85 else '✗ CLOSE'
    
    ax.text(0, -0.12, svd_status, transform=ax.transData, 
            ha='center', fontsize=11, fontweight='bold',
            color='green' if svd_rmse <= 0.90 else 'red')
    ax.text(1, -0.12, pmf_status, transform=ax.transData, 
            ha='center', fontsize=11, fontweight='bold',
            color='green' if pmf_rmse <= 0.85 else 'orange')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {save_path}")
    plt.close()


def plot_user_comparison(user_id, svd_pred, pmf_pred, user_map, item_map, save_path):
    """
    Compare SVD vs PMF predictions for a specific user.
    
    Shows top predictions from each model side-by-side.
    """
    print(f"\n[Step 5] Creating user comparison plot (User {user_id})...")
    
    # Get user index
    if user_id not in user_map:
        print(f"  ✗ User {user_id} not found, using User 1")
        user_id = 1
    
    user_idx = user_map[user_id]
    
    # Get predictions for this user
    svd_user_pred = svd_pred[user_idx, :]
    pmf_user_pred = pmf_pred[user_idx, :]
    
    # Load movie titles
    movies_df = pd.read_csv("data/processed/movies.csv")
    item_ids = np.load("reports/svd_model/item_ids.npy")
    
    # Create movie lookup
    movie_lookup = {row['MovieID']: row['Title'] for _, row in movies_df.iterrows()}
    
    # Get top 15 movies from each model
    svd_top_idx = np.argsort(svd_user_pred)[-15:][::-1]
    pmf_top_idx = np.argsort(pmf_user_pred)[-15:][::-1]
    
    svd_top_movies = [(item_ids[i], svd_user_pred[i]) for i in svd_top_idx]
    pmf_top_movies = [(item_ids[i], pmf_user_pred[i]) for i in pmf_top_idx]
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # SVD recommendations
    ax1 = axes[0]
    svd_titles = [movie_lookup.get(mid, f"Movie {mid}")[:40] for mid, _ in svd_top_movies]
    svd_ratings = [rating for _, rating in svd_top_movies]
    
    y_pos = np.arange(len(svd_titles))
    ax1.barh(y_pos, svd_ratings, color='#3498db', alpha=0.8, edgecolor='black')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(svd_titles, fontsize=9)
    ax1.set_xlabel('Predicted Rating', fontsize=11, fontweight='bold')
    ax1.set_title(f'SVD Top 15 Recommendations\nUser {user_id}', 
                  fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 5.5)
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # PMF recommendations
    ax2 = axes[1]
    pmf_titles = [movie_lookup.get(mid, f"Movie {mid}")[:40] for mid, _ in pmf_top_movies]
    pmf_ratings = [rating for _, rating in pmf_top_movies]
    
    ax2.barh(y_pos, pmf_ratings, color='#2ecc71', alpha=0.8, edgecolor='black')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(pmf_titles, fontsize=9)
    ax2.set_xlabel('Predicted Rating', fontsize=11, fontweight='bold')
    ax2.set_title(f'PMF Top 15 Recommendations\nUser {user_id}', 
                  fontsize=13, fontweight='bold')
    ax2.set_xlim(0, 5.5)
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {save_path}")
    plt.close()


def plot_top_recommendations(svd_pred, pmf_pred, save_path):
    """
    Create histogram showing distribution of top recommended movies.
    
    Shows which movies are most frequently recommended by each model.
    """
    print("\n[Step 6] Creating top recommendations histogram...")
    
    # Get top movie for each user from each model
    svd_top_movies = np.argmax(svd_pred, axis=1)
    pmf_top_movies = np.argmax(pmf_pred, axis=1)
    
    # Count frequency
    from collections import Counter
    svd_counts = Counter(svd_top_movies)
    pmf_counts = Counter(pmf_top_movies)
    
    # Get top 15 most recommended movies
    svd_top_15 = svd_counts.most_common(15)
    pmf_top_15 = pmf_counts.most_common(15)
    
    # Load movie titles
    movies_df = pd.read_csv("data/processed/movies.csv")
    item_ids = np.load("reports/svd_model/item_ids.npy")
    movie_lookup = {row['MovieID']: row['Title'] for _, row in movies_df.iterrows()}
    
    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # SVD histogram
    ax1 = axes[0]
    svd_movie_names = [movie_lookup.get(item_ids[idx], f"Movie {item_ids[idx]}")[:35] 
                       for idx, _ in svd_top_15]
    svd_frequencies = [count for _, count in svd_top_15]
    
    x_pos = np.arange(len(svd_movie_names))
    ax1.bar(x_pos, svd_frequencies, color='#3498db', alpha=0.8, edgecolor='black')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(svd_movie_names, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Number of Users', fontsize=11, fontweight='bold')
    ax1.set_title('SVD: Top 15 Most Recommended Movies (Across All Users)', 
                  fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # PMF histogram
    ax2 = axes[1]
    pmf_movie_names = [movie_lookup.get(item_ids[idx], f"Movie {item_ids[idx]}")[:35] 
                       for idx, _ in pmf_top_15]
    pmf_frequencies = [count for _, count in pmf_top_15]
    
    x_pos = np.arange(len(pmf_movie_names))
    ax2.bar(x_pos, pmf_frequencies, color='#2ecc71', alpha=0.8, edgecolor='black')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(pmf_movie_names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Number of Users', fontsize=11, fontweight='bold')
    ax2.set_title('PMF: Top 15 Most Recommended Movies (Across All Users)', 
                  fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {save_path}")
    plt.close()


def main():
    """Generate all visualization plots."""
    print("="*60)
    print("Generating Model Comparison Visualizations")
    print("="*60)
    print()
    
    # Load data
    test_df, svd_pred, pmf_pred, user_map, item_map = load_data()
    
    # Extract predictions
    actual, svd_ratings, pmf_ratings = extract_predictions(
        test_df, svd_pred, pmf_pred, user_map, item_map
    )
    
    # Create output directory
    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    plot_predicted_vs_actual(
        actual, svd_ratings, pmf_ratings,
        output_dir / "predicted_vs_actual.png"
    )
    
    plot_rmse_comparison(
        actual, svd_ratings, pmf_ratings,
        output_dir / "rmse_comparison.png"
    )
    
    plot_user_comparison(
        user_id=42,  # Sample user
        svd_pred=svd_pred,
        pmf_pred=pmf_pred,
        user_map=user_map,
        item_map=item_map,
        save_path=output_dir / "user_comparison.png"
    )
    
    plot_top_recommendations(
        svd_pred, pmf_pred,
        output_dir / "top_recommendations.png"
    )
    
    print()
    print("="*60)
    print("All Visualizations Generated Successfully!")
    print("="*60)
    print()
    print("Generated files:")
    print("  ✓ reports/predicted_vs_actual.png")
    print("  ✓ reports/rmse_comparison.png")
    print("  ✓ reports/user_comparison.png")
    print("  ✓ reports/top_recommendations.png")
    print()


if __name__ == "__main__":
    main()
