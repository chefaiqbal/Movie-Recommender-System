"""
Test script to demonstrate recommendation system usage.

This shows how to use the generate_recommendations() function
as required by the project specification.
"""

from utils.recommendation import generate_recommendations, RecommendationSystem


def test_basic_usage():
    """Test the basic generate_recommendations function."""
    print("="*60)
    print("Test 1: Basic generate_recommendations() function")
    print("="*60)
    print()
    
    # Generate recommendations using the simple interface
    user_id = 42
    recommendations = generate_recommendations(user_id=user_id, model='pmf', top_n=10)
    
    print(f"Top 10 PMF recommendations for User {user_id}:")
    print(recommendations.to_string(index=False))
    print()


def test_model_comparison():
    """Test comparing SVD and PMF recommendations."""
    print("="*60)
    print("Test 2: Compare SVD vs PMF Recommendations")
    print("="*60)
    print()
    
    rec_system = RecommendationSystem()
    user_id = 100
    
    comparison = rec_system.compare_models(user_id, top_n=5)
    
    print(f"User {user_id}'s Top-Rated Movies (from history):")
    print(comparison['top_rated'].to_string(index=False))
    print()
    
    print(f"\nSVD Recommendations:")
    print(comparison['svd'].to_string(index=False))
    print()
    
    print(f"\nPMF Recommendations:")
    print(comparison['pmf'].to_string(index=False))
    print()


def test_save_recommendations():
    """Test saving recommendations to CSV."""
    print("="*60)
    print("Test 3: Save Recommendations to CSV")
    print("="*60)
    print()
    
    rec_system = RecommendationSystem()
    
    user_id = 1234
    output_path = rec_system.save_user_recommendations(
        user_id=user_id,
        model='pmf',
        top_n=10,
        output_dir='reports'
    )
    
    print(f"✓ Saved recommendations for User {user_id}")
    print(f"  File: {output_path}")
    print()


if __name__ == "__main__":
    print("\n")
    print("*"*60)
    print("Recommendation System - Usage Examples")
    print("*"*60)
    print("\n")
    
    # Run tests
    test_basic_usage()
    test_model_comparison()
    test_save_recommendations()
    
    print("*"*60)
    print("All tests completed successfully!")
    print("*"*60)
    print("\n")
