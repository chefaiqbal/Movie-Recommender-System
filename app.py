"""
Movie Recommender System - Streamlit Dashboard

Interactive dashboard for exploring movie recommendations using SVD and PMF models.
Users can input their ID to get personalized movie recommendations and compare model outputs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent))
from utils.recommendation import RecommendationSystem

# Page configuration
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
    }
    .recommendation-table {
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_recommendation_system():
    """Load recommendation system (cached for performance)."""
    return RecommendationSystem()


@st.cache_data
def load_metrics():
    """Load model performance metrics."""
    import json
    with open("reports/model_metrics.json") as f:
        return json.load(f)


def plot_user_ratings_distribution(top_rated_df):
    """Create histogram of user's rating distribution."""
    if len(top_rated_df) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ratings = top_rated_df['Rating'].values
    ax.hist(ratings, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 
            color='#3498db', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Rating', fontweight='bold')
    ax.set_ylabel('Number of Movies', fontweight='bold')
    ax.set_title("User's Rating Distribution", fontweight='bold')
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_model_comparison_chart(svd_recs, pmf_recs):
    """Create side-by-side comparison of top recommendations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # SVD plot
    ax1 = axes[0]
    svd_top5 = svd_recs.head(5)
    y_pos = np.arange(len(svd_top5))
    ax1.barh(y_pos, svd_top5['PredictedRating'], color='#3498db', alpha=0.8, edgecolor='black')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([title[:30] + '...' if len(title) > 30 else title 
                         for title in svd_top5['Title']], fontsize=9)
    ax1.set_xlabel('Predicted Rating', fontweight='bold')
    ax1.set_title('SVD Top 5 Recommendations', fontweight='bold', fontsize=12)
    ax1.set_xlim(0, 5.5)
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # PMF plot
    ax2 = axes[1]
    pmf_top5 = pmf_recs.head(5)
    ax2.barh(y_pos, pmf_top5['PredictedRating'], color='#2ecc71', alpha=0.8, edgecolor='black')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([title[:30] + '...' if len(title) > 30 else title 
                         for title in pmf_top5['Title']], fontsize=9)
    ax2.set_xlabel('Predicted Rating', fontweight='bold')
    ax2.set_title('PMF Top 5 Recommendations', fontweight='bold', fontsize=12)
    ax2.set_xlim(0, 5.5)
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    # Header
    st.markdown('<p class="main-header">🎬 Movie Recommender System</p>', unsafe_allow_html=True)
    st.markdown("### Powered by SVD and PMF Matrix Factorization")
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown("---")
    
    # Load recommendation system
    with st.spinner("Loading recommendation system..."):
        rec_system = load_recommendation_system()
        metrics = load_metrics()
    
    # User input
    st.sidebar.subheader("User Selection")
    user_id = st.sidebar.number_input(
        "Enter User ID (1-6040):",
        min_value=1,
        max_value=6040,
        value=42,
        step=1,
        help="Enter a user ID to get personalized recommendations"
    )
    
    st.sidebar.markdown("---")
    
    # Number of recommendations
    st.sidebar.subheader("Recommendation Settings")
    top_n = st.sidebar.slider(
        "Number of recommendations:",
        min_value=5,
        max_value=20,
        value=10,
        step=1
    )
    
    st.sidebar.markdown("---")
    
    # Model performance metrics
    st.sidebar.subheader("📊 Model Performance")
    st.sidebar.metric("SVD RMSE", f"{metrics['SVD_RMSE']:.2f}", 
                     delta="✓ Pass (≤0.90)" if metrics['SVD_passes_audit'] else "✗ Fail")
    st.sidebar.metric("PMF RMSE", f"{metrics['PMF_RMSE']:.2f}", 
                     delta=f"{metrics['improvement_%']:.1f}% better")
    
    # Main content
    try:
        # Check if user exists
        if rec_system.get_user_index(user_id) is None:
            st.error(f"❌ User ID {user_id} not found in the dataset. Please enter a valid User ID (1-6040).")
            return
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Recommendations",
            "⭐ User Profile", 
            "📊 Model Comparison",
            "📈 Visualizations"
        ])
        
        # Tab 1: Recommendations
        with tab1:
            st.markdown(f'<p class="sub-header">Personalized Recommendations for User {user_id}</p>', 
                       unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔵 SVD Recommendations")
                with st.spinner("Generating SVD recommendations..."):
                    svd_recs = rec_system.generate_recommendations(
                        user_id, model='svd', top_n=top_n
                    )
                
                # Format the dataframe
                svd_display = svd_recs[['Rank', 'Title', 'Genres', 'PredictedRating']].copy()
                svd_display['PredictedRating'] = svd_display['PredictedRating'].round(2)
                
                st.dataframe(svd_display, use_container_width=True, height=400)
                
                # Download button
                csv = svd_display.to_csv(index=False)
                st.download_button(
                    label="📥 Download SVD Recommendations",
                    data=csv,
                    file_name=f"user_{user_id}_svd_recommendations.csv",
                    mime="text/csv"
                )
            
            with col2:
                st.markdown("#### 🟢 PMF Recommendations")
                with st.spinner("Generating PMF recommendations..."):
                    pmf_recs = rec_system.generate_recommendations(
                        user_id, model='pmf', top_n=top_n
                    )
                
                # Format the dataframe
                pmf_display = pmf_recs[['Rank', 'Title', 'Genres', 'PredictedRating']].copy()
                pmf_display['PredictedRating'] = pmf_display['PredictedRating'].round(2)
                
                st.dataframe(pmf_display, use_container_width=True, height=400)
                
                # Download button
                csv = pmf_display.to_csv(index=False)
                st.download_button(
                    label="📥 Download PMF Recommendations",
                    data=csv,
                    file_name=f"user_{user_id}_pmf_recommendations.csv",
                    mime="text/csv"
                )
        
        # Tab 2: User Profile
        with tab2:
            st.markdown(f'<p class="sub-header">User {user_id} Profile</p>', 
                       unsafe_allow_html=True)
            
            top_rated = rec_system.get_top_rated_movies(user_id, top_n=20)
            
            if len(top_rated) == 0:
                st.warning(f"No rating history found for User {user_id}")
            else:
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Rated Movies", len(top_rated))
                
                with col2:
                    avg_rating = top_rated['Rating'].mean()
                    st.metric("Average Rating", f"{avg_rating:.2f} ⭐")
                
                with col3:
                    max_rating = top_rated['Rating'].max()
                    st.metric("Highest Rating", f"{max_rating:.0f} ⭐")
                
                with col4:
                    min_rating = top_rated['Rating'].min()
                    st.metric("Lowest Rating", f"{min_rating:.0f} ⭐")
                
                st.markdown("---")
                
                # Display top-rated movies
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("#### 🌟 Top-Rated Movies")
                    top_rated_display = top_rated.head(15)[['Rank', 'Title', 'Genres', 'Rating']]
                    st.dataframe(top_rated_display, use_container_width=True, height=400)
                
                with col2:
                    st.markdown("#### 📊 Rating Distribution")
                    fig = plot_user_ratings_distribution(top_rated)
                    if fig:
                        st.pyplot(fig)
        
        # Tab 3: Model Comparison
        with tab3:
            st.markdown(f'<p class="sub-header">SVD vs PMF Comparison for User {user_id}</p>', 
                       unsafe_allow_html=True)
            
            # Visual comparison
            st.markdown("#### 📊 Top 5 Recommendations Comparison")
            fig = plot_model_comparison_chart(svd_recs, pmf_recs)
            st.pyplot(fig)
            
            st.markdown("---")
            
            # Side-by-side comparison table
            st.markdown("#### 📋 Detailed Comparison")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**SVD Top 10**")
                svd_compare = svd_recs.head(10)[['Rank', 'Title', 'PredictedRating']].copy()
                svd_compare['PredictedRating'] = svd_compare['PredictedRating'].round(2)
                st.dataframe(svd_compare, use_container_width=True)
            
            with col2:
                st.markdown("**PMF Top 10**")
                pmf_compare = pmf_recs.head(10)[['Rank', 'Title', 'PredictedRating']].copy()
                pmf_compare['PredictedRating'] = pmf_compare['PredictedRating'].round(2)
                st.dataframe(pmf_compare, use_container_width=True)
            
            # Overlap analysis
            st.markdown("---")
            st.markdown("#### 🔄 Recommendation Overlap")
            
            svd_movies = set(svd_recs.head(10)['MovieID'])
            pmf_movies = set(pmf_recs.head(10)['MovieID'])
            overlap = svd_movies & pmf_movies
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Common Movies", len(overlap))
            with col2:
                st.metric("SVD Unique", len(svd_movies - pmf_movies))
            with col3:
                st.metric("PMF Unique", len(pmf_movies - svd_movies))
        
        # Tab 4: Visualizations
        with tab4:
            st.markdown('<p class="sub-header">Model Performance Visualizations</p>', 
                       unsafe_allow_html=True)
            
            # Load and display saved visualizations
            viz_dir = Path("reports")
            
            st.markdown("#### 📊 RMSE Comparison")
            if (viz_dir / "rmse_comparison.png").exists():
                st.image(str(viz_dir / "rmse_comparison.png"), 
                        use_container_width=True)
            else:
                st.warning("RMSE comparison plot not found")
            
            st.markdown("---")
            
            st.markdown("#### 🎯 Predicted vs Actual Ratings")
            if (viz_dir / "predicted_vs_actual.png").exists():
                st.image(str(viz_dir / "predicted_vs_actual.png"), 
                        use_container_width=True)
            else:
                st.warning("Predicted vs actual plot not found")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 PMF Training Convergence")
                if (viz_dir / "pmf_convergence.png").exists():
                    st.image(str(viz_dir / "pmf_convergence.png"), 
                            use_container_width=True)
                else:
                    st.warning("PMF convergence plot not found")
            
            with col2:
                st.markdown("#### 🎬 Top Recommended Movies")
                if (viz_dir / "top_recommendations.png").exists():
                    st.image(str(viz_dir / "top_recommendations.png"), 
                            use_container_width=True)
                else:
                    st.warning("Top recommendations plot not found")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
        <p>🎬 Movie Recommender System | Built with Matrix Factorization (SVD & PMF)</p>
        <p>MovieLens 1M Dataset | 6,040 users × 3,683 movies × 1M ratings</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
