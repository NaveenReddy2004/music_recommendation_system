import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import warnings

# Suppress RuntimeWarning for explained_variance_ratio_
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.decomposition._truncated_svd")

# Configuration: Set to False to skip SVD and use target as collab_scores
USE_SVD = False

# Load preprocessed data
feature_matrix = pd.read_csv('feature_matrix.csv')
track_indices = pd.read_csv('track_indices.csv', index_col=0)
processed_data = pd.read_csv('processed_data.csv')

# Check for duplicate song_id entries
duplicate_songs = processed_data[processed_data['song_id'].duplicated(keep=False)]
if not duplicate_songs.empty:
    print(f"Found {len(duplicate_songs)} duplicate song_id entries:")
    print(duplicate_songs[['song_id', 'song_title', 'artist', 'target']])
    print("Aggregating duplicates by keeping the first occurrence.")
    processed_data = processed_data.drop_duplicates(subset='song_id', keep='first').reset_index(drop=True)
    print(f"Dataset after removing duplicates: {len(processed_data)} rows.")
    feature_matrix = feature_matrix.loc[processed_data.index].reset_index(drop=True)
    track_indices = pd.Series(processed_data.index, index=processed_data['song_id'])

# Check target distribution
print("Target distribution:")
print(processed_data['target'].value_counts(normalize=True))

# Content-Based Filtering: Compute cosine similarity
similarity_matrix = cosine_similarity(feature_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=processed_data['song_id'], columns=processed_data['song_id'])

# Collaborative Filtering
if USE_SVD:
    if processed_data['target'].nunique() == 1:
        print("Warning: 'target' has only one unique value. Using zero scores.")
        collab_scores = np.zeros(len(processed_data))
    else:
        # Simulate user-item matrix
        user_item_matrix = pd.DataFrame({
            'user_id': ['user_1'] * len(processed_data),
            'song_id': processed_data['song_id'],
            'rating': processed_data['target']
        })

        # Pivot to create user-item matrix
        try:
            user_item_pivot = user_item_matrix.pivot(index='user_id', columns='song_id', values='rating').fillna(0)
        except ValueError as e:
            print(f"Error during pivot: {e}")
            raise

        user_item_sparse = csr_matrix(user_item_pivot.values)

        # Check matrix sparsity
        non_zero_count = user_item_sparse.nnz
        total_elements = user_item_sparse.shape[0] * user_item_sparse.shape[1]
        print(f"User-item matrix: {user_item_sparse.shape}, non-zero entries: {non_zero_count} ({non_zero_count / total_elements:.2%})")

        # Apply SVD
        try:
            svd = TruncatedSVD(n_components=1, random_state=42)  # Reduced to 1 due to rank-1 matrix
            latent_matrix = svd.fit_transform(user_item_sparse)
            song_latent_matrix = svd.components_.T
            collab_scores = np.dot(latent_matrix, svd.components_)[0]
        except ValueError as e:
            print(f"SVD failed: {e}. Using zero scores.")
            collab_scores = np.zeros(user_item_sparse.shape[1])
else:
    # Use target as collaborative scores (no SVD)
    print("Skipping SVD. Using normalized target as collaborative scores.")
    collab_scores = processed_data['target'].values

# Normalize collaborative scores to [0, 1]
if np.any(collab_scores):
    collab_scores = (collab_scores - collab_scores.min()) / (collab_scores.max() - collab_scores.min())
else:
    collab_scores = np.zeros(len(processed_data))
    print("Collaborative scores are all zero. Relying on content-based filtering.")

# Hybrid Recommendation Function
def hybrid_recommend(song_id, n_recommendations=5, content_weight=0.9):
    if song_id not in similarity_df.index:
        return f"Song {song_id} not found."
    
    # Content-based scores
    content_scores = similarity_df.loc[song_id]
    
    # Collaborative scores (aligned with song_id order)
    collab_df = pd.Series(collab_scores, index=user_item_pivot.columns if 'user_item_pivot' in locals() else processed_data['song_id'])
    
    # Combine scores
    hybrid_scores = content_weight * content_scores + (1 - content_weight) * collab_df
    
    # Exclude the input song and sort
    hybrid_scores = hybrid_scores.drop(song_id, errors='ignore')
    recommendations = hybrid_scores.sort_values(ascending=False).head(n_recommendations)
    
    return recommendations.index.tolist()

# Save similarity matrix and collaborative scores
similarity_df.to_csv('similarity_matrix.csv')
np.save('collab_scores.npy', collab_scores)

# Example: Recommend songs similar to "Mask Off - Future"
print("Recommendations for 'Mask Off - Future':")
print(hybrid_recommend('Mask Off - Future', n_recommendations=5))