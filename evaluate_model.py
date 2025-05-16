import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
processed_data = pd.read_csv('processed_data.csv')
similarity_df = pd.read_csv('similarity_matrix.csv', index_col=0)
collab_scores = np.load('collab_scores.npy')

# Hybrid recommendation function
def hybrid_recommend(song_id, n_recommendations=5, content_weight=0.9):
    if song_id not in similarity_df.index:
        return []
    content_scores = similarity_df.loc[song_id]
    collab_df = pd.Series(collab_scores, index=similarity_df.columns)
    hybrid_scores = content_weight * content_scores + (1 - content_weight) * collab_df
    hybrid_scores = hybrid_scores.drop(song_id, errors='ignore')
    return hybrid_scores.sort_values(ascending=False).head(n_recommendations).index.tolist()

# Split data: Use 'target' as ground truth
train_data, test_data = train_test_split(processed_data, test_size=0.2, random_state=42)

# Precision@k
def precision_at_k(test_data, k=5):
    relevant_songs = test_data[test_data['target'] == 1]['song_id'].tolist()
    if not relevant_songs:
        print("No relevant songs (target=1) in test set.")
        return 0.0
    precisions = []
    for song_id in relevant_songs[:10]:  # Evaluate on subset
        recommended = hybrid_recommend(song_id, n_recommendations=k)
        relevant_recommended = len([s for s in recommended if s in relevant_songs])
        precisions.append(relevant_recommended / k)
    return np.mean(precisions) if precisions else 0.0

# Evaluate
precision = precision_at_k(test_data, k=5)
print(f"Precision@5: {precision:.4f}")

# Qualitative evaluation
test_song = 'Mask Off - Future'
print(f"\nRecommendations for '{test_song}':")
print(hybrid_recommend(test_song, n_recommendations=5))