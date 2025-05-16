import streamlit as st
import pandas as pd
import numpy as np

# Load precomputed data
similarity_df = pd.read_csv('similarity_matrix.csv', index_col=0)
collab_scores = np.load('collab_scores.npy')
processed_data = pd.read_csv('processed_data.csv')

# Hybrid recommendation function
def hybrid_recommend(song_id, n_recommendations=5, content_weight=0.9):
    if song_id not in similarity_df.index:
        return []
    content_scores = similarity_df.loc[song_id]
    collab_df = pd.Series(collab_scores, index=similarity_df.columns)
    hybrid_scores = content_weight * content_scores + (1 - content_weight) * collab_df
    hybrid_scores = hybrid_scores.drop(song_id, errors='ignore')
    return hybrid_scores.sort_values(ascending=False).head(n_recommendations).index.tolist()

# Streamlit app
st.title("Music Recommendation System")
st.write("Select a song to get recommendations.")

# Song selection
song_options = processed_data['song_id'].tolist()
selected_song = st.selectbox("Choose a song:", song_options)

# Number of recommendations
n_recommendations = st.slider("Number of recommendations:", 1, 10, 5)

# Get recommendations
if st.button("Get Recommendations"):
    recommendations = hybrid_recommend(selected_song, n_recommendations=n_recommendations)
    if recommendations:
        st.write(f"Recommendations for '{selected_song}':")
        for i, song in enumerate(recommendations, 1):
            st.write(f"{i}. {song}")
    else:
        st.error(f"Song '{selected_song}' not found.")

# Display dataset preview
if st.checkbox("Show dataset preview"):
    st.write(processed_data[['song_id', 'artist', 'target']].head())