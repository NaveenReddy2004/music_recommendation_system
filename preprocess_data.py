import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew

# Load dataset
df = pd.read_csv('data.csv')

# Check for duplicate song_title and artist combinations
df['temp_song_id'] = df['song_title'] + ' - ' + df['artist']
duplicate_songs = df[df['temp_song_id'].duplicated(keep=False)]
if not duplicate_songs.empty:
    print(f"Found {len(duplicate_songs)} duplicate song entries:")
    print(duplicate_songs[['song_title', 'artist', 'target']])
    print("Keeping the first occurrence of each duplicate.")
    df = df.drop_duplicates(subset=['song_title', 'artist'], keep='first').reset_index(drop=True)
    print(f"Dataset after removing duplicates: {len(df)} rows.")
df = df.drop(columns=['temp_song_id'])

# Define numerical features for outlier capping
numerical_features = ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 
                     'liveness', 'loudness', 'speechiness', 'tempo', 'valence']

# Define content features for preprocessing
content_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                   'liveness', 'loudness', 'speechiness', 'tempo', 'valence']

# Outlier capping using IQR
def cap_outliers(df, features):
    df_capped = df.copy()
    capping_summary = {}
    for feature in features:
        Q1 = df_capped[feature].quantile(0.25)
        Q3 = df_capped[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Count values to be capped
        lower_outliers = (df_capped[feature] < lower_bound).sum()
        upper_outliers = (df_capped[feature] > upper_bound).sum()
        capping_summary[feature] = {'lower_capped': lower_outliers, 'upper_capped': upper_outliers}
        # Cap outliers
        df_capped[feature] = df_capped[feature].clip(lower=lower_bound, upper=upper_bound)
    return df_capped, capping_summary

# Apply outlier capping
df_clean, capping_summary = cap_outliers(df, numerical_features)
print(f"Outlier capping summary:")
for feature, summary in capping_summary.items():
    print(f"{feature}: {summary['lower_capped']} values capped at lower bound, {summary['upper_capped']} values capped at upper bound")
print(f"Dataset size after capping outliers: {len(df_clean)} rows.")

# Function to select features for log1p transformation
def select_features_for_log1p(df, features, skew_threshold=0.75):
    features_to_transform = []
    for feature in features:
        feature_skew = skew(df[feature].dropna())
        if feature_skew > skew_threshold or feature in ['loudness', 'tempo']:
            features_to_transform.append(feature)
    return features_to_transform

# Function to apply log1p transformation
def apply_log1p_transformation(df, features):
    df_transformed = df.copy()
    for feature in features:
        if feature == 'loudness':
            min_loudness = df_transformed[feature].min()
            shift = abs(min_loudness) + 1
            df_transformed[feature] = np.log1p(df_transformed[feature] + shift)
        else:
            df_transformed[feature] = df_transformed[feature].clip(lower=0)
            df_transformed[feature] = np.log1p(df_transformed[feature])
    return df_transformed

# Apply log1p transformation
features_to_transform = select_features_for_log1p(df_clean, content_features)
df_clean = apply_log1p_transformation(df_clean, features_to_transform)

# Scale the content features
scaler = StandardScaler()
df_scaled = df_clean.copy()
df_scaled[content_features] = scaler.fit_transform(df_clean[content_features])

# Add artist information using one-hot encoding
artist_dummies = pd.get_dummies(df_scaled['artist'], prefix='artist')

# Combine audio features with artist encoding
feature_matrix = pd.concat([df_scaled[content_features], artist_dummies], axis=1)

# Create a mapping from song_id to index
df_scaled['song_id'] = df_scaled['song_title'] + ' - ' + df_scaled['artist']
track_indices = pd.Series(df_scaled.index, index=df_scaled['song_id'])

# Save outputs
feature_matrix.to_csv('feature_matrix.csv', index=False)
track_indices.to_csv('track_indices.csv')
df_scaled.to_csv('processed_data.csv', index=False)
print("Preprocessing complete. Saved feature_matrix.csv, track_indices.csv, processed_data.csv")