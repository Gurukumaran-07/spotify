import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SpotifyRecommenderApp:
    def __init__(self, artifact_path='spotify_recommender_artifacts.pkl'):
        with open(artifact_path, 'rb') as f:
            data = pickle.load(f)
        self.scaler = data['scaler']
        self.kmeans = data['kmeans']
        self.feature_cols = data['feature_cols']
        self.X_scaled = data['X_scaled']
        self.catalog = data['catalog']

    def recommend_by_song(self, song_name, artist_name=None, top_n=10, same_cluster_only=True, min_popularity=0):
        matches = self.catalog[self.catalog['track_name'].str.lower() == song_name.lower()]
        if artist_name:
            matches = matches[matches['track_artist'].str.lower().str.contains(artist_name.lower())]
        if matches.empty:
            matches = self.catalog[self.catalog['track_name'].str.lower().str.contains(song_name.lower())]
        if matches.empty:
            raise ValueError(f"Song '{song_name}' not found in catalog.")

        target_idx = matches.index[0]
        target_row = self.catalog.loc[target_idx]
        target_vec = self.X_scaled[target_idx].reshape(1, -1)

        mask = (self.catalog.index != target_idx) & (self.catalog['track_popularity'] >= min_popularity)
        if same_cluster_only:
            mask = mask & (self.catalog['cluster'] == target_row['cluster'])

        candidates = self.catalog[mask].copy()
        candidates['similarity_score'] = cosine_similarity(target_vec, self.X_scaled[mask]).flatten()

        recs = (
            candidates.sort_values(by=['similarity_score', 'track_popularity'], ascending=[False, False])
            .drop_duplicates(subset=['base_track_name', 'track_artist'])
            .head(top_n)
        )
        return target_row, recs[['track_name', 'track_artist', 'playlist_genre', 'playlist_subgenre', 'cluster', 'track_popularity', 'similarity_score']]

    def recommend_by_mood(self, target_features_dict, top_n=10, min_popularity=30):
        default_vec = self.catalog[self.feature_cols].median().to_dict()
        default_vec.update(target_features_dict)
        query_df = pd.DataFrame([default_vec])[self.feature_cols]
        query_scaled = self.scaler.transform(query_df)
        pred_cluster = self.kmeans.predict(query_scaled)[0]

        mask = (self.catalog['cluster'] == pred_cluster) & (self.catalog['track_popularity'] >= min_popularity)
        candidates = self.catalog[mask].copy()
        candidates['similarity_score'] = cosine_similarity(query_scaled, self.X_scaled[mask]).flatten()

        recs = (
            candidates.sort_values(by=['similarity_score', 'track_popularity'], ascending=[False, False])
            .drop_duplicates(subset=['base_track_name', 'track_artist'])
            .head(top_n)
        )
        return pred_cluster, recs[['track_name', 'track_artist', 'playlist_genre', 'playlist_subgenre', 'cluster', 'track_popularity', 'similarity_score']]

if __name__ == '__main__':
    app = SpotifyRecommenderApp('spotify_recommender_artifacts.pkl')
    seed, recs = app.recommend_by_song("Memories - Dillon Francis Remix", artist_name="Maroon 5", top_n=10)
    print(f"Seed Track: {seed['track_name']} - {seed['track_artist']} (Cluster {seed['cluster']})")
    print(recs.to_string(index=False))
