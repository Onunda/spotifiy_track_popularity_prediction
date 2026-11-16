import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBClassifier


def main():
    print("Loading datasets for training...")
    df_tracks = pd.read_csv("data/spotify_tracks.csv")
    df_market = pd.read_csv("data/featured_Spotify_track_info.csv")

    primary_artist_col = 'artists' if 'artists' in df_tracks.columns else 'artist_name'
    df_tracks['artist_key'] = df_tracks[primary_artist_col].astype(str).str.lower().str.strip()
    df_market['artist_key'] = df_market['names'].astype(str).str.lower().str.strip()
    df_market_unique = df_market.drop_duplicates(subset=['artist_key'])

    df_merged = pd.merge(
        df_tracks,
        df_market_unique[['artist_key', 'followers', 'monthly_listeners', 'playlists_found', 'popularity']],
        on='artist_key',
        how='left',
        suffixes=('', '_market')
    )

    df_merged['has_market_data'] = (df_merged['followers'] > 0).astype(int)
    df_merged['followers'] = df_merged['followers'].fillna(0)
    df_merged['monthly_listeners'] = df_merged['monthly_listeners'].fillna(0)
    df_merged['playlists_found'] = df_merged['playlists_found'].fillna(0)
    df_merged['major_label'] = (df_merged['followers'] > 1000000).astype(int)

    target_col = 'popularity_y' if 'popularity_y' in df_merged.columns else 'popularity'
    df_merged['target'] = (pd.to_numeric(df_merged[target_col], errors='coerce') >= 50).astype(int)
    df_merged['duration_bucket'] = pd.cut(
        df_merged['duration_ms'],
        bins=[0, 120000, 240000, np.inf],
        labels=['short', 'medium', 'long']
    ).astype(str)
    df_merged['acoustic_energy_ratio'] = df_merged['acousticness'] / (df_merged['energy'] + 1e-5)

    extended_features = [
        'track_genre', 'explicit', 'key', 'mode', 'time_signature',
        'duration_ms', 'energy', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo',
        'danceability', 'loudness', 'followers', 'monthly_listeners',
        'playlists_found', 'major_label', 'has_market_data',
        'duration_bucket', 'acoustic_energy_ratio'
    ]

    df_train, _ = train_test_split(df_merged, test_size=0.2, random_state=42, stratify=df_merged['target'])
    y_train = df_train['target'].values

    train_dicts = df_train[extended_features].to_dict(orient='records')
    dv = DictVectorizer(sparse=True)
    X_train = dv.fit_transform(train_dicts)

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight_val = neg_count / pos_count

    print("Training XGBoost Classifier...")
    xgb_model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_pos_weight_val,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    spotify_artifact = {
        "dv": dv,
        "model": xgb_model,
        "features": extended_features,
        "target_definition": "High Popularity (Score >= 50)"
    }

    output_path = "models/model.pkl"
    joblib.dump(spotify_artifact, output_path)
    print(f"Model artifact bundle successfully saved to '{output_path}'!")


if __name__ == "__main__":
    main()