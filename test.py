import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import joblib
from urllib.parse import quote_plus
import requests
from openai import OpenAI
from dotenv import load_dotenv

# =========================
# Load environment variables
# =========================
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
client = OpenAI(api_key=HF_API_KEY, base_url="https://router.huggingface.co/v1")

# =========================
# Load dataset & scaler
# =========================
DATASET_PATH = "dataset.csv"
SCALER_PATH = "models/scaler.pkl"

df_full = pd.read_csv(DATASET_PATH)
scaler = joblib.load(SCALER_PATH)

# Metadata columns
metadata_cols = ["track_name", "artists", "album_name", "track_id"]
metadata = df_full[metadata_cols].fillna("Unknown").reset_index(drop=True)

# Features
drop_cols = metadata_cols + ["index", "popularity"]
df_features = df_full.drop(columns=[c for c in drop_cols if c in df_full.columns], errors="ignore")
if "track_genre" in df_features.columns or "explicit" in df_features.columns:
    df_features = pd.get_dummies(df_features, columns=[c for c in ["track_genre","explicit"] if c in df_features.columns], drop_first=True)

# Align columns to scaler
feature_columns = list(scaler.feature_names_in_)
for col in feature_columns:
    if col not in df_features.columns:
        df_features[col] = 0.0
df_features = df_features[feature_columns].reset_index(drop=True)
df_features_scaled = scaler.transform(df_features)

# =========================
# Function to get LLM features
# =========================
def get_features_from_llm(track_name, artist):
    prompt = f"""
Estimate realistic Spotify-style audio features for the song '{track_name}' by '{artist}'.
Return ONLY a JSON object (no extra text) with numeric values for:
duration_ms, danceability, energy, key, loudness, mode, speechiness, acousticness,
instrumentalness, liveness, valence, tempo, time_signature, explicit (0 or 1),
and track_genre as one of ['acoustic','pop','rock','hip hop','jazz','classical'].
"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b:groq",
        messages=[{"role": "user", "content": prompt}],
    )
    generated_text = response.choices[0].message.content
    try:
        return json.loads(generated_text)
    except:
        return None

# =========================
# Map LLM JSON -> feature vector
# =========================
def derive_features_from_metadata(features_dict, feature_columns):
    row = {c: 0.0 for c in feature_columns}
    lower_features = {k.lower(): v for k, v in features_dict.items()}
    normalized_to_raw = {k.replace("_", ""): k for k in lower_features.keys()}

    for model_col in feature_columns:
        mlow = model_col.lower()
        norm = mlow.replace("_", "")

        # Numeric match
        if norm in normalized_to_raw:
            raw = normalized_to_raw[norm]
            try:
                row[model_col] = float(lower_features[raw])
            except:
                row[model_col] = 0.0
            continue

        # Genre one-hot
        if mlow.startswith("track_genre"):
            genre_val = str(lower_features.get("track_genre", "")).strip().lower()
            row[model_col] = 1.0 if genre_val and genre_val in mlow else 0.0
            continue

        # Explicit handling
        if "explicit" in mlow:
            explicit_val = float(lower_features.get("explicit", 0))
            if mlow == "explicit" or mlow.endswith("_1"):
                row[model_col] = 1.0 if explicit_val else 0.0

    return pd.DataFrame([row], columns=feature_columns)

# =========================
# YouTube helper
# =========================
def get_youtube_link(track_name, artist):
    query = quote_plus(f"{track_name} {artist}")
    return f"https://www.youtube.com/results?search_query={query}"

# =========================
# Recommendation engine
# =========================
nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
nn_model.fit(df_features_scaled)

# =========================
# Test with a single song
# =========================
track_name = "Shape of You"
artist = "Ed Sheeran"

llm_features = get_features_from_llm(track_name, artist)
if llm_features is None:
    raise Exception("Failed to get features from LLM.")

features_df = derive_features_from_metadata(llm_features, feature_columns)
features_scaled = scaler.transform(features_df)

distances, indices = nn_model.kneighbors(features_scaled)
recs = []
for dist, idx in zip(distances[0], indices[0]):
    md = metadata.iloc[idx].to_dict()
    recs.append({
        "track_name": md.get("track_name", "Unknown"),
        "artists": md.get("artists", "Unknown"),
        "distance": float(dist),
        "youtube_link": get_youtube_link(md.get("track_name", ""), md.get("artists", ""))
    })

# =========================
# Print results
# =========================
print("LLM Features for:", track_name, "by", artist)
print(json.dumps(llm_features, indent=2))
print("\nTop 5 Recommendations:")
for r in recs:
    print(f"{r['track_name']} by {r['artists']} - Distance: {r['distance']:.4f}")
    print(f"Watch: {r['youtube_link']}")
