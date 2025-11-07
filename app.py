import os
import re
import json
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from flask_cors import CORS
from tensorflow import keras
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import requests
from urllib.parse import quote_plus
from openai import OpenAI

# =========================
# Load environment variables
# =========================
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

client = OpenAI(api_key=HF_API_KEY, base_url="https://router.huggingface.co/v1")

# =========================
# Flask setup
# =========================
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")
CORS(app)

# =========================
# Load models and dataset
# =========================
MODELS_DIR = os.getenv("MODELS_DIR", "models")
fnn_model = keras.models.load_model(os.path.join(MODELS_DIR, "fnn_model.keras"))
xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

DATASET_PATH = os.getenv("DATASET_PATH", "dataset.csv")
df_full = pd.read_csv(DATASET_PATH)

# -------------------------
# Metadata and features
# -------------------------
metadata_cols = ["track_name", "artists", "album_name", "track_id"]
metadata = df_full[metadata_cols].fillna("Unknown").reset_index(drop=True)

drop_cols = metadata_cols + ["index", "popularity"]
df_features = df_full.drop(columns=[c for c in drop_cols if c in df_full.columns], errors='ignore')

# One-hot encode categorical columns
cats = [c for c in ["track_genre", "explicit"] if c in df_features.columns]
if cats:
    df_features = pd.get_dummies(df_features, columns=cats, drop_first=True)

# Align features with scaler
feature_columns = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else list(df_features.columns)
for col in feature_columns:
    if col not in df_features.columns:
        df_features[col] = 0.0
df_features = df_features[feature_columns].reset_index(drop=True)
df_features_scaled = scaler.transform(df_features)
# =========================
# Safe float conversion
# =========================
def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0

# =========================
# Utility functions
# =========================
def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0

def extract_json_from_text(text: str):
    """Extract JSON robustly from LLM output."""
    if not text:
        return None
    try:
        return json.loads(text)
    except:
        pass
    m = re.search(r"```json(.*?)```", text, flags=re.S | re.I)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except:
            pass
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except:
                        break
    return None

# =========================
# LLM features
# =========================
def get_features_from_llm(track_name, artist):
    prompt = f"""
Estimate realistic Spotify-style audio features for the song '{track_name}' by '{artist}'.
Return ONLY a JSON object (no extra text) with numeric values for:
duration_ms, danceability, energy, key, loudness, mode, speechiness, acousticness,
instrumentalness, liveness, valence, tempo, time_signature, explicit (0 or 1),
and track_genre as one of ['acoustic','pop','rock','hip hop','jazz','classical'].
"""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b:groq",
            messages=[{"role": "user", "content": prompt}],
        )
        generated_text = response.choices[0].message.content
    except Exception as e:
        raise Exception(f"LLM request failed: {e}")

    features = extract_json_from_text(generated_text)
    if features is None:
        raise Exception(f"Could not parse JSON from LLM output: {generated_text[:1000]}")
    
    cleaned = {}
    for k, v in features.items():
        if isinstance(v, (int, float)):
            cleaned[k] = v
        elif isinstance(v, str):
            try:
                cleaned[k] = float(v)
            except:
                cleaned[k] = v
        else:
            cleaned[k] = v
    return cleaned

# =========================
# Map LLM JSON -> model features
# =========================
def derive_features_from_metadata(features_dict):
    row = {c: 0.0 for c in feature_columns}
    lower_features = {k.lower(): v for k, v in features_dict.items()}
    normalized_to_raw = {k.replace("_", ""): k for k in lower_features.keys()}

    for model_col in feature_columns:
        mlow = model_col.lower()
        norm = mlow.replace("_", "")

        # numeric match
        if norm in normalized_to_raw:
            raw = normalized_to_raw[norm]
            row[model_col] = safe_float(lower_features[raw])
            continue

        # genre one-hot
        if mlow.startswith("track_genre"):
            genre_val = str(lower_features.get("track_genre", "")).strip().lower()
            row[model_col] = 1.0 if genre_val and genre_val in mlow else 0.0
            continue

        # explicit
        if "explicit" in mlow:
            explicit_num = safe_float(lower_features.get("explicit", 0))
            if mlow == "explicit" or mlow.endswith("_1") or "true" in mlow:
                row[model_col] = 1.0 if explicit_num else 0.0
            else:
                row[model_col] = 0.0

    return pd.DataFrame([row], columns=feature_columns)

# =========================
# YouTube helper
# =========================
def get_youtube_link1(track_name, artist):
    query = f"{track_name} {artist}"
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&key={YOUTUBE_API_KEY}&maxResults=1&type=video"
    try:
        res = requests.get(url, timeout=6)
        res.raise_for_status()
        data = res.json()
        if "items" in data and len(data["items"]) > 0:
            vid = data["items"][0].get("id", {}).get("videoId")
            if vid:
                return f"https://www.youtube.com/watch?v={vid}"
    except Exception as e:
        print(f"⚠️ YouTube API error: {e}")
    return f"https://www.youtube.com/results?search_query={query}"

def get_youtube_link(track_name, artist):
    q = f"{track_name} {artist}".strip()
    query = quote_plus(q)
    if not YOUTUBE_API_KEY:
        return f"https://www.youtube.com/results?search_query={query}"
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&key={YOUTUBE_API_KEY}&maxResults=1&type=video"
    try:
        res = requests.get(url, timeout=6)
        res.raise_for_status()
        data = res.json()
        if "items" in data and len(data["items"]) > 0:
            vid = data["items"][0].get("id", {}).get("videoId")
            if vid:
                return f"https://www.youtube.com/watch?v={vid}"
    except Exception as e:
        print(f"⚠️ YouTube API error: {e}")
    return f"https://www.youtube.com/results?search_query={query}"

# =========================
# Recommendation engine
# =========================
def get_recommendations1(features_scaled):
    similarities = cosine_similarity(features_scaled, df_features_scaled)
    top_indices = similarities[0].argsort()[::-1][1:6]

    recommendations = []
    for idx in top_indices:
        rec = metadata.iloc[idx].to_dict()
        rec_features = df_features_scaled[idx:idx + 1]
        rec["predicted_popularity"] = float(
            (fnn_model.predict(rec_features) + xgb_model.predict(rec_features)) / 2
        )
        rec["youtube_link"] = get_youtube_link1(rec["track_name"], rec["artists"])
        recommendations.append(rec)
    return recommendations


def get_recommendations(features_scaled, top_k=5):
    arr = np.asarray(features_scaled)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    similarities = cosine_similarity(arr, df_features_scaled)
    sorted_idx = similarities[0].argsort()[::-1]

    recs = []
    picked = 0
    for idx in sorted_idx:
        if picked >= top_k:
            break
        i = int(idx)
        if i < 0 or i >= len(metadata):
            continue

        md = metadata.iloc[i].to_dict()
        track_name = str(md.get("track_name", "Unknown"))
        artists = str(md.get("artists", "Unknown"))

        rec_features = df_features_scaled[i:i+1]
        try:
            fnn_p = float(fnn_model.predict(rec_features).ravel()[0])
        except:
            fnn_p = None
        try:
            xgb_p = float(xgb_model.predict(rec_features).ravel()[0])
        except:
            xgb_p = None
        avg_pred = (fnn_p + xgb_p)/2.0 if fnn_p is not None and xgb_p is not None else None

        recs.append({
            "track": track_name,
            "artist": artists,
            "predicted_popularity": avg_pred,
            "distance": float(similarities[0][i]),
            "youtube": get_youtube_link(track_name, artists)
        })
        picked += 1
    return recs

# =========================
# Flask routes
# =========================
@app.route("/")
def home_manual():
    return render_template("manual.html", feature_columns=feature_columns, result=None)

@app.route("/manual", methods=["POST"])
def manual_predict():
    # Safe float conversion for manual inputs
    data = {c: safe_float(request.form.get(c)) for c in feature_columns}
    features = pd.DataFrame([data])
    features_scaled = scaler.transform(features)

    fnn_pred = fnn_model.predict(features_scaled).flatten()
    xgb_pred = xgb_model.predict(features_scaled)
    final_pred = (fnn_pred + xgb_model.predict(features_scaled)) / 2

    recommendations = get_recommendations1(features_scaled)
    result = {
        "prediction": {
            "fnn_prediction": float(fnn_pred[0]),
            "xgb_prediction": float(xgb_pred[0]),
            "popularity_prediction": float(final_pred[0])
        },
        "recommendations": recommendations
    }

    return render_template("manual.html", feature_columns=feature_columns, result=result)


@app.route("/search", methods=["GET", "POST"])
def search_song():
    result = None
    error_message = None

    if request.method == "POST":
        track_name = request.form.get("track_name", "").strip()
        artist = request.form.get("artist", "").strip()
        if not track_name or not artist:
            error_message = "Please enter both Track Name and Artist."
        else:
            try:
                metadata_found = get_features_from_llm(track_name, artist)
                features_df = derive_features_from_metadata(metadata_found)
                features_scaled = scaler.transform(features_df)

                fnn_arr = fnn_model.predict(features_scaled).ravel()
                xgb_arr = xgb_model.predict(features_scaled).ravel()
                final = (float(fnn_arr[0]) + float(xgb_arr[0])) / 2.0

                recs = get_recommendations(features_scaled)
                result = {
                    "metadata": metadata_found,
                    "prediction": {
                        "fnn_prediction": float(fnn_arr[0]),
                        "xgb_prediction": float(xgb_arr[0]),
                        "popularity_prediction": final,
                    },
                    "recommendations": recs
                }
            except Exception as e:
                error_message = f"Error: {e}"

    return render_template("search.html", result=result, error_message=error_message)

# =========================
# Run server
# =========================
if __name__ == "__main__":
    print("🚀 Starting Flask Music Popularity Predictor on port 5000...")
    app.run(debug=True, port=5000)
