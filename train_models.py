import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib

# =========================
# 1. Load dataset
# =========================
file_path = "dataset.csv"
try:
    df_full = pd.read_csv(file_path)
    print("✅ Dataset loaded successfully.")
except Exception as e:
    print("❌ Error loading dataset. Make sure 'dataset.csv' is in the working directory.")
    print(f"Error: {e}")
    exit()

# =========================
# 2. Save metadata separately
# =========================
metadata_cols = ['track_name', 'artists', 'album_name', 'track_id']
metadata = df_full[metadata_cols].copy()
df = df_full.drop(columns=metadata_cols + ['index'], errors='ignore')

# =========================
# 3. One-hot encode categorical columns
# =========================
cols_to_encode = [col for col in ['track_genre', 'explicit'] if col in df.columns]
df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

# =========================
# 4. Handle zeros as NaN
# =========================
cols_to_impute = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]
for col in cols_to_impute:
    if col in df.columns:
        df[col] = df[col].replace(0, np.nan)

# =========================
# 5. Shuffle the dataframe
# =========================
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
metadata = metadata.iloc[df_shuffled.index].reset_index(drop=True)

# =========================
# 6. Convert popularity to classes (0,1,2)
# =========================
bins = [0, 33, 66, 100]
labels = [0, 1, 2]
df_shuffled['popularity_class'] = pd.cut(
    df_shuffled['popularity'],
    bins=bins,
    labels=labels,
    include_lowest=True
).cat.codes

# =========================
# 7. Train-test split
# =========================
train_ratio = 0.8
train_split_point = int(len(df_shuffled) * train_ratio)

train_df = df_shuffled.iloc[:train_split_point]
test_df = df_shuffled.iloc[train_split_point:]

metadata_train = metadata.iloc[:train_split_point]
metadata_test = metadata.iloc[train_split_point:]

# =========================
# 8. Separate features and target
# =========================
X_train = train_df.drop(['popularity', 'popularity_class'], axis=1)
y_train = train_df['popularity_class']
X_test = test_df.drop(['popularity', 'popularity_class'], axis=1)
y_test = test_df['popularity_class']

# =========================
# 9. Impute missing values with training mean
# =========================
for col in cols_to_impute:
    if col in X_train.columns:
        mean_val = X_train[col].mean()
        X_train[col] = X_train[col].fillna(mean_val)
        X_test[col] = X_test[col].fillna(mean_val)

# =========================
# 10. Standardize features using StandardScaler
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Data preprocessing complete for classification.")

# =========================
# 11. Define FNN model
# =========================
def create_fnn_model(input_dim):
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # regression output
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# =========================
# 12. Train FNN
# =========================
fnn_model = create_fnn_model(X_train_scaled.shape[1])
fnn_model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.1)

# =========================
# 13. Train XGBoost
# =========================
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6
)
xgb_model.fit(X_train_scaled, y_train)

# =========================
# 14. Save models and scaler
# =========================
os.makedirs("models", exist_ok=True)
fnn_model.save('models/fnn_model.keras')  # Use Keras native format
joblib.dump(xgb_model, 'models/xgb_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("✅ Models and scaler saved successfully!")
