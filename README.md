## Problem statement
The goal of this project is to build a robust binary classification model that predicts whether a given Spotify track
will achieve **"High Popularity"** (defined as a 'popularity score >= 50 or 0.5') or remain in the low-to-moderate
popularity tier.
This helps identify the underlying audio characteristics and genre traits that separate mainstream hits from underground
or niche releases.
### Question 
1. How accurately can song popularity be predicted using audio features alone?
2. What drives track success more: artist reach and playlists, or acoustic qualities?
### Feature engineering methodology options
> 1. **Acoustic-Energy Ratio**
* **Transformation:** `acousticness / (energy + 1e-5)`
* **Purpose:** Separates mellow acoustic tracks from high-energy electronic/rock.
> 2. **Speech-to-Music Intensity**
* **Transformation:** Interaction between speech and instrumentals
> 3. **Duration Buckets(range)**
* **Transformation:** `duration_ms` (`<2m`, `2–4m`, `>4m`)
* **Purpose:** Captures streaming trends favoring shorter tracks for replay value.
> 4. **Key-Mode Combinations**
* **Transformation:** Mapped numeric key (0–11) & mode (0/1)
* **Purpose:** Converts raw integers into categorical labels (e.g., `"C_Major"`).
---
## Dataset description
### Datasets overview
They are two datasets that are explored that are closely related and that are merged for better analysis:
1. [Spotify tracks dataset source](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset/code)
* notebook/spotify_track_eda.ipynb
* The dataset contains the following audio characteristics as seen in the defined columns;

```text
Data columns (total 21 columns):
 #   Column            Non-Null Count   Dtype  
---  ------            --------------   -----  
 0   Unnamed: 0        114000 non-null  int64  
 1   track_id          114000 non-null  object 
 2   artists           113999 non-null  object 
 3   album_name        113999 non-null  object 
 4   track_name        113999 non-null  object 
 5   popularity        114000 non-null  int64  
 6   duration_ms       114000 non-null  int64  
 7   explicit          114000 non-null  bool   
 8   danceability      114000 non-null  float64
 9   energy            114000 non-null  float64
 10  key               114000 non-null  int64  
 11  loudness          114000 non-null  float64
 12  mode              114000 non-null  int64  
 13  speechiness       114000 non-null  float64
 14  acousticness      114000 non-null  float64
 15  instrumentalness  114000 non-null  float64
 16  liveness          114000 non-null  float64
 17  valence           114000 non-null  float64
 18  tempo             114000 non-null  float64
 19  time_signature    114000 non-null  int64  
 20  track_genre       114000 non-null  object 
dtypes: bool(1), float64(9), int64(6), object(5)
```
**Feature and target col selection from dataset**
> Features planning to use to explore data set both from columns and extra columns from derived values:
```text
features = [
    'track_genre', 'explicit', 'key', 'mode', 'time_signature', 
    'duration_ms', 'energy', 'speechiness', 'acousticness', 
    'instrumentalness', 'liveness', 'valence', 'tempo',
    'danceability', 'loudness' 
]
extended_features = features + ['duration_range', 'acoustic_energy_ratio']
target_col = 'popularity'
```
* Artist popularity is what determines success reason why its choosen as the target column to use.
** Feature engineering Methodology **
2. [Featured Spotify Artists/Tracks with Metadata source](https://www.kaggle.com/datasets/sarahjeffreson/featured-spotify-artiststracks-with-metadata)
* notebook/spotify_market_driver.ipynb
* The dataset contains the following audio characteristics as seen in the defined columns;
```text
Data columns (total 13 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   dates              20251 non-null  object 
 1   ids                20251 non-null  object 
 2   names              20251 non-null  object 
 3   monthly_listeners  20251 non-null  float64
 4   popularity         20251 non-null  int64  
 5   followers          20251 non-null  int64  
 6   genres             20251 non-null  object 
 7   first_release      20251 non-null  int64  
 8   last_release       20251 non-null  int64  
 9   num_releases       20251 non-null  int64  
 10  num_tracks         20251 non-null  int64  
 11  playlists_found    20251 non-null  object 
 12  feat_track_ids     20251 non-null  object 
 dtypes: float64(1), int64(6), object(6)
```
3. Initial Features selected from merged datasets merge for further analysis
* notebook/spotify_market_driver.ipynb
* Market Drivers selected as seen in notebook/spotify_track_eda.ipynb
* Feature selected to be added to spotify_track selected features
```text
market_driver_features = ['followers', 'monthly_listeners', 'playlists_found', 'major_label']
```
Initial features selected on first dataset merge for analysis
```text
Comprehensive feature set 
extended_features = [
    # Audio features
    'track_genre', 'explicit', 'key', 'mode', 'time_signature', 
    'duration_ms', 'energy', 'speechiness', 'acousticness', 
    'instrumentalness', 'liveness', 'valence', 'tempo',
    'danceability', 'loudness',
    # Market Drivers
    'followers', 'monthly_listeners', 'playlists_found', 'major_label',
    # Engineered Features
    'duration_bucket', 'acoustic_energy_ratio'
]
```
4. Improved Merged Datasets for final prediction
* notebook/spotify_improved_final.ipynb
The following is the final datasets and dtype combined
```text
Data columns (total 26 columns):
 #   Column             Non-Null Count   Dtype  
---  ------             --------------   -----  
 0   Unnamed: 0         114000 non-null  int64  
 1   track_id           114000 non-null  object 
 2   artists            113999 non-null  object 
 3   album_name         113999 non-null  object 
 4   track_name         113999 non-null  object 
 5   popularity         114000 non-null  int64  
 6   duration_ms        114000 non-null  int64  
 7   explicit           114000 non-null  bool   
 8   danceability       114000 non-null  float64
 9   energy             114000 non-null  float64
 10  key                114000 non-null  int64  
 11  loudness           114000 non-null  float64
 12  mode               114000 non-null  int64  
 13  speechiness        114000 non-null  float64
 14  acousticness       114000 non-null  float64
 15  instrumentalness   114000 non-null  float64
 16  liveness           114000 non-null  float64
 17  valence            114000 non-null  float64
 18  tempo              114000 non-null  float64
 19  time_signature     114000 non-null  int64  
 20  track_genre        114000 non-null  object 
 21  artist_key         114000 non-null  object 
 22  followers          20437 non-null   float64
 23  monthly_listeners  20437 non-null   float64
 24  playlists_found    20437 non-null   object 
 25  popularity_market  20437 non-null   float64
dtypes: bool(1), float64(12), int64(6), object(7)
```
> Final features selected for datasets merge for prediction
```text
extended_features = [
    # Track acoustic qualities
    'track_genre', 'explicit', 'key', 'mode', 'time_signature', 
    'duration_ms', 'energy', 'speechiness', 'acousticness', 
    'instrumentalness', 'liveness', 'valence', 'tempo',
    'danceability', 'loudness',
    # Playlist exposure and flag
    'followers', 'monthly_listeners', 'playlists_found', 'major_label', 'has_market_data',
    # Extra engineered Features
    'duration_bucket', 'acoustic_energy_ratio'
]
```
---
##  Spotify initial data exploration and modelling
###  EDA summary
#### Exploratory Data Analysis for audio characteristics revealed several key patterns:
* **Genre Dominance:** Categorical variables like `track_genre` carry the strongest predictive weight; 
mainstream genres (Pop, K-Pop, Hip-Hop) skew heavily toward high popularity, whereas ambient or classical 
tracks cluster below the threshold.
* **Audio Correlation Limits:** Individual audio attributes (like `valence` or `tempo`) show surprisingly weak 
linear correlation with popularity on their own, proving that raw sound profiles require non-linear modeling and 
interaction features to capture predictive signals.
* **Scale Discrepancies:** Numerical features span wildly different ranges making feature scaling and dictionary-based vectorization necessary.
#### Modeling approach & metrics for Spotify tracks audio characteristics
* **Data Transformation:**
* `DictVectorizer` used to seamlessly encode mixed categorical data (`track_genre`, `duration_bucket`) and numerical attributes without manual one-hot encoding boilerplate.
* `StandardScaler` applied to prevent convergence failures in linear algorithms and add stability.
* **Models Evaluated:**
* *Baseline:* `LogisticRegression` (with solver convergence optimization via scaling and `max_iter=2000`).
* *Advanced:* `XGBClassifier` (gradient boosting ensemble to capture complex feature interactions).
* **Evaluation Metrics:** 
ROC-AUC, F1-Score, Precision, Recall, and Confusion Matrix visualization.
* **Artifact Packaging:** 
The trained model, vectorizer, feature list, and target definition are bundled into a single dictionary artifact 
and saved using `joblib` (`models/spotify_xgb_model.pkl`).
#### Known limitations / next steps
##### Limitations
* The model relies entirely on audio attributes and genre tags, omitting powerful real-world market drivers 
like artist historical fame, social media virality (e.g., TikTok trends), and major playlist curation.
* Binary thresholding (`Popularity >= 50`) discards granular ranking nuances.
##### Next steps
* Integrate artist-level metadata or historical artist popularity averages as a feature.
* Apply NLP techniques (TF-IDF or embeddings) on `track_name` and `album_name` to capture title-based keywords 
(e.g., "Remix", "feat.").
* Experiment with hyperparameter tuning via Optuna or GridSearchCV for XGBoost to further boost ROC-AUC scores.
---
##  Spotify final merge and data analysis used for prediction
### Exploratory Data Analysis for combined audio and track metadata revealed several key patterns:
###  Modeling approach & metrics
Algorithm: XGBoost Classifier (XGBClassifier), an optimized gradient-boosted decision tree ensemble known for handling tabular 
data with mixed feature types efficiently.
#### Target Definition: 
Binary Classification where tracks are labeled as 1 (High Popularity) if their popularity score  >= 50, and 0 otherwise.
#### Hyperparameters:
* n_estimators=150 (number of boosting rounds)
* learning_rate=0.05 (shrinkage to prevent overfitting)
* max_depth=6 (controls tree complexity)
### Known limitations / next steps
#### Limitations
* Pass or fail rule: Treating popularity as a simple pass/fail (>= 50) loses the nuance of exact rankings.
* Name Matching Errors: Minor spelling differences or featured artists break the artist name merge, leaving market data blank (0).
* New Artist Bias: Emerging artists with low followers or playlist counts struggle to be predicted as hits, even if the audio is great.
* Outdated or static data: The model relies on past data and cannot keep up with sudden viral trends or seasonal changes.
### #Next steps
In the future to deal with limitations maybe could:
* Move from Pass/Fail to Exact Scores: To predict the exact popularity score (0–100) instead of a binary pass/fail, 
swap out XGBClassifier for XGBRegressor where keep the raw numeric score (0–100) instead of converting it with >= 50
* Replace exact string matching on artist_key with unique Spotify Artist IDs or fuzzy map strings
*  Fix new artist bias: Build a secondary model trained strictly on audio features (tempo, energy, danceability, acoustic_energy_ratio)
for tracks where has_market_data == 0, preventing high follower counts from drowning out upcoming popular tracks
* Improve on the outdated data by constantly refreshing the data pipeline to capture viral and upcoming tracks
---
## System Architecture Overview
A lightweight, containerized Flask machine learning application featuring modular training (`train.py`), 
dedicated inference (`predict.py`), API routing (`serve.py`), and interactive Swagger documentation 
via Flasgger.

```text
┌────────────────────────────────────────────────────────┐
│                   Client / API User                    │
│      (Browser / Swagger UI at /apidocs / cURL)         │
└───────────────────────────┬────────────────────────────┘
                            │ HTTPS / REST Requests
                            ▼
┌────────────────────────────────────────────────────────┐
│               Render / Docker Runtime          │
│ ┌────────────────────────────────────────────────────┐ │
│ │                  Flask Server (serve.py)           │ │
│ │  - Exposes REST Endpoints (/health, /predict)      │ │
│ │  - Integrated Flasgger Swagger UI (/apidocs)       │ │
│ └─────────────────────────┬──────────────────────────┘ │
│                           │ Delegates Inference        │
│                           ▼                            │
│ ┌────────────────────────────────────────────────────┐ │
│ │             Prediction Engine (predict.py)         │ │
│ │  - Handles Feature Validation & Reshaping          │ │
│ │  - Executes scikit-learn Model Predictions         │ │
│ └─────────────────────────┬──────────────────────────┘ │
└───────────────────────────┼────────────────────────────┘
                            │ Loads on Startup
                            ▼
┌────────────────────────────────────────────────────────┐
│                  Model Artifact Store                  │
│               - models/model.pkl (Joblib)              │
│       (Generated via train.py & scikit-learn)          │
└────────────────────────────────────────────────────────┘
```
---
## Project structure Overview
train.py # trains model and saves artifact
predict.py # loads model, runs inference
serve.py # starts web service
Files with dependencies- requirements.txt 

```text
project/
├── data/                          # Contains all spotify related csv files used int he project
├── notebooks/                     # Contains all the notebooks for data analysis before use in web app
├── models/                         # Storage directory for trained model artifacts
│   └──model.pkl                   # Serialized ML model artifact generated by train.py
├── src/                            # Core application source code package
│   ├── __init__.py                 # Marks src as a Python package
│   ├── predict.py                  # Handles inference logic, feature validation, and model execution
│   ├── serve.py                    # Flask web server exposing REST endpoints and Swagger UI
│   └── train.py                    # Trains the machine learning model and saves artifact
├── .dockerignore                   # Specifies files/folders to exclude from the Docker build
├── Dockerfile                      # Instructions to containerize the app and bake in model training
├── README.md                       # Project documentation, system architecture, and setup guides
└── requirements.txt                # Python package dependencies (Flask, Flasgger, scikit-learn, etc.)
```
---
## Project setup
### Docker
### How to run locally and via Docker
$ docker build -t spotify-flask-api .
$ docker run -d -p 5000:5000 --name spotify-flask-container spotify-flask-api
## API usage example
> Implemented sawgger UI for interactive testing of the endpoint but can also use curl as seen in the documentation
http://localhost:8000/apidocs/
```text
curl.exe -X POST "http://localhost:5000/predict" `
  -H "accept: application/json" `
  -H "Content-Type: application/json" `
  -d '{"acousticness": 0.2, "artists": "Kina Grannis", "danceability": 0.7, 
  "duration_ms": 180000, "energy": 0.65, "explicit": 0, "followers": 500000, 
  "instrumentalness": 0, "key": 5, "liveness": 0.1, "loudness": -6, "mode": 1, 
  "monthly_listeners": 150000, "playlists_found": 10, "speechiness": 0.04, 
  "tempo": 120, "time_signature": 4, "track_genre": "pop", 
   "track_name": "Can'\''t Help Falling In Love", "valence": 0.55}'
```
---
## Cloud Deployment(Render)
Selected Render free-tier to deploy my lightweight flask-based web app and machine learning API mainly due cost
### Live API URL
[Spotify Track Popularity Predictor Live url_apidocs](https://spotifiy-track-popularity-prediction.onrender.com/apidocs/ )
> NB: Render's free tier API may take up a minute to respond on the first request, due to waking up from idle sleep
### Screenshots or video of deployment

---


