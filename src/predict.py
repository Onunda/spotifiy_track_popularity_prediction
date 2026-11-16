import joblib


class SongPopularityPredictor:
    def __init__(self, model_path="models/model.pkl"):
        artifact = joblib.load(model_path)
        self.dv = artifact["dv"]
        self.model = artifact["model"]
        self.features = artifact["features"]
        self.target_definition = artifact.get("target_definition", "High Popularity (Score >= 50)")

    def preprocess(self, data: dict) -> dict:
        data['has_market_data'] = 1 if data.get('followers', 0) > 0 else 0
        data['major_label'] = 1 if data.get('followers', 0) > 1000000 else 0

        duration_ms = data.get('duration_ms', 0)
        if duration_ms <= 120000:
            data['duration_bucket'] = 'short'
        elif duration_ms <= 240000:
            data['duration_bucket'] = 'medium'
        else:
            data['duration_bucket'] = 'long'

        acousticness = data.get('acousticness', 0.0)
        energy = data.get('energy', 0.0)
        data['acoustic_energy_ratio'] = acousticness / (energy + 1e-5)

        return {f: data.get(f, 0) for f in self.features}

    def predict(self, data: dict, threshold: float = 0.6) -> dict:
        processed_dict = self.preprocess(data)
        X_input = self.dv.transform([processed_dict])

        prob = float(self.model.predict_proba(X_input)[:, 1][0])
        prediction = int(prob >= threshold)

        return {
            "track_name": data.get("track_name", "Unknown"),
            "artists": data.get("artists", "Unknown"),
            "hit_probability": round(prob, 4),
            "predicted_high_popularity": prediction,
            "threshold_used": threshold,
            "status": "High Popularity Hit (Score >= 50)" if prediction == 1 else "Low/Indie Track"
        }