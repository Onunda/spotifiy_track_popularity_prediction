from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import Swagger
from werkzeug.middleware.proxy_fix import ProxyFix
from src.predict import SongPopularityPredictor

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
CORS(app)
swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "Spotify Track Popularity Predictor API",
        "description": (
            "A data science machine learning API powered by XGBoost, Flask, and Flasgger. "
            "This service evaluates track success to investigate two core research questions:\n\n"
            "1. **Audio-Only Predictability:** To what extent do audio features alone predict song popularity?\n"
            "2. **Acoustics vs. Market Reach:** Do market reach and playlist exposure outweigh acoustic qualities in driving track success?"
        ),
        "version": "1.0.0"
    },
    "schemes": ["https", "http"]
}
app.config['SWAGGER'] = {'title': 'Spotify Track Popularity Predictor API','uiversion': 3 }
swagger = Swagger(app, template=swagger_template)


try:
    predictor = SongPopularityPredictor(model_path="models/model.pkl")
except Exception as e:
    predictor = None

@app.route("/", methods=["GET"])
def home():
    """
    API Health Check & Info
    ---
    responses:
      200:
        description: Returns API status
    """
    if not predictor:
        return jsonify({"status": "error", "message": "Model artifact not loaded."}), 500
    return jsonify({
        "status": "online",
        "target_definition": predictor.target_definition,
        "message": "Send a POST request to /predict to score tracks."
    })

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict Spotify Track Popularity
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: TrackInput
          properties:
            track_name:
              type: string
              example: "Can't Help Falling In Love"
            artists:
              type: string
              example: "Kina Grannis"
            track_genre:
              type: string
              example: "pop"
            explicit:
              type: integer
              example: 0
            key:
              type: integer
              example: 5
            mode:
              type: integer
              example: 1
            time_signature:
              type: integer
              example: 4
            duration_ms:
              type: number
              example: 180000
            energy:
              type: number
              example: 0.65
            speechiness:
              type: number
              example: 0.04
            acousticness:
              type: number
              example: 0.20
            instrumentalness:
              type: number
              example: 0.0
            liveness:
              type: number
              example: 0.1
            valence:
              type: number
              example: 0.55
            tempo:
              type: number
              example: 120.0
            danceability:
              type: number
              example: 0.70
            loudness:
              type: number
              example: -6.0
            followers:
              type: number
              example: 500000
            monthly_listeners:
              type: number
              example: 150000
            playlists_found:
              type: number
              example: 10
    responses:
      200:
        description: Successful prediction output
    """
    if not predictor:
        return jsonify({"error": "Model predictor is not initialized."}), 500
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided."}), 400
        result = predictor.predict(data, threshold=0.6)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)