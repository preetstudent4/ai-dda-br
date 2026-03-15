import os

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)

# -----------------------------
# Load Model & Preprocessing
# -----------------------------

print("Loading ML model and scaler...")

model = joblib.load("rf_difficulty_model.pkl")
scaler = joblib.load("feature_scaler.pkl")

with open("selected_features.pkl", "rb") as f:
    feature_names = pickle.load(f)

print(f"Model loaded with {len(feature_names)} features")


# -----------------------------
# Skill Classification
# -----------------------------

def classify_skill_level(win_place_perc):
    """Convert predicted win placement to skill level"""

    if win_place_perc < 0.25:
        return "Beginner", 1
    elif win_place_perc < 0.50:
        return "Intermediate", 2
    elif win_place_perc < 0.75:
        return "Advanced", 3
    else:
        return "Expert", 4


# -----------------------------
# Difficulty Adjustment Rules
# -----------------------------

def get_difficulty_adjustment(skill_level):
    """Return gameplay adjustments for Unity"""

    adjustments = {

        "Beginner": {
            "enemyAccuracy": 0.6,
            "enemyHealth": 0.8,
            "enemyDamage": 0.7,
            "enemyCount": 0.8,
            "lootQuality": 1.3,
            "lootSpawnRate": 1.4,
            "aimAssist": 1.5,
            "playerDamageMultiplier": 1.2,
            "respawnTime": 0.7
        },

        "Intermediate": {
            "enemyAccuracy": 0.85,
            "enemyHealth": 1.0,
            "enemyDamage": 1.0,
            "enemyCount": 1.0,
            "lootQuality": 1.1,
            "lootSpawnRate": 1.1,
            "aimAssist": 1.0,
            "playerDamageMultiplier": 1.0,
            "respawnTime": 1.0
        },

        "Advanced": {
            "enemyAccuracy": 1.0,
            "enemyHealth": 1.2,
            "enemyDamage": 1.15,
            "enemyCount": 1.2,
            "lootQuality": 1.0,
            "lootSpawnRate": 0.9,
            "aimAssist": 0.8,
            "playerDamageMultiplier": 0.9,
            "respawnTime": 1.2
        },

        "Expert": {
            "enemyAccuracy": 1.2,
            "enemyHealth": 1.5,
            "enemyDamage": 1.3,
            "enemyCount": 1.5,
            "lootQuality": 0.9,
            "lootSpawnRate": 0.8,
            "aimAssist": 0.5,
            "playerDamageMultiplier": 0.8,
            "respawnTime": 1.5
        }
    }

    return adjustments.get(skill_level, adjustments["Intermediate"])


# -----------------------------
# Feature Engineering
# -----------------------------

def compute_features(player_data):
    """Generate ML features from raw player stats"""

    kills = player_data.get("kills", 0)
    assists = player_data.get("assists", 0)
    damage = player_data.get("damageDealt", 0)
    boosts = player_data.get("boosts", 0)
    DBNOs = player_data.get("DBNOs", 0)
    headshots = player_data.get("headshotKills", 0)

    player_data["kill_efficiency"] = kills / (damage + 1)
    player_data["headshot_rate"] = headshots / (kills + 1)
    player_data["assist_ratio"] = assists / (kills + assists + 1)
    player_data["DBNO_conversion"] = kills / (DBNOs + 1)
    player_data["boost_efficiency"] = (kills + assists) / (boosts + 1)
    player_data["combat_score"] = kills + assists + (damage / 100)

    return player_data


# -----------------------------
# Health Check Endpoint
# -----------------------------

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "DDA API is running"
    }), 200


# -----------------------------
# Single Prediction Endpoint
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict_difficulty():

    try:

        data = request.get_json()
        # print("Line 149: ", data)
        if not data:
            return jsonify({
                "success": False,
                "error": "Invalid or missing JSON"
            }), 400

        print("Incoming player data:", data)

        player_data = compute_features(data)
        # print(player_data)

        features_df = pd.DataFrame([player_data])
        features_df = features_df[feature_names]

        features_scaled = scaler.transform(features_df)

        prediction = model.predict(features_scaled)[0]

        skill_level, skill_score = classify_skill_level(prediction)

        adjustments = get_difficulty_adjustment(skill_level)

        response = {
            "success": True,
            "predictedWinPlacePerc": float(prediction),
            "skillLevel": skill_level,
            "skillScore": skill_score,
            "difficultyAdjustments": adjustments
        }

        return jsonify(response), 200

    except Exception as e:

        print("Prediction error:", str(e))

        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Error processing player data"
        }), 500

# -----------------------------
# Run Server
# -----------------------------

if __name__ == "__main__":

    print("Starting Dynamic Difficulty Adjustment API...")
    print("API running on http://localhost:5000")
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )
