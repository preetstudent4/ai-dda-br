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

    # print("Player data:", player_data)
    # print("DBNOs: ", player_data.get("DBNOs"))
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
# Batch Prediction Endpoint
# -----------------------------

@app.route("/batch_predict", methods=["POST"])
def batch_predict():

    try:

        players = request.json.get("players", [])

        results = []

        for player_data in players:

            player_data = compute_features(player_data)

            features_df = pd.DataFrame([player_data])
            features_df = features_df[feature_names]

            features_scaled = scaler.transform(features_df)

            prediction = model.predict(features_scaled)[0]

            skill_level, skill_score = classify_skill_level(prediction)

            adjustments = get_difficulty_adjustment(skill_level)

            results.append({
                "predictedWinPlacePerc": float(prediction),
                "skillLevel": skill_level,
                "skillScore": skill_score,
                "difficultyAdjustments": adjustments
            })

        return jsonify({
            "success": True,
            "results": results
        }), 200

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
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
# from flask import Flask, request, jsonify
# import joblib
# import numpy as np
# import pandas as pd
# from flask_cors import CORS
# import pickle
# app = Flask(__name__)
# CORS(app) # Enable CORS for Unity requests
#
# # Loading trained model and scaler at startup
# model = joblib.load('rf_difficulty_model.pkl')
# scaler = joblib.load('feature_scaler.pkl')
#
# # Load feature names
# with open('selected_features.pkl', 'rb') as f:
#     feature_names = pickle.load(f)
#
# def classify_skill_level(win_place_perc):
#     """Classify player skill based on predicted win placement"""
#     if win_place_perc < 0.25:
#         return "Beginner", 1
#     elif win_place_perc < 0.50:
#         return "Intermediate", 2
#     elif win_place_perc < 0.75:
#         return "Advanced", 3
#     else:
#         return "Expert", 4
#
# def get_difficulty_adjustment(skill_level):
#     """Return difficulty parameters for Unity"""
#     adjustments = {
#         "Beginner": {
#             "enemyAccuracy": 0.6,
#             "enemyHealth": 0.8,
#             "enemyDamage": 0.7,
#             "enemyCount": 0.8,
#             "lootQuality": 1.3,
#             "lootSpawnRate": 1.4,
#             "aimAssist": 1.5,
#             "playerDamageMultiplier": 1.2,
#             "respawnTime": 0.7
#         },
#         "Intermediate": {
#             "enemyAccuracy": 0.85,
#             "enemyHealth": 1.0,
#             "enemyDamage": 1.0,
#             "enemyCount": 1.0,
#             "lootQuality": 1.1,
#             "lootSpawnRate": 1.1,
#             "aimAssist": 1.0,
#             "playerDamageMultiplier": 1.0,
#             "respawnTime": 1.0
#         },
#         "Advanced": {
#             "enemyAccuracy": 1.0,
#             "enemyHealth": 1.2,
#             "enemyDamage": 1.15,
#             "enemyCount": 1.2,
#             "lootQuality": 1.0,
#             "lootSpawnRate": 0.9,
#             "aimAssist": 0.8,
#             "playerDamageMultiplier": 0.9,
#             "respawnTime": 1.2
#         },
#         "Expert": {
#             "enemyAccuracy": 1.2,
#             "enemyHealth": 1.5,
#             "enemyDamage": 1.3,
#             "enemyCount": 1.5,
#             "lootQuality": 0.9,
#             "lootSpawnRate": 0.8,
#             "aimAssist": 0.5,
#             "playerDamageMultiplier": 0.8,
#             "respawnTime": 1.5
#         }
#     }
#     return adjustments[skill_level]
#
# @app.route('/health', methods=['GET'])
# def health_check():
#     """Check if API is running"""
#     return jsonify({"status": "healthy", "message": "DDA API is running"}), 200
#
# @app.route('/predict', methods=['POST'])
# def predict_difficulty():
#     """
#     Receive player stats from Unity and return difficulty adjustments
#
#     Expected JSON format:
#     {
#         "assists:" 2,
#         "boosts:" 3,
#         "damageDealt": 450.5,
#         "DBONs": 3,
#         "headshotKills": 1,
#         "kills": 3,
#         "killPoints": 1200
#         "winPoints": 1500,
#     }
#     """
#
#     print("Headers:", request.headers)
#     print("Raw data:", request.data)
#     print("JSON parsed:", request.get_json())
#
#     data = request.get_json()
#
#     if not data:
#         return jsonify({"error": "Invalid or missing JSON"}), 400
#
#     # return jsonify({"success": True})
#
#     try:
#         # Get player data from request
#         player_data = data
#
#         # Calculate engineered features (same as training pipeline)
#         kills = player_data.get("kills", 0)
#         damage = player_data.get("damageDealt", 0)
#         player_data['kill_efficiency'] = player_data['kills'] / (player_data['damageDealt'] + 1)
#         player_data['headshot_rate'] = player_data['headshotKills'] / (player_data['kills'] + 1)
#         player_data['assist_ratio'] = player_data['assists'] / (player_data['kills'] + player_data['assists'] + 1)
#         player_data['DBNO_conversion'] = player_data['kills'] / (player_data['DBNOs'] + 1)
#         player_data['boost_efficiency'] = (player_data['kills'] + player_data['assists']) / (player_data['boosts'] + 1)
#         player_data['combat_score'] = player_data['kills'] + player_data['assists'] + (player_data['damageDealt'] / 100)
#
#         # Create DataFrame with proper feature order
#         features_df = pd.DataFrame([player_data])
#         features_df = features_df[feature_names]
#
#         # Scale features
#         features_scaled = scaler.transform(features_df)
#
#         # Make prediction
#         prediction = model.predict(features_scaled)[0]
#
#         # Classify skill level
#         skill_level, skill_score = classify_skill_level(prediction)
#
#         # Get difficulty adjustments
#         adjustments = get_difficulty_adjustment(skill_level)
#
#         # Return response
#         response = {
#             "success": True,
#             "predictedWinPlacePerc": float(prediction),
#             "skillLevel": skill_level,
#             "skillScore": skill_score,
#             "difficultydjustments": adjustments,
#             "message": f"Player classified as {skill_level}"
#         }
#
#         return jsonify(response), 200
#
#     except Exception as e:
#         return jsonify({
#             "success": False,
#             "error": str(e),
#             "message": "Error processing player data"
#         }), 400
#
# @app.route('/batch_predict', methods=['POST'])
# def batch_predict():
#     """Handle multiple player predictions at once"""
#     try:
#         players_data = request.json['players']
#         results = []
#
#         for player_data in players_data:
#             # Calculate engineered features (same as training pipeline)
#             player_data['kill_efficiency'] = player_data['kills'] / (player_data['damageDealt'] + 1)
#             player_data['headshot_rate'] = player_data['headshotKills'] / (player_data['kills'] + 1)
#             player_data['assist_ratio'] = player_data['assists'] / (player_data['kills'] + player_data['assists'] + 1)
#             player_data['DBNO_conversion'] = player_data['kills'] / (player_data['DBNOs'] + 1)
#             player_data['boost_efficiency'] = (player_data['kills'] + player_data['assists']) / (
#                         player_data['boosts'] + 1)
#             player_data['combat_score'] = player_data['kills'] + player_data['assists'] + (
#                         player_data['damageDealt'] / 100)
#
#             # Create DataFrame with proper feature order
#             features_df = pd.DataFrame([player_data])
#             features_df = features_df[feature_names]
#
#             # Scale features
#             features_scaled = scaler.transform(features_df)
#
#             # Make prediction
#             prediction = model.predict(features_scaled)[0]
#
#             # Classify skill level
#             skill_level, skill_score = classify_skill_level(prediction)
#
#             # Get difficulty adjustments
#             adjustments = get_difficulty_adjustment(skill_level)
#
#             # Return response
#             response = {
#                 "success": True,
#                 "predictedWinPlacePerc": float(prediction),
#                 "skillLevel": skill_level,
#                 "skillScore": skill_score,
#                 "difficultyAdjustments": adjustments,
#                 "message": f"Player classified as {skill_level}"
#             }
#             results.append(response)
#
#         return jsonify({"success": True, "results": results}), 200
#
#     except Exception as e:
#         return jsonify({"success": False, "error": str(e)}), 400
#
# if __name__ == '__main__':
#     print("Starting Dynamic Difficulty Adjustment API...")
#     print(f"Model loaded with {len(feature_names)} features")
#     print("API running on http://localhost:5000")
#     app.run(host='0.0.0.0', port=5000, debug=True)
#
#
#
#
#
#
#
#
#
