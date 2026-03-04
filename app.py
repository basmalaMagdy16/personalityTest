from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import json
import os
from collections import OrderedDict

app = Flask(__name__)
# السطر ده هو السر اللي هيخلي الترتيب يظهر للباك إند زي ما إنتي عايزة بالظبط
app.config['JSON_SORT_KEYS'] = False 
CORS(app)

MODEL_PATH = "mbti_model.joblib"
ENCODER_PATH = "encoder.joblib"        
FEATURES_PATH = "selected features.json"
DICT_PATH = "dictionary.json"

model = None
le = None
feature_names = None
mbti_dict = None

try:
    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    with open(FEATURES_PATH, 'r') as f:
        feature_names = json.load(f)
    with open(DICT_PATH, 'r') as f:
        mbti_dict = json.load(f)
except Exception as e:
    pass

@app.route('/')
def home():
    if model is None or feature_names is None:
        return "API is Running, but files failed to load."
    return "Personality Test AI API is Running Successfully!"

@app.route('/predict', methods=['POST'])
def predict():
    if feature_names is None or model is None:
        return jsonify({"error": "Model files not loaded on server."}), 500

    try:
        data = request.get_json()
        answers = data.get('answers')

        if not answers or len(answers) != len(feature_names):
            return jsonify({"error": f"Expected {len(feature_names)} answers."}), 400

        input_df = pd.DataFrame([answers], columns=feature_names)
        pred_idx = model.predict(input_df)[0]
        predicted_type = le.inverse_transform([pred_idx])[0]
        info = mbti_dict.get(predicted_type, {})
        
        response_data = OrderedDict([
            ("personality", predicted_type),
            ("title", info.get('title', 'Unknown')),
            ("description", info.get('description', '')),
            ("suggestedCareers", info.get('suggestedCareers', []))
        ])
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
