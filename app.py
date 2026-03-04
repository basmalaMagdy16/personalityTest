from flask import Flask, request, jsonify
from flask_cors import CORS  # تفعيل الاتصال مع الـ Frontend
import joblib
import pandas as pd
import json
import os

app = Flask(__name__)
CORS(app)  # السماح لأي Frontend بالوصول للـ API

# 1. تعريف مسارات الملفات (تأكدي من وجودها في نفس الفولدر على GitHub)
MODEL_PATH = "mbti_model.joblib"
ENCODER_PATH = "label_encoder.joblib"
FEATURES_PATH = "selected_features.json"
DICT_PATH = "dictionary.json"

# 2. تحميل الموديل والملفات المساعدة عند بدء التشغيل
try:
    # تحميل الموديل والإنكودر باستخدام joblib لضمان التوافق مع الصيغة الجديدة
    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    
    with open(FEATURES_PATH, 'r') as f:
        feature_names = json.load(f)

    with open(DICT_PATH, 'r') as f:
        mbti_dict = json.load(f)
        
    print("🚀 CareerPro: All models and files loaded successfully!")
except Exception as e:
    print(f"❌ Error loading files: {str(e)}")

@app.route('/')
def home():
    return "CareerPro AI API is Running Successfully!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # استقبال الإجابات (المتوقع قائمة بـ 30 إجابة من -3 لـ +3)
        data = request.get_json()
        answers = data.get('answers')

        if not answers or len(answers) != len(feature_names):
            return jsonify({"error": f"Please provide exactly {len(feature_names)} answers."}), 400

        # تحويل البيانات لـ DataFrame متوافق مع الموديل
        input_df = pd.DataFrame([answers], columns=feature_names)

        # تنفيذ عملية التوقع
        pred_idx = model.predict(input_df)[0]
        predicted_type = le.inverse_transform([pred_idx])[0]

        # جلب البيانات من الديكشنري
        info = mbti_dict.get(predicted_type, {})
        
        return jsonify({
            "personality": predicted_type,
            "title": info.get('title', 'Unknown'),
            "description": info.get('description', ''),
            "suggestedCareers": info.get('suggestedCareers', [])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # إعداد البورت ليتوافق مع بيئة Azure
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)