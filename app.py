from flask import Flask, request, jsonify
from flask_cors import CORS
from catboost import CatBoostClassifier
import numpy as np
import os

# ---------- CREATE APP ----------
app = Flask(__name__)
CORS(app)

# ---------- LOAD MODEL ----------
model = CatBoostClassifier()
MODEL_PATH = os.path.join(os.path.dirname(__file__), "catboost_model.cbm")
model.load_model(MODEL_PATH)

print("Model loaded successfully!")

# ---------- HOME ROUTE ----------
@app.route('/')
def home():
    return "Backend is running with model!"

# ---------- PREDICTION ROUTE ----------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # ---------- INPUT ----------
        total_time = float(data['total_time'])
        session_count = float(data['session_count'])
        micro_ratio = float(data['micro_ratio'])
        late_night_ratio = float(data['late_night_ratio'])
        sessions_per_hour = float(data['sessions_per_hour'])

        # ---------- MODEL INPUT ----------
        input_data = np.array([[total_time, session_count, micro_ratio, late_night_ratio, sessions_per_hour]])

        # ---------- MODEL PREDICTION (FOR ANALYSIS ONLY) ----------
        model_pred = int(model.predict(input_data)[0])

        # ---------- RISK SCORE ----------
        risk_score = (
            total_time +
            session_count +
            micro_ratio +
            late_night_ratio +
            sessions_per_hour
        ) / 5

        risk_score = round(risk_score, 2)

        # ---------- FINAL CLASSIFICATION (PURE LOGIC) ----------
        if risk_score < 0.5:
            final_result = "Not Addicted"

        elif 0.5 <= risk_score < 0.75:
            final_result = "Mildly Addicted"

        else:
            final_result = "Addicted"

        # ---------- SMART SUGGESTIONS ----------
        suggestions = []

        # 🔴 ADDICTED
        if final_result == "Addicted":
            suggestions.append("Your usage pattern indicates high dependency on your phone.")

            if total_time > 0.7:
                suggestions.append("Gradually reduce your daily screen time by setting usage limits.")

            if session_count > 0.7:
                suggestions.append("Avoid frequent phone checking. Try fixed intervals instead.")

            if micro_ratio > 0.6:
                suggestions.append("Disable unnecessary notifications to reduce impulsive checking.")

            if late_night_ratio > 0.5:
                suggestions.append("Avoid late-night phone usage to improve sleep quality.")

            if sessions_per_hour > 0.7:
                suggestions.append("Keep your phone away during work/study to improve focus.")

            suggestions.append("Consider digital detox routines like no-phone hours.")

        # 🟡 MILDLY ADDICTED
        elif final_result == "Mildly Addicted":
            suggestions.append("Your usage shows early signs of dependency.")

            if total_time > 0.5:
                suggestions.append("Try reducing your screen time gradually.")

            if session_count > 0.5:
                suggestions.append("Reduce how often you check your phone.")

            if micro_ratio > 0.4:
                suggestions.append("Limit short, impulsive phone usage.")

            if late_night_ratio > 0.3:
                suggestions.append("Avoid using your phone late at night.")

            if sessions_per_hour > 0.5:
                suggestions.append("Improve focus by minimizing interruptions.")

            suggestions.append("Build healthier digital habits before it worsens.")

        # 🟢 NOT ADDICTED
        else:
            suggestions.append("Great job! Your phone usage is under control.")

            if total_time > 0.4:
                suggestions.append("Try to keep your screen time within healthy limits.")

            if session_count > 0.4:
                suggestions.append("Avoid increasing phone checking frequency.")

            if micro_ratio > 0.4:
                suggestions.append("Reduce unnecessary quick checks.")

            if late_night_ratio > 0.2:
                suggestions.append("Maintain good sleep habits by limiting night usage.")

            if sessions_per_hour > 0.4:
                suggestions.append("Stay focused and avoid distractions.")

        # ---------- RESPONSE ----------
        return jsonify({
            "prediction": final_result,
            "risk_score": risk_score,
            "model_prediction": model_pred,
            "suggestions": suggestions
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


# ---------- RUN ----------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
