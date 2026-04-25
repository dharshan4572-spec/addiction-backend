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

        # ---------- MODEL PREDICTION ----------
        prediction = model.predict(input_data)[0]

        # ---------- BEHAVIOR RISK SCORE ----------
        risk_score = (
            total_time +
            session_count +
            micro_ratio +
            late_night_ratio +
            sessions_per_hour
        ) / 5

        # ---------- FINAL CLASSIFICATION ----------
        if prediction == 1:
            final_result = "Addicted"
        else:
            if risk_score >= 0.5:
                final_result = "Mildly Addicted"
            else:
                final_result = "Not Addicted"

        # ---------- SUGGESTIONS ----------
        suggestions = []

        if final_result == "Addicted":
            if total_time > 0.7:
                suggestions.append("Reduce your overall screen time gradually.")

            if session_count > 0.7:
                suggestions.append("Avoid frequent phone checking. Try scheduled usage.")

            if micro_ratio > 0.6:
                suggestions.append("Limit short impulsive phone checks. Disable unnecessary notifications.")

            if late_night_ratio > 0.5:
                suggestions.append("Avoid using your phone late at night to improve sleep.")

            if sessions_per_hour > 0.7:
                suggestions.append("Increase focus time by keeping your phone away during work or study.")

        elif final_result == "Mildly Addicted":
            suggestions = [
                "You show some signs of excessive phone usage.",
                "Try reducing screen time gradually.",
                "Avoid frequent checking habits.",
                "Maintain better sleep habits by limiting night usage."
            ]

        else:
            suggestions = [
                "Great job maintaining healthy phone usage!",
                "Continue your balanced usage habits.",
                "Avoid increasing late-night usage."
            ]

        # ---------- RESPONSE ----------
        return jsonify({
            "prediction": final_result,
            "risk_score": round(risk_score, 2),
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