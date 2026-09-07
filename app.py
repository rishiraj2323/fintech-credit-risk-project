from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

model = joblib.load('fintech_credit_model.pkl')
scaler = joblib.load('fintech_scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    features_scaled = scaler.transform(features)
    probability = model.predict_proba(features_scaled)[0][1]
    prediction = int(probability >= 0.25)
    return jsonify({
        'default_probability': float(probability),
        'prediction': prediction
    })

if __name__ == '__main__':
    app.run(debug=True)