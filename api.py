from flask import Flask, request, jsonify
import joblib
import json
import pandas as pd

app = Flask(__name__)

# Load model and encoders once when server starts
model = joblib.load('churn_model.pkl')
encoders = joblib.load('encoders.pkl')

with open('window_lookup.json') as f:
    windows = json.load(f)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    required = ['customer_state', 'first_order_spend', 'items_in_order',
                'payment_type', 'delivery_delay_days', 'first_order_category',
                'review_score', 'days_since_purchase']

    for field in required:
        if field not in data:
            return jsonify({'error': f'Missing field: {field}'}), 400

    expected_window = windows['lookup'].get(
        data['first_order_category'],
        windows['global_avg']
    )
    recency_ratio = data['days_since_purchase'] / expected_window

    input_df = pd.DataFrame([{
        'customer_state': data['customer_state'],
        'first_order_spend': data['first_order_spend'],
        'items_in_order': data['items_in_order'],
        'payment_type': data['payment_type'],
        'delivery_delay_days': data['delivery_delay_days'],
        'first_order_category': data['first_order_category'],
        'review_score': data['review_score'],
        'recency_ratio': recency_ratio
    }])

    for col in ['customer_state', 'payment_type', 'first_order_category']:
        try:
            input_df[col] = encoders[col].transform(input_df[col])
        except ValueError:
            input_df[col] = 0

    prob = float(model.predict_proba(input_df)[0][1])

    return jsonify({
        'return_probability': round(prob, 4),
        'return_probability_pct': f"{prob:.1%}",
        'expected_repurchase_window_days': round(expected_window, 0),
        'recency_ratio': round(recency_ratio, 2),
        'recommendation': (
            'High priority — target with retention campaign' if prob > 0.15
            else 'Moderate — consider low cost nudge' if prob > 0.07
            else 'Low priority — below base rate'
        )
    })

if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug=False, port=5000)