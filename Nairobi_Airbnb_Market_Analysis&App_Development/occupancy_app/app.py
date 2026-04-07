from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load saved model
try:
    model = joblib.load("rf_occupancy_pipeline.pkl")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# -------------------------------
# Routes
# -------------------------------

@app.route('/')
def home():
    return render_template('index.html', prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        data = {
            'photos_count': float(request.form.get('photos_count', 0)),
            'guests': float(request.form.get('guests', 0)),
            'bedrooms': float(request.form.get('bedrooms', 0)),
            'beds': float(request.form.get('beds', 0)),
            'baths': float(request.form.get('baths', 0)),
            'cleaning_fee': float(request.form.get('cleaning_fee', 0)),
            'extra_guest_fee': float(request.form.get('extra_guest_fee', 0)),
            'num_reviews_x': float(request.form.get('num_reviews_x', 0)),
            'rating_overall': float(request.form.get('rating_overall', 0)),
            'rating_accuracy': float(request.form.get('rating_accuracy', 0)),
            'rating_checkin': float(request.form.get('rating_checkin', 0)),
            'rating_cleanliness': float(request.form.get('rating_cleanliness', 0)),
            'rating_communication': float(request.form.get('rating_communication', 0)),
            'rating_location': float(request.form.get('rating_location', 0)),
            'rating_value': float(request.form.get('rating_value', 0)),
            'ttm_avg_rate': float(request.form.get('ttm_avg_rate', 0)),
            'ttm_available_days': float(request.form.get('ttm_available_days', 0)),
            'l90d_avg_rate': float(request.form.get('l90d_avg_rate', 0)),
            'l90d_available_days': float(request.form.get('l90d_available_days', 0)),
            'rate_avg': float(request.form.get('rate_avg', 0)),
            'booking_lead_time_avg': float(request.form.get('booking_lead_time_avg', 0)),
            'length_of_stay_avg': float(request.form.get('length_of_stay_avg', 0)),
            'min_nights_avg': float(request.form.get('min_nights_avg', 0)),
            'superhost': request.form.get('superhost', 'No'),
            'instant_book': request.form.get('instant_book', 'No'),
            'listing_type': request.form.get('listing_type', 'Entire home/apt'),
            'room_type': request.form.get('room_type', 'Entire home/apt'),
            'cancellation_policy': request.form.get('cancellation_policy', 'Flexible'),
            'day_of_week': request.form.get('day_of_week', 'Monday')
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Predict log(occupancy)
        y_pred_log = model.predict(input_df)

        # Back-transform and convert to percentage
        predicted_occupancy = np.expm1(y_pred_log)[0] * 100

        return render_template(
            'index.html',
            prediction_text=f"🎯 Predicted Occupancy: {predicted_occupancy:.2f}%"
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"❌ Error: {str(e)}"
        )

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)


    
