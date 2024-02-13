from flask import Flask, request, jsonify
from datetime import datetime as dt  
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS  

app = Flask(__name__)
CORS(app)  

# Load model
model_path = 'dp_model.h5' 
model = tf.keras.models.load_model(model_path)

# Preprocessing function
def preprocess_data(data_df):
    # Your preprocessing code here
    columns_to_drop = ['org_id', 'user_id', 'status_id', 'loan_id', 'work_start_date', 'work_email', 'loan_request_day',
                       'current_employer', 'work_email_validated', 'first_account', 'last_account', 'created_on',
                       'process_time', 'photo_url', 'logins']

    print("Columns in DataFrame:", data_df.columns)
    new_data = data_df.drop(columns=columns_to_drop)
    # new_data = new_data.dropna()
    # new_data = new_data[new_data['status_id'] != 1]

    new_data["requested_amount"] = new_data["requested_amount"].astype(int)

    columns_to_encode = ['gender', 'marital_status', 'type_of_residence', 'educational_attainment',
                         'sector_of_employment', 'monthly_net_income', 'country', 'city', 'lga', 'purpose',
                         'selfie_bvn_check', 'selfie_id_check', 'device_name', 'mobile_os', 'os_version',
                         'no_of_dependent', 'employment_status']

    label_encoder = LabelEncoder()
    for column in columns_to_encode:
        new_data[column] = label_encoder.fit_transform(new_data[column])

    def process_column(column):
        new_column = []
        for value in column:
            if value.endswith("days"):
                new_value = int(value[:-5])
            elif value.endswith("months"):
                new_value = int(value[:-6]) * 30
            elif value.endswith("weeks"):
                new_value = int(value[:-5]) * 7
            else:
                new_value = 1
            new_column.append(new_value)
        return new_column

    new_data['proposed_payday'] = process_column(new_data['proposed_payday'])

    return new_data

# Prediction route
@app.route('/predict', methods=['POST'])  
def predict():
    data = request.get_json()

    # Convert JSON data to DataFrame
    data_df = pd.DataFrame(data)
    
    # Preprocess
    preprocessed_data = preprocess_data(data_df)
    
    # Make predictions
    predictions = model.predict(preprocessed_data)

    # Round prediction 
    rounded_prediction = round(predictions[0])

    # Get default status
    default = rounded_prediction < 0.5
    status = 'Default' if default else 'Not Default'

    # Construct response
    result = {
        'prediction': predictions[0],
        'rounded': rounded_prediction, 
        'status': status
    }

    # Optional - add timestamp
    timestamp = dt.now().isoformat()  
    result['timestamp'] = timestamp

    return jsonify(result)  

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=True, port=os.getenv("PORT", default=5000))
