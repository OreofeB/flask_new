from flask import Flask, request, jsonify
from datetime import datetime as dt  
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from flask_cors import CORS  
import os

app = Flask(__name__)
CORS(app)  

# Load model
model_path_1 = 'dp_model_6.h5' 
model_1 = tf.keras.models.load_model(model_path_1)

model_path_2 = 'dp_model_7.h5' 
model_2 = tf.keras.models.load_model(model_path_2)

model_path_3 = 'dp_model_3.h5' 
model_3 = tf.keras.models.load_model(model_path_3)


# Preprocessing function
def preprocess_data(data_df):
    # Your preprocessing code here
    columns_to_drop = ['org_id', 'user_id', 'status_id', 'loan_id', 'work_start_date', 'work_email', 'loan_request_day',
                       'current_employer', 'work_email_validated', 'first_account', 'last_account', 'created_on',
                       'process_time', 'photo_url', 'logins']

    print("Columns in DataFrame:", data_df.columns)
    new_data = data_df.drop(columns=columns_to_drop)

    remaining_fields = ['no_of_dependent', 'requested_amount']
    
    new_data[remaining_fields] = new_data[remaining_fields].astype(int)

    columns_to_encode = ['gender', 'marital_status', 'type_of_residence', 'educational_attainment',
                         'sector_of_employment', 'monthly_net_income', 'country', 'city', 'lga', 'purpose',
                         'selfie_bvn_check', 'selfie_id_check', 'device_name', 'mobile_os', 'os_version',
                         'no_of_dependent', 'employment_status','phone_network','bank']

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

@app.route('/index')
def index():
    result = {'Date': 1, 'Loan ID': 1, 'prediction': 1, 'prediction (%)': 1, 'rounded prediction': 1, 'status': 'Not Default'}
    return jsonify(result)

# Prediction route
@app.route('/predict', methods=['POST'])  
def predict():
    data = request.get_json()

    # Convert JSON data to DataFrame
    data_df = pd.DataFrame(data)
    
    # Preprocess
    preprocessed_data = preprocess_data(data_df)
    sc = StandardScaler()
    preprocessed_data = sc.fit_transform(preprocessed_data)
    loan_id = data_df['loan_id']
    loan_id = int(loan_id[0])
    
    # Make predictions using the model
    predictions_1 = model_1.predict(preprocessed_data)
    predictions_2 = model_2.predict(preprocessed_data)
    predictions_3 = model_3.predict(preprocessed_data)


    # Adjust the prediction output format and Round the prediction to 0 or 1
    r_prediction_1 = rounded_prediction_1 = np.round(float(predictions_1[0]), decimals=6)
    predictions_percentage_1 = predictions_1 * 100
    r_percentage_1 = rounded_percentage_1 = np.round(float(predictions_percentage_1), decimals=2)
    
    r_prediction_2 = rounded_prediction_2 = np.round(float(predictions_2[0]), decimals=6)
    predictions_percentage_2 = predictions_2 * 100
    r_percentage_2 = rounded_percentage_2 = np.round(float(predictions_percentage_2), decimals=2)
    
    r_prediction_3 = rounded_prediction_3 = np.round(float(predictions_3[0]), decimals=6)
    predictions_percentage_3 = predictions_3 * 100
    r_percentage_3 = rounded_percentage_3 = np.round(float(predictions_percentage_3), decimals=2)
    
    # Timestamp
    timestamp = dt.now().strftime("%Y-%m-%d %H:%M:%S")


    # Determine the result based on the rounded prediction
    result = {'Date': timestamp 
              ,'Loan ID': loan_id
              ,'Prediction 1': (r_prediction_1)
              ,'Prediction 1 (%)': (r_percentage_1)
              ,'Prediction 2': (r_prediction_2)
              ,'Prediction 2 (%)': (r_percentage_2)
              ,'Prediction 3': (r_prediction_3)
              ,'Prediction 3 (%)': (r_percentage_3)}

    return jsonify(result)

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=True, port=os.getenv("PORT", default=5000))
