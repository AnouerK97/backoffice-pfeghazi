import os
import warnings
from flask import Flask, request, jsonify
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from flask import request
from dataProcessing import *

warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)

##########################################################################################
################################### Model Version 1.0 ####################################
##########################################################################################
# Read data from CSV file using Pandas
df_v1_0 = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "data-v1.0.csv")))
# Convert the Pandas DataFrame to a list of dictionaries
data_v1_0 = df_v1_0.to_dict(orient='records')

@app.route('/api/model/v1', methods=['POST'])
def model_v1_0():
    try:
        data = request.get_json()
        print("Received JSON data:", data)

        categorical_columns = {
            'BusinessTravel': ['Never', 'Rarely', 'Frequently'],
            'Department': ['DealAdvisory', 'Audit', 'ITS', 'Marketing', 'HR', 'TECH', 'ITA', 'Legal', 'Consulting', 'Sales', 'TAX', 'AAS', 'Payrol'],
            'EducationField': ['Finance', 'Economie', 'Gestion', 'Informatique', 'HR', 'Comptabilite', 'Marketing', 'Audit', 'DataScience', 'Management', 'Droit', 'Fiscalite', 'Administration'],
            'JobRole': ['Junior', 'Senior', 'Manager', 'Director', 'Partner'],
            'MaritalStatus': ['Married', 'Single', 'Divorced']
        }

        df = preprocess_data_v1_0(data, categorical_columns)

        model = load_model("v1.0")

        prediction = model.predict_proba(df.iloc[0:1])
        predicted_probability = prediction[0][1]
        predicted_percentage = round(predicted_probability * 100, 2)

        data = {
            'prediction': predicted_percentage,
        }
        return jsonify(data)

    except Exception as e:
        print(e)
        return jsonify({"error": "Invalid JSON data"}), 400


@app.route('/api/data/v1', methods=['GET'])
def get_data():
    try:
        # Get the filter parameters from the query string
        attrition_filter = request.args.get('Attrition')
        department_filter = request.args.get('Department')
        job_role_filter = request.args.get('JobRole')

        # Apply filters based on the query parameters
        filtered_data = data_v1_0  # Assuming data_v1_0 is your original data

        if attrition_filter:
            filtered_data = [item for item in filtered_data if item.get('Attrition') == attrition_filter]
        if department_filter:
            filtered_data = [item for item in filtered_data if item.get('Department') == department_filter]
        if job_role_filter:
            filtered_data = [item for item in filtered_data if item.get('JobRole') == job_role_filter]

        return jsonify({"data": filtered_data})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/api/stats/v1/<string:column>/<int:bins>', methods=['GET'])
def get_stats_plot_v_1_0(column,bins):
    try:
        # Get the filter parameters from the query string
        attrition_filter = request.args.get('Attrition')
        department_filter = request.args.get('Department')
        job_role_filter = request.args.get('JobRole')

        # Apply filters based on the query parameters
        filtered_data = df_v1_0  # Assuming df_v1_0 is your original DataFrame

        if attrition_filter:
            filtered_data = filtered_data[filtered_data['Attrition'] == attrition_filter]
        if department_filter:
            filtered_data = filtered_data[filtered_data['Department'] == department_filter]
        if job_role_filter:
            filtered_data = filtered_data[filtered_data['JobRole'] == job_role_filter]

        # Create a histogram of the "Age" column
        counts, bins, _ = plt.hist(filtered_data[column], bins=bins, edgecolor='black')  # You can adjust the number of bins as needed

        # Prepare the data in a JSON-friendly format
        histogram_data = {
            "labels": [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins) - 1)],
            "data": counts.tolist()
        }

        return jsonify({"stats": histogram_data})
    except Exception as e:
        return jsonify({"error": str(e)})



##########################################################################################
################################### Model Version 2.0 ####################################
##########################################################################################

# Read data from CSV file using Pandas
df_v2_0 = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "data-v2.0.csv")))
# Convert the Pandas DataFrame to a list of dictionaries
data_v2_0 = df_v2_0.to_dict(orient='records')

@app.route('/api/model/v2', methods=['POST'])
def model_v2_0():
    try:
        data = request.get_json()
        print("Received JSON data:", data)

        categorical_columns = {
            'BusinessTravel': ['Never', 'Rarely', 'Frequently'],
            'Department': ['DealAdvisory', 'Audit', 'ITS', 'Marketing', 'HR', 'TECH', 'ITA', 'Legal', 'Consulting', 'Sales', 'TAX', 'AAS', 'Payrol'],
            'EducationField': ['Finance', 'Economie', 'Gestion', 'Informatique', 'HR', 'Comptabilite', 'Marketing', 'Audit', 'DataScience', 'Management', 'Droit', 'Fiscalite', 'Administration'],
            'JobRole': ['Junior', 'Senior', 'Manager', 'Director', 'Partner'],
            'MaritalStatus': ['Married', 'Single', 'Divorced']
        }

        df = preprocess_data_v1_0(data, categorical_columns)

        model = load_model("v2.0")

        prediction = model.predict_proba(df.iloc[0:1])
        predicted_probability = prediction[0][1]
        predicted_percentage = round(predicted_probability * 100, 2)

        data = {
            'prediction': predicted_percentage,
        }
        return jsonify(data)

    except Exception as e:
        print(e)
        return jsonify({"error": "Invalid JSON data"}), 400


##########################################################################################
################################### Model Version 3.0 ####################################
##########################################################################################


# Read data from CSV file using Pandas
df_v3_0 = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "data-v3.0.csv")))
# Convert the Pandas DataFrame to a list of dictionaries
data_v3_0 = df_v3_0.to_dict(orient='records')

@app.route('/api/model/v3', methods=['POST'])
def model_v3_0():
    try:
        data = request.get_json()
        print("Received JSON data:", data)

        categorical_columns = {
            'BusinessTravel': ['Never', 'Rarely', 'Frequently'],
            'Department': ['DealAdvisory', 'Audit', 'ITS', 'Marketing', 'HR', 'TECH', 'ITA', 'Legal', 'Consulting', 'Sales', 'TAX', 'AAS', 'Payrol'],
            'EducationField': ['Finance', 'Economie', 'Gestion', 'Informatique', 'HR', 'Comptabilite', 'Marketing', 'Audit', 'DataScience', 'Management', 'Droit', 'Fiscalite', 'Administration'],
            'JobRole': ['Junior', 'Senior', 'Manager', 'Director', 'Partner'],
            'MaritalStatus': ['Married', 'Single', 'Divorced']
        }

        df = preprocess_data_v1_0(data, categorical_columns)

        model = load_model("v3.0")

        prediction = model.predict_proba(df.iloc[0:1])
        predicted_probability = prediction[0][1]
        predicted_percentage = round(predicted_probability * 100, 2)

        data = {
            'prediction': predicted_percentage,
        }
        return jsonify(data)

    except Exception as e:
        print(e)
        return jsonify({"error": "Invalid JSON data"}), 400


if __name__ == '__main__':
    app.run()


