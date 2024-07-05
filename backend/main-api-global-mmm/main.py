import io
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import pandas as pd
import os
from google.cloud import storage
from flask_caching import Cache
from werkzeug. utils import secure_filename
from Explorer import read
from Explorer.explorer_model import explorer 
from predictor_user_options import predictor_user_options
import configparser
from flask import send_file
import os
from flask import Flask
from flask_session import Session
from flask import session
import json
import datetime
from pathlib import Path
import re
from pull_gcs import get_latest_folder_with_files,extract_json_from_gcs
from push_gcs import create_directory_with_timestamp
import requests
from preprocessing_inputs import pre_processing_class
from flask_apscheduler import APScheduler
from push_gcs import write_json_to_gcs
from model_comparison import compare_models
from flask_socketio import SocketIO, emit
# CONSTANTS
ERROR_DICT = {
    "5002": "Value Error",
    "5003": "Type Error",
    "5004": "Incorrect Date Format",
}
Config = configparser.ConfigParser()
Config.read('config.ini')
if os.environ.get('ENVIRONMENT') == 'PRODUCTION':
    SERVICE_ACCOUNT = Config.get('PRODUCTION', 'service_account')
    BUCKET_NAME = Config.get('PRODUCTION', 'bucket_name')
    UPLOAD_FOLDER = Config.get('PRODUCTION', 'UPLOAD_FOLDER')
    filepath = Config.get('PRODUCTION', 'filepath')
    IP_ADDRESS = Config.get('PRODUCTION', 'IP_ADDRESS')
    UserGuidepath = Config.get('LOCAL', 'UserGuidepath')
elif os.environ.get('ENVIRONMENT') == 'DOCKER' :
    SERVICE_ACCOUNT = Config.get('DOCKER', 'service_account')
    BUCKET_NAME = Config.get('DOCKER', 'bucket_name')
    UPLOAD_FOLDER = Config.get('DOCKER', 'UPLOAD_FOLDER')
    filepath = Config.get('DOCKER', 'filepath')
    IP_ADDRESS = Config.get('DOCKER', 'IP_ADDRESS')
    UserGuidepath = Config.get('DOCKER', 'UserGuidepath')
    file_path = Config.get('DOCKER', 'file_path')
else :
    SERVICE_ACCOUNT = Config.get('LOCAL', 'service_account')
    BUCKET_NAME = Config.get('LOCAL', 'bucket_name')
    UPLOAD_FOLDER = Config.get('LOCAL', 'UPLOAD_FOLDER')
    filepath = Config.get('LOCAL', 'filepath')
    IP_ADDRESS = Config.get('LOCAL', 'IP_ADDRESS')
    UserGuidepath = Config.get('LOCAL', 'UserGuidepath')

    
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

class Config:
    SCHEDULER_API_ENABLED = True

app.config.from_object(Config())

scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()
# CORS CONFIG
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app, origins=['*'])
# APIS 
# API to upload file to GCS

global_explorer_user_input = {}
global_output_dict = {}
global_response_df = {}
global_filename_uploaded = ""
global_robyn_status = {}
global_pymc_status = {}

@app.route('/notify')
def notify():
    # Emit a notification event when this endpoint is accessed
    socketio.emit('notification', {'message': 'Model has been trained'})
    return 'Notification sent!'

@app.route('/api/explorer/uploadfile',methods=['POST'])
def uploadFile():
    try:
        print("cache cleared")
        client = storage.Client.from_service_account_json(SERVICE_ACCOUNT)
        bucket = client.bucket(BUCKET_NAME)
        f=request.files['csv']
        bucket.blob(secure_filename(f.filename)).upload_from_string(request.files['csv'].read(),content_type='text/csv')
        columns = read.read_csv_from_gcs(secure_filename(f.filename))
        global global_filename_uploaded
        global_filename_uploaded = secure_filename(f.filename)
        data = pd.read_pickle(UPLOAD_FOLDER+"data.pkl")
        eo = explorer(data)
        user_options = eo.onSubmit_selection()
        result={
               "filename":secure_filename(f.filename),
               "columns":columns,
                "user_options": user_options,
                "message":"uploaded successfully"
                }
        return result,200

    except Exception as e:
        print(e)
        return {'error':str(e)} ,500

# API to perform date checks        
@app.route('/api/explorer/datecheck',methods=['POST'])
def dateCheck():
    try:
        data = pd.read_pickle(UPLOAD_FOLDER+"data.pkl")
        eo = explorer(data)
        body = request.get_json()
        dateSelector = body['date_selector']
        eo.date = dateSelector
        print(eo.date)
        eo.date_check()
        data_granularity = eo.get_data_granularity()
        print(data_granularity)
        result={
                
                'data_granularity':data_granularity,
                "message":"date validated successfully"
            }
        return result,200
    except Exception as exp_date_check:
        return {"error": ERROR_DICT[str(exp_date_check)]}, 500

# API to perform Investment checks
@app.route('/api/explorer/investmentcheck',methods=['POST'])
def investmentCheck():
    try:
        data = pd.read_pickle(UPLOAD_FOLDER+"data.pkl")
        eo = explorer(data)
        body = request.get_json()
        investmentSelectorList = body['investment_selector']
        eo.channel_spend = investmentSelectorList
        eo.channel_spend_check()
        print("spend validated")
        result={
                "message":"spend variable validated successfully",
                "status":200
            }
        return result,200
    except Exception as exp_spend_check:
        print(exp_spend_check)
        return {"error": ERROR_DICT[str(exp_spend_check)]}, 500

# API to perform Target Checks
@app.route('/api/explorer/targetcheck',methods=['POST'])
def targetCheck():
    try:
        data = pd.read_pickle(UPLOAD_FOLDER+"data.pkl")
        eo = explorer(data)
        body = request.get_json()
        target_selector = body['target_selector']
        eo.target = target_selector
        eo.target_numeric_check()
        result={
            "message":"uploaded successfully",
            "status":200
            }
        return result,200
    except Exception as exp_target_check:
        print(exp_target_check)
        return {"error": ERROR_DICT[str(exp_target_check)]}, 500

# API to generate EDA report 
@app.route('/api/explorer/generateEDAReport', methods=['POST'])
def generate_eda_report():
    try:
        data = pd.read_pickle(UPLOAD_FOLDER+"data.pkl")
        eo = explorer(data)
        body = request.get_json()
        eo.date = body['dateSelector']
        eo.channel_spend = body['investmentSelector']
        eo.granularity_selected = body['dataGranularity']
        eo.target = body['targetSelector']
        eo.target_type = body['targetTypeSelector']
        (validation_report, sample_report, explorer_user_input, UI_stats_json) = eo.execute(filepath)
        global global_explorer_user_input
        global_explorer_user_input = explorer_user_input
        result={
            "validation_report":validation_report,
            "sample_report":sample_report,
            "explorer_report":explorer_user_input,
            "UI_stats":UI_stats_json,
            "status":200
            }
        return result,200
    except Exception as exp:
        print(str(exp))
        return {"error": str(exp)}, 500

@app.route('/api/explorer/downloadEDAReport', methods=['GET'])
def download_eda_report():
    try:
        filepath = UPLOAD_FOLDER + 'EDA_Report.html'
        return send_file(filepath, as_attachment=True)
    except Exception as exp:
        print(str(exp))
        return {"error": str(exp)}, 500

@app.route('/api/explorer/downloadUserguide', methods=['GET'])
def download_user_guide():
    try:
        filepath = UserGuidepath + 'userguide.pdf'
        return send_file(filepath, as_attachment=True)
    except Exception as exp:
        print(str(exp))
        return {"error": str(exp)}, 500
    
@app.route('/api/predictor',methods=['GET'])
def predictor_user_inputs():
    try:
        data = pd.read_pickle(UPLOAD_FOLDER+"data.pkl")
        user_params = global_explorer_user_input
        user_options = predictor_user_options(data, user_params)
        result = { 
                    'body': {
                        'user_options':user_options,
                        'message': 'Predictor inputs done'
                        },
                    'status':200
                }
        return result,200
    except Exception as exp:
        print(str(exp))
        return {"error": str(exp)}, 500



@app.route('/api/predictor',methods=['POST'])
def build_the_models():
    try:
        body = request.get_json()
        predictor_user_input = body
        explorer_user_input = {}
        print(global_explorer_user_input)
        explorer_user_input = global_explorer_user_input
        print("1")
        explorer_user_input['filename'] = global_filename_uploaded 
        print("2",explorer_user_input)
        body_to_send = {
            'explorer_user_inputs':explorer_user_input,
            'predictor_user_inputs':predictor_user_input
        }
        
        pre_processor = pre_processing_class(global_filename_uploaded)
        required_inputs = pre_processor.execute(body_to_send)
        robyn_inputs = required_inputs[0]
        pymc_inputs = required_inputs[1]
        
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        robyn_response = requests.post('http://robyn:8001/api/build_robyn_model', headers=headers,json=robyn_inputs, timeout=700.0)
        pymc_response = requests.post('http://pymc:8003/api/build_pymc_model', headers=headers,json=pymc_inputs, timeout=700.0)
        # Check the response
        if robyn_response.status_code != 200:
            print(f"Robyn Request failed: {robyn_response.status_code} - {robyn_response.text}")
        else:
            print("Robyn Request succeeded:", robyn_response.json())
            scheduler.add_job(id='robynJob', func=read_robyn_log_file_periodically, trigger='interval', minutes=10)
            data_robyn = robyn_response.json()
        if pymc_response.status_code != 200:
            print(f"PYMC Request failed: {pymc_response.status_code} - {pymc_response.text}")
        else:
            print("PYMC Request succeeded:", pymc_response.json())
            scheduler.add_job(id='pymcJob', func=read_pymc_log_file_periodically, trigger='interval', minutes=10)
            data_pymc = pymc_response.json()
        scheduler.add_job(id='modelComparisonJob', func=compare_models_helper, trigger='interval', minutes=10)
        data = {
            'robyn':data_robyn,
            'pymc':data_pymc
        }
        return data,200
    except Exception as exp:
        print(str(exp))
        return {"error": str(exp)}, 500

@app.route('/api/predictor/generate_response_curves',methods=['GET'])
def generate_resp_curves():
    try:
        file_name =  get_latest_folder_with_files(BUCKET_NAME,file_path)
        if file_name is None:
            result = { 
                    'body': {
                        'message': 'Model is in training phase'
                        },
                    'status':400
                }
            return result,400
        else:
            print(file_name)
           
            output_json_file = str(file_path)  + '/'+ str(file_name) + '/Output.json'
            output_json_file =  extract_json_from_gcs(BUCKET_NAME, output_json_file)
            result = { 
                        'body': {
                        
                            'output_dict':output_json_file,
                            'message': 'Predictor response curves data sent'
                            },
                        'status':200
                    }
        return result,200
    except Exception as exp:
        print(str(exp))
        return {"error": str(exp)}, 500

def read_robyn_log_file_periodically():
    robyn_status_response = requests.get('http://robyn:8001/api/check_robyn_logs')
    global global_robyn_status
    robyn_status_response_json = robyn_status_response.json()
    global_robyn_status = robyn_status_response_json["body"]["robyn_model_status"]
    print("robyn",global_robyn_status)
    
def read_pymc_log_file_periodically():
    pymc_status_response = requests.get('http://pymc:8003/api/check_pymc_logs')
    global global_pymc_status
    pymc_status_response_json = pymc_status_response.json()
    global_pymc_status = pymc_status_response_json["body"]["pymc_model_status"]
    print("pymc",global_pymc_status)

@app.route('/api/model_results',methods=['GET'])    
def compare_models_helper():
    print("model comparison function got triggered")
    global global_robyn_status
    global global_pymc_status
    if global_robyn_status == "5002" and global_pymc_status == "5002":
        print("both robyn and pymc successfully executed")
        robyn_model_results = requests.get('http://robyn:8001/api/robyn_model_results')
        pymc_model_results = requests.get('http://pymc:8003/api/pymc_model_results')
        robyn_model_results_json = robyn_model_results.json()
        pymc_model_results_json = pymc_model_results.json()
        best_model = compare_models(robyn_model_results_json['body']['output_dict'], pymc_model_results_json['body']['output_dict'])
        filename = create_directory_with_timestamp(BUCKET_NAME, file_path)
        if best_model == "robyn":
            print("robyn results written")
            write_json_to_gcs(BUCKET_NAME, file_path=filename,output_dict=robyn_model_results_json['body']['output_dict'], file_name_to_be_put_gcs='Output.json')
           
        elif best_model == "pymc":
            print("pymc results written")
            write_json_to_gcs(BUCKET_NAME, file_path=filename,output_dict=pymc_model_results_json['body']['output_dict'], file_name_to_be_put_gcs='Output.json')
        status_log_file_to_gcs = requests.get('http://robyn:8001/api/log_file_to_gcs')
        delete_log_file_from_system_robyn = requests.delete('http://robyn:8001/api/log_file')
        status_log_file_to_gcs = requests.get('http://pymc:8003/api/log_file_to_gcs')
        delete_log_file_from_system_pymc = requests.delete('http://pymc:8003/api/log_file')
        scheduler.remove_job('robynJob')
        scheduler.remove_job('pymcJob')
        scheduler.remove_job('modelComparisonJob')
        # produce the notification
        notify()
        # stop the scheduler
    elif global_robyn_status == "5003" and global_pymc_status == "5003":
        print("both robyn and pymc failed")
        print("Consider re-training the model")
    elif global_robyn_status == "5003" and global_pymc_status == "5004":
        print("robyn model failed but pymc is in progress")
        print("DO NOTHING")
    elif global_robyn_status == "5004" and global_pymc_status == "5003":
        print("pymc model failed but robyn model is in progress")
        print("DO NOTHING")
    elif global_robyn_status == "5003" and global_pymc_status == "5002":
        print("robyn model failed but pymc model succeeded")
        pymc_model_results = requests.get('http://pymc:8003/api/pymc_model_results')
        filename = create_directory_with_timestamp(BUCKET_NAME, file_path)
        pymc_model_results_json =  pymc_model_results.json()
        write_json_to_gcs(BUCKET_NAME, file_path=filename,output_dict=pymc_model_results_json['body']['output_dict'], file_name_to_be_put_gcs='Output.json')
        status_log_file_to_gcs_robyn = requests.get('http://robyn:8001/api/log_file_to_gcs')
        status_log_file_to_gcs_pymc = requests.get('http://pymc:8003/api/log_file_to_gcs')
        delete_log_file_from_system_robyn = requests.delete('http://robyn:8001/api/log_file')
        delete_log_file_from_system_pymc = requests.delete('http://pymc:8003/api/log_file')
        scheduler.remove_job('robynJob')
        scheduler.remove_job('pymcJob')
        scheduler.remove_job('modelComparisonJob')
        # delete the log file
        # produce the notification
        notify()
    elif global_robyn_status == "5002" and global_pymc_status == "5003":
        print("pymc model failed but robyn model succeded")
        robyn_model_results = requests.get('http://robyn:8001/api/robyn_model_results')
        filename = create_directory_with_timestamp(BUCKET_NAME, file_path)
        robyn_model_results_json = robyn_model_results.json()
        write_json_to_gcs(BUCKET_NAME, file_path=filename,output_dict=robyn_model_results_json['body']['output_dict'], file_name_to_be_put_gcs='Output.json')
        status_log_file_to_gcs_robyn = requests.get('http://robyn:8001/api/log_file_to_gcs')
        status_log_file_to_gcs_robyn = requests.get('http://pymc:8003/api/log_file_to_gcs')
        # delete the log file
        delete_log_file_from_system_pymc = requests.delete('http://robyn:8001/api/log_file')
        delete_log_file_from_system_pymc = requests.delete('http://pymc:8003/api/log_file')
        scheduler.remove_job('robynJob')
        scheduler.remove_job('pymcJob')
        scheduler.remove_job('modelComparisonJob')
        # produce the notification
        notify()
    elif global_robyn_status == "5004" and global_pymc_status == "5004":
        print("DO NOTHING")
    elif global_robyn_status == "5004" and global_pymc_status == "5002":
        print("DO NOTHING")
    elif global_robyn_status == "5002" and global_pymc_status == "5004":
        print("DO NOTHING")

if __name__ == '__main__':
    PORT = int(os.getenv('PORT')) if os.getenv('PORT') else 8080
    # # This is used when running locally. Gunicorn is used to run the
    # # application on Cloud Run. See entrypoint in Dockerfile.
    socketio.run(app, host='127.0.0.1', port=PORT, debug=True)
    #app.run(host='127.0.0.1', port=PORT, debug=True)
    #app.run(host="0.0.0.0", port=5000)