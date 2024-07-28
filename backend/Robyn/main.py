import os
import configparser
import pandas as pd
from tasks import build_model_output_with_celery,celery_app
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from pull_gcs import get_latest_folder_with_files,extract_json_from_gcs
from push_gcs import create_directory_with_timestamp,upload_log_to_gcs
import json
from check_robyn_logs import check_robyn_logs
import datetime
app = Flask(__name__)
# CORS CONFIG
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app, origins=['*'])
# APIS 


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
    log_file_path = Config.get('DOCKER','log_file_path')
else :
    SERVICE_ACCOUNT = Config.get('LOCAL', 'service_account')
    BUCKET_NAME = Config.get('LOCAL', 'bucket_name')
    UPLOAD_FOLDER = Config.get('LOCAL', 'UPLOAD_FOLDER')
    filepath = Config.get('LOCAL', 'filepath')
    IP_ADDRESS = Config.get('LOCAL', 'IP_ADDRESS')
    UserGuidepath = Config.get('LOCAL', 'UserGuidepath')



@app.route('/api/build_robyn_model',methods=['POST'])
def build_the_model():
    try:
        print("1",request)
        body = request.get_json()
        print("2",body)
        #filename = create_directory_with_timestamp(BUCKET_NAME, file_path)
        filename = file_path + '/' + body["model_folder_to_write"] + '/'
        res = celery_app.send_task('tasks.build_model_output_with_celery', args=[body, filename])
        if len(body['control_variables']) == 0:
            body['control_variables'] = None
        result = { 
                    'body': {
                        'message':'Robyn model has been sent for building',
                        'task_id':res.id
                        },
                    'status':200
                }
        return result,200
    except Exception as exp:
        print(str(exp))
        return {"error": str(exp)}, 500
    
    
@app.route('/api/check_robyn_logs',methods=['GET'])
def check_robyn_model_status():
    try:
        log_file = 'robyn_info.log'
        robyn_model_status = check_robyn_logs(log_file)
        print(robyn_model_status)
        result = { 
                    'body': {
                        'robyn_model_status':robyn_model_status
                        },
                    'status':200
                }
        return result,200
    except Exception as exp:
        print(str(exp))
        return {"error": str(exp)}, 500
    

@app.route('/api/robyn_model_results',methods=['GET'])
def robyn_model_results():
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
            print(output_json_file)
            output_dict = extract_json_from_gcs(BUCKET_NAME, output_json_file)
            print(output_dict)
            result = { 
                        'body': {
                            'output_dict':output_dict
                            },
                        'status':200
                    }
        return result,200
    except Exception as exp:
        print(str(exp))
        return {"error": str(exp)}, 500


@app.route('/api/log_file',methods=['DELETE'])
def delete_log_file():
    try:
        # Check if file exists
        if os.path.exists(log_file_path):
            # Remove the file
            os.remove(log_file_path)
            print(f"{log_file_path} has been deleted successfully.")
            result = { 
            'body': {
                'message':'log file deleted'
                },
            'status':200
            }
            return result,200
        else:
            print(f"The file {log_file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred while trying to delete the file: {e}")
        
@app.route('/api/log_file_to_gcs',methods=['GET'])
def log_file_to_gcs():
    # Generate timestamp
    file_name =  get_latest_folder_with_files(BUCKET_NAME,file_path)
    remote_file_path =  str(file_path) + '/' + str(file_name) + '/robyn_info.log' 
    upload_log_to_gcs(BUCKET_NAME, log_file_path, remote_file_path)
    result = { 
            'body': {
                'message':'log file pushed to GCS'
                },
            'status':200
            }
    return result,200
    

    
   

    
if __name__ == '__main__':
    PORT = int(os.getenv('PORT')) if os.getenv('PORT') else 8080
    # # This is used when running locally. Gunicorn is used to run the
    # # application on Cloud Run. See entrypoint in Dockerfile.
    app.run(host='127.0.0.1', port=PORT, debug=True)
    #app.run(host="0.0.0.0", port=5000)