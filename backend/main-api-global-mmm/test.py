import requests
from model_comparison import compare_models
from push_gcs import create_directory_with_timestamp,write_json_to_gcs
import configparser
import os 
from pull_gcs import extract_json_from_gcs
os.environ['ENVIRONMENT'] = 'DOCKER'
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
global global_robyn_status
global global_pymc_status
global_robyn_status='5002'
global_pymc_status='5002'   
def compare_models_helper():
    print("model comparison function got triggered")
    global global_robyn_status
    global global_pymc_status
    if global_robyn_status == "5002" and global_pymc_status == "5002":
        print("both robyn and pymc successfully executed")
        #robyn_model_results = requests.get('http://robyn:8001/api/robyn_model_results')
        #pymc_model_results = requests.get('http://pymc:8003/api/pymc_model_results')
        robyn_model_results = extract_json_from_gcs(BUCKET_NAME,'models/robyn/Model_1718215034868482/Output.json')
        pymc_model_results = extract_json_from_gcs(BUCKET_NAME,'models/pymc/Model_1718215037875614/Output.json')
        #best_model = compare_models(robyn_model_results.json(), pymc_model_results.json())
        best_model = compare_models(robyn_model_results, pymc_model_results)
        print(best_model)
        filename = create_directory_with_timestamp(BUCKET_NAME, file_path)
        if best_model == "robyn":
            print("robyn results written")
            write_json_to_gcs(BUCKET_NAME, file_path=filename,output_dict=robyn_model_results, file_name_to_be_put_gcs='Output.json')
        elif best_model == "pymc":
            print("pymc results written")
            write_json_to_gcs(BUCKET_NAME, file_path=filename,output_dict=pymc_model_results, file_name_to_be_put_gcs='Output.json')
        # store the log file to GCS with timestamp inside the Model folder
        # delete the same log file from system
        # produce the notification
        #notify()
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
        write_json_to_gcs(BUCKET_NAME, file_path=filename,output_dict=pymc_model_results.json(), file_name_to_be_put_gcs='Output.json')
        # delete the log file
        # produce the notification
        #notify()
    elif global_robyn_status == "5002" and global_pymc_status == "5003":
        print("pymc model failed but robyn model succeded")
        robyn_model_results = requests.get('http://robyn:8001/api/robyn_model_results')
        filename = create_directory_with_timestamp(BUCKET_NAME, file_path)
        write_json_to_gcs(BUCKET_NAME, file_path=filename,output_dict=robyn_model_results.json(), file_name_to_be_put_gcs='Output.json')
        # delete the log file
        # produce the notification
        #notify()
    elif global_robyn_status == "5004" and global_pymc_status == "5004":
        print("DO NOTHING")
    elif global_robyn_status == "5004" and global_pymc_status == "5002":
        print("DO NOTHING")
    elif global_robyn_status == "5002" and global_pymc_status == "5004":
        print("DO NOTHING")
compare_models_helper()
