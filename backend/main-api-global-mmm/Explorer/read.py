# Imports the Google Cloud client library
import pandas as pd
from google.cloud import storage
import configparser
import os

Config = configparser.ConfigParser()
Config.read('config.ini')
if os.environ.get('ENVIRONMENT') == 'PRODUCTION':
    SERVICE_ACCOUNT = Config.get('PRODUCTION', 'service_account')
    BUCKET_NAME = Config.get('PRODUCTION', 'bucket_name')
    UPLOAD_FOLDER = Config.get('PRODUCTION', 'UPLOAD_FOLDER')
    filepath = Config.get('PRODUCTION', 'filepath')
    IP_ADDRESS = Config.get('PRODUCTION', 'IP_ADDRESS')
elif os.environ.get('ENVIRONMENT') == 'DOCKER':
    SERVICE_ACCOUNT = Config.get('DOCKER', 'service_account')
    BUCKET_NAME = Config.get('DOCKER', 'bucket_name')
    UPLOAD_FOLDER = Config.get('DOCKER', 'UPLOAD_FOLDER')
    filepath = Config.get('DOCKER', 'filepath')
    IP_ADDRESS = Config.get('DOCKER', 'IP_ADDRESS')
else:
    SERVICE_ACCOUNT = Config.get('LOCAL', 'service_account')
    BUCKET_NAME = Config.get('LOCAL', 'bucket_name')
    UPLOAD_FOLDER = Config.get('LOCAL', 'UPLOAD_FOLDER')
    filepath = Config.get('LOCAL', 'filepath')
    IP_ADDRESS = Config.get('LOCAL', 'IP_ADDRESS')
def read_csv_from_gcs(blob_name):
    client = storage.Client.from_service_account_json(SERVICE_ACCOUNT)
    bucket = client.bucket("global_mso_data_files")
    upload = client.bucket("upload_folder")
    blob = bucket.blob(blob_name)
    with blob.open("r") as data:
        df = pd.read_csv(data)
        list_of_selectors = list(df.columns)
        df.to_pickle(UPLOAD_FOLDER+"data.pkl")
        return list_of_selectors
        # selectors_data_list = list(data.columns)  # column header for dropdown
        # print("list created")