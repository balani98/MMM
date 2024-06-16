import os
import configparser
import pandas as pd
from tasks import build_model_output_with_celery,celery_app
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from pull_gcs import get_latest_folder_with_files,extract_json_from_gcs
from push_gcs import create_directory_with_timestamp
import configparser
import json
from check_pymc_logs import check_pymc_logs
from google.cloud import storage
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
    file_path = Config.get('DOCKER', 'file_path')
def get_latest_folder_with_files(bucket_name, directory):
    try:
        # Initialize GCS client
        client = storage.Client.from_service_account_json(SERVICE_ACCOUNT)

        # Get the bucket
        bucket = client.bucket(bucket_name)

        # List blobs (folders) in the directory
        blobs = bucket.list_blobs(prefix=directory + '/')
        print("blobs",blobs)
        # Extract folder names from blob names
        folders = [blob.name[len(directory) + 1:].split('/')[0] for blob in blobs if blob.name.endswith('/')]
        folders = [folder for folder in folders if folder]
        print("pymc folders",folders)
        if not folders:
            return None  # No folders found in the directory

        # Extract timestamps from folder names and sort them in descending order
        timestamps = sorted([float(folder.split('_')[-1]) for folder in folders], reverse=True)

        # Get the last two timestamps
        last_two_timestamps = timestamps[:2]

        # Find the folders corresponding to the last two timestamps
        last_two_folders = [folder for folder in folders if float(folder.split('_')[-1]) in last_two_timestamps]
        
        # Sort the last_two_folders based on their corresponding timestamps
        last_two_folders.sort(key=lambda x: float(x.split('_')[-1]), reverse=True)
        print('last 2 folders',  last_two_folders)
        # Check if each of the last two folders contains any files
        for folder in last_two_folders:
            blob_prefix = f"{directory}/{folder}/"
            blobs_in_folder = bucket.list_blobs(prefix=blob_prefix)
            if any(blob.name != blob_prefix for blob in blobs_in_folder):
                return folder

        return None  # None of the last two folders contain any files
    except Exception as e:
        print("An error occurred:", e)
        

print(get_latest_folder_with_files(BUCKET_NAME,file_path))