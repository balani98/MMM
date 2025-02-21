from google.cloud import storage
import datetime
from pathlib import Path
import configparser
Config = configparser.ConfigParser()
import os
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

def create_directory_with_timestamp(bucket_name, directory):
    # Initialize GCS client
    client = storage.Client.from_service_account_json(SERVICE_ACCOUNT)

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # Generate timestamp
    timestamp_ = str(datetime.datetime.now().timestamp())
    # Format timestamp
    timestamp_ = timestamp_.replace(".", "")  # Remove dot from timestamp

    # Construct filename with directory path
    filename = directory + "/Model_" + timestamp_ + '/'

    # Create directory structure in GCS
    blob = bucket.blob(filename)
    blob.upload_from_string('')  # Upload an empty string to create a "directory"
    return filename

from google.cloud import storage
import json

def write_json_to_gcs(bucket_name, file_path, output_dict, file_name_to_be_put_gcs):
    # Initialize GCS client
    client = storage.Client.from_service_account_json(SERVICE_ACCOUNT)

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # Construct the blob (file) name with directory path
    blob_name = file_path + file_name_to_be_put_gcs

    # Convert output_dict to JSON string
    json_data = json.dumps(output_dict, ensure_ascii=False, indent=4)

    # Create blob object and upload JSON data
    blob = bucket.blob(blob_name)
    blob.upload_from_string(json_data, content_type='application/json')

    print(f"JSON file uploaded to gs://{bucket_name}/{blob_name}")

