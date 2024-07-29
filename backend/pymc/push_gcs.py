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
    log_file_path = Config.get('DOCKER','log_file_path')
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
    

def upload_log_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """
    Uploads a file to Google Cloud Storage.

    Args:
    - bucket_name: Name of the bucket to upload to.
    - source_file_name: Local path to the file to be uploaded.
    - destination_blob_name: Name of the blob (object) in GCS.

    Returns:
    - None
    """
    # Initialize a client
    storage_client = storage.Client.from_service_account_json(SERVICE_ACCOUNT)

    # Get bucket object
    bucket = storage_client.bucket(bucket_name)

    # Path to local file
    source_file_path = source_file_name

    # Destination blob in the bucket
    blob = bucket.blob(destination_blob_name)

    # Upload the file
    blob.upload_from_filename(source_file_path)

    print(f"File {source_file_name} uploaded to {destination_blob_name} in bucket {bucket_name}")
    

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

