from google.cloud import storage
import json
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

def get_latest_folder_with_files(bucket_name, directory):
    try:
        # Initialize GCS client
        client = storage.Client.from_service_account_json(SERVICE_ACCOUNT)

        # Get the bucket
        bucket = client.bucket(bucket_name)

        # List blobs (folders) in the directory
        blobs = bucket.list_blobs(prefix=directory + '/')
        # Extract folder names from blob names
        folders = [blob.name[len(directory) + 1:].split('/')[0] for blob in blobs]
        #folders = [blob.name[len(directory) + 1:].split('/')[0] for blob in blobs if blob.name.endswith('/')]
        print(folders)
        #cleaning
        folders = [folder for folder in folders if folder]
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
        


def extract_json_from_gcs(bucket_name, file_name):
    # Initialize GCS client
    client = storage.Client.from_service_account_json(SERVICE_ACCOUNT)

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # Get the blob (file) from GCS
    blob = bucket.blob(file_name)

    # Read the file as a string
    file_content = blob.download_as_string()

    # Parse the file content as JSON
    json_data = json.loads(file_content)

    return json_data


