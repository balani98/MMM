import pandas as pd
from google.cloud import storage
import configparser
import os
import copy
import datetime 
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
class pre_processing_class:
    def __init__(self, blob_name):
        """
        initialization of the class
        blob_name: name of the CSV file in Google Cloud Storage
        """
        self.df = self.read_csv_from_gcs(blob_name)
        
    def read_csv_from_gcs(self, blob_name):
        """Read CSV file from Google Cloud Storage
        Args:
            blob_name: name of the CSV file in Google Cloud Storage
        Returns:
            DataFrame: DataFrame read from the CSV file
        """
        client = storage.Client.from_service_account_json(SERVICE_ACCOUNT)
        bucket = client.bucket("global_mso_data_files")
        blob = bucket.blob(blob_name)
        with blob.open("r") as data:
            df = pd.read_csv(data)
        return df

    def get_variable_list(self, user_params):
        """Get the list of the variables from dataset which are already selected by user in explorer
        Args:
            user_params: dictionary of user inputs selected by user in explorer
        Returns: list containing the user inputs
        """  
        check_var = ['date_variable', 'target_variable', 'spend_variables', 'media_variables', 'organic_variables']
                
        var_list = []
        for var in check_var:
            if user_params[var] == None:
                continue
            else:
                if isinstance(user_params[var], list):
                    var_list = var_list + user_params[var]
                else:
                    var_list = var_list + [user_params[var]]
                    
        var_list = list(set(var_list))
        
        return var_list
                
    def user_params_rules(self, user_params):
        """Defining the input rules for robyn and pymc MMM based on user inputs
            Getting parameters like media variables, orgaic variables and contextual variables
        Args:
            user_params: dictionary of user inputs selected by user in explorer and predictor
        Returns: list containing the user inputs
        """ 
        data_var = list(self.df.columns)
        #Calling user inputs list in explorer
        var_used = self.get_variable_list(user_params)
        
        # defining media variables apart from spend if available, impressions or clicks are considered for each channel
        # variable are selected based media spend variable names
        # if required variable is unavailable or doesn't meet the criteria for a particular channel, it's spend variable is considered
        if user_params['media_variables'] == None:
            user_params['media_variables'] = []
            
            for var in user_params['spend_variables']:
                media_name_check = [media_name for media_name in data_var if var.split('_')[0].lower() in media_name.lower()]
                media_var_check = [media_var for media_var in media_name_check if 'impression'.lower() in media_var.lower()]
                if not media_var_check:
                    print(var,1)
                    media_var_check = [media_var for media_var in media_name_check if 'click'.lower() in media_var.lower()]
                if not media_var_check:
                    print(var,2)
                    media_var_check = [var]
                print(var,media_var_check)
                user_params['media_variables'] = user_params['media_variables'] + [media_var_check[0]]
            
        if user_params['media_variables'] == None:
            user_params['media_variables'] = user_params['spend_variables']

        # Preparing contextual variables for robyn if user doesn't provide any input    
        if user_params['control_variables'] == None:
            user_params['control_variables'] = data_var
        params_to_exclude = list(set(var_used + user_params['media_variables']))
        if isinstance(params_to_exclude, list):
            user_params['control_variables'] = list(set(user_params['control_variables']).difference(set(params_to_exclude)))
        else:
            user_params['control_variables'].remove(params_to_exclude)
        
        # Checking the contextual variable inputs if it contaians any variables with non-numerical type
        con_var_list = []
        for col in user_params['control_variables']:
            if ((self.df[col].dtypes == object) or (self.df[col].dtypes == str) or 
            (self.df[col].dtypes == bool) or isinstance(self.df[col], datetime.date) or 
            pd.api.types.is_datetime64_dtype(self.df[col]) or (col == user_params['target_variable'])):
                continue
            else:
                con_var_list = con_var_list + [col]
        user_params['control_variables'] = con_var_list

            # Checking for organic variable in contextual variable and adjusting the inputs    
        if user_params['organic_variables'] == None:
            user_params['organic_variables'] = [org_var for org_var in user_params['control_variables'] if 'organic'.lower() in org_var.lower()]
            user_params['control_variables'] = list(set(user_params['control_variables']) - set(user_params['organic_variables']))
        
        return user_params
    
    def execute(self, inputjson):
        """main function to execute robyn mmm and other functionalities
        Args: 
            explorer_user_input: explorer page user inputs
            predictor_user_params: predictor page user inputs
            file_path: file path where EDA report has to be saved
        Returns:
            DataFrame: Response_df_py: response curves for channels
            Dictionary: output_dict: effect share for channels and adj r2 for test
        """
        # defining seasonal and other variables for robyn and pymc
        explorer_user_input = inputjson['explorer_user_inputs']
        predictor_user_params = inputjson['predictor_user_inputs']
        user_params = {
            'media_variables': None,
            'organic_variables': None
            }
        
        # merging inputs fron explorer and predictor
        user_params = {**predictor_user_params, **user_params}
        user_params = {**explorer_user_input, **user_params}

        # format the date column
        _format = "%Y-%m-%d"
        if pd.to_datetime(self.df[user_params['date_variable']], format=_format, errors='coerce').notnull().all(): 
            self.df[user_params['date_variable']] = pd.to_datetime(self.df[user_params['date_variable']], format=_format)
        self.df[user_params['date_variable']] = self.df[user_params['date_variable']].astype(str)
        
        self.df=self.df.fillna(0)
        
        user_params_robyn = copy.deepcopy(user_params)
        user_params_robyn = self.user_params_rules(user_params_robyn)
        user_params_pymc = {key: value for key, value in user_params_robyn.items() if key not in ['include_holiday', 'country', 'adstock']}

        return user_params_robyn, user_params_pymc
