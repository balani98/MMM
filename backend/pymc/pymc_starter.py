import pymc_func
import numpy as np
import logging
import json
import os
from datetime import datetime
from push_gcs import write_json_to_gcs
import configparser
# logging.basicConfig(level=logging.INFO)

# # Create file handler for INFO logs
# info_handler = logging.FileHandler('pymc_info.log')
# info_handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# info_handler.setFormatter(formatter)
# logging.getLogger('').addHandler(info_handler)
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

def build_pymc_model(data,filename):
    print(data,type(data['control_variables']))
    processed_data = process_data(data,filename)
    return 'Data received and processed successfully'
# data={"include_holiday":True,"prophet_country":"UNITED STATES OF AMERICA",
#          "date_variable":"POS.WeekEndDate",
#          "target_variable":"POS.Net...Sold",
#          "control_variables":["weighted_unit_price", "web_Direct", "POS.Inventory", "web_Events", "web_AvgPageViews", "corona_cases", "HA_promo_EML", "POS.Net.Quantity.Sold", "HA_promo_COM_weighted", "HA_promo_other"],
# 		 "media_variables":["PaidSearch_Impressions", "PaidSocial_Cost", "Display_Cost", "eml_Cost"],
# 		 "spend_variables":["PaidSearch_Cost", "PaidSocial_Cost", "Display_Cost", "eml_Cost"],
#          "organic_variables":[],
#          "adstock":"Exponential Fixed Decay","filename":"Brother_MMM.csv"}

def process_data(data, filename):
    try:   
        logging.basicConfig(level=logging.INFO)

        # Create file handler for INFO logs
        info_handler = logging.FileHandler('pymc_info.log')
        info_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        info_handler.setFormatter(formatter)
        logging.getLogger('').addHandler(info_handler)
        logging.info('5004') 
        df = pymc_func.read_file(data['filename'])
        logging.info("File has been read")
        prior_mu = pymc_func.calculate_prior_mu(df, data['spend_variables'],data['organic_variables'])
        logging.info("prior mu has been calculated")
        scaled_lam = pymc_func.calculate_scaled_lam(df, data['spend_variables'],data['organic_variables'])
        logging.info("scaled lam has been calculated")
        #data['media_variables'].extend(data['organic_variables'])
        #data['spend_variables'].extend(data['organic_variables'])
        org_med_variables=data['media_variables']+data['organic_variables']
        org_spend_variables=data['spend_variables']+data['organic_variables']
        X,y = pymc_func.data_set_training(df,data['date_variable'],data['target_variable']
                                        ,org_med_variables,data['control_variables'])
        logging.info("Data has been distributed into X and y")
        X_train_sorted,y_train_sorted,X_test_sorted,y_test_sorted = pymc_func.train_test_split(data['date_variable'],X,y,0.9)
        logging.info("train and test has been generated")
        prior_mu_array = np.array(prior_mu)
        scaled_lam_array = np.array(scaled_lam)
        mmm, trace = pymc_func.model_training(prior_mu_array,scaled_lam_array,org_med_variables,
                    data['control_variables'],X_train_sorted,y_train_sorted,data['target_variable'],
                    data['date_variable'],adstock=4,seasonality=10)
        logging.info("model is trained")
        contribution_percentage,volume_contribution = pymc_func.share_percentage(df, mmm, data['spend_variables'],data['media_variables'], original_scale=True)
        logging.info("Contribution percentage is generated")
        response_df = pymc_func.create_response_df(volume_contribution, df,data['spend_variables'],data['media_variables'])
        logging.info("Response curve is generated")
        response_curve = pymc_func.get_multi_line_chart_data2(response_df)
        metrics_dict, output_train, output_test = pymc_func.predict_metrics(mmm,X_train_sorted,y_train_sorted,X_test_sorted,y_test_sorted,data,contribution_percentage,data['target_variable'])
        logging.info("Different metrics are generated")
        pymc_func.graphs(output_train,y_train_sorted[data['date_variable']],y_train_sorted[data['target_variable']],metrics_dict['r2_train'],metrics_dict['adjusted_r2_train'],metrics_dict['rmse_train'],metrics_dict['nrmse_train'],metrics_dict['mape_train'],data['filename'])
        pymc_func.test_graphs(output_test,y_test_sorted[data['date_variable']],y_test_sorted[data['target_variable']],metrics_dict['r2_test'],metrics_dict['adjusted_r2_test'],metrics_dict['rmse_test'],metrics_dict['nrmse_test'],metrics_dict['mape_test'],data['filename'])
        output_dict = {
            'metrics':metrics_dict,
            'effective_share':contribution_percentage,
            'response_curve':response_curve,
            'response_curve_stats': {
                    'spend': [response_df['spend'].min(), response_df['spend'].max()],
                    'target': [response_df['target'].min(), response_df['target'].max()]
            }
        }
        output_json_folder = 'Output_json'  
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename_for_saving = "Output.json"
        write_json_to_gcs(BUCKET_NAME, filename, output_dict, file_name_to_be_put_gcs=filename_for_saving)
        # with open(output_json_path, 'w') as json_file:
        #     json.dump(output_dict, json_file, indent=4)
    except Exception as error:
        logging.info('5003')
    logging.info('5002')
    return output_dict

