import warnings
import json
#import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
from scipy.stats import beta
from scipy.stats import gamma
from scipy.optimize import curve_fit, minimize
from math import ceil
from sklearn.model_selection import train_test_split
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from google.cloud import storage
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_percentage_error
import math
import os
from datetime import datetime
#import pymc.sampling_jax
import configparser
import logging 
from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation
from pymc_marketing.mmm.delayed_saturated_mmm import DelayedSaturatedMMM

warnings.filterwarnings("ignore")
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
#az.style.use("arviz-darkgrid")
plt.rcParams["figure.figsize"] = [12, 7]
plt.rcParams["figure.dpi"] = 100


def read_csv_from_gcs(blob_name):
    client = storage.Client.from_service_account_json(SERVICE_ACCOUNT)
    bucket = client.bucket("global_mso_data_files")
    upload = client.bucket("upload_folder")
    blob = bucket.blob(blob_name)
    with blob.open("r") as data:
        df = pd.read_csv(data)
    return df

def read_file(filename):
    data=read_csv_from_gcs(filename)
    logging.info("file read successfully")
    filename = os.path.splitext(filename)[0]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # filename_for_saving = f"Input_{filename}_{timestamp}.csv" 
    # filename_for_saving_json = f"Input_{filename}_{timestamp}.json"  
    # input_csv_folder = 'Input_csv' 
    # input_json_folder = 'Input_json' 
    # input_csv_path = os.path.join(input_csv_folder, filename_for_saving) 
    # input_json_path = os.path.join(input_json_folder, filename_for_saving_json)  
    # data.to_csv(input_csv_path, index=False)
    # data_json = data.to_json(orient='records') 
    # with open(input_json_path, 'w') as json_file:
    #     json.dump(data_json, json_file, indent=4)  
    return data

def calculate_prior_mu(data, spend_variables,organic_variables):
    total_spend_per_channel = data[spend_variables].sum(axis=0)
    spend_share = total_spend_per_channel / total_spend_per_channel.sum()
    HALFNORMAL_SCALE = 1 / np.sqrt(1 - 2 / np.pi)
    n_channels = len(spend_variables)
    prior_mu = HALFNORMAL_SCALE * n_channels * spend_share
    prior_mu_normalized = prior_mu / np.sum(prior_mu)
    prior_mu = prior_mu_normalized/2
    for i in organic_variables:
        prior_mu[i]=0.01
    return prior_mu
def calculate_scaled_lam(data, spend_variables,organic_variables,acceptable_lam=2.5):
    max_impressions_per_channel = data[spend_variables].max(axis=0)  #spend_variables
    scaled_lam = 3 * max_impressions_per_channel / max_impressions_per_channel.max()
    min_acceptable_lam = acceptable_lam
    scaled_lam = np.interp(scaled_lam, (scaled_lam.min(), scaled_lam.max()), (min_acceptable_lam, 3))
    scaled_lam_series = pd.Series(scaled_lam, index=spend_variables)
    for i in organic_variables:
        scaled_lam_series[i]=0.12
    return scaled_lam_series

def data_set_training(data,date_variable,independent_variable,media_variables,control_variables):
    y = data[[date_variable,independent_variable]]
    selected_columns = media_variables + control_variables +[date_variable]
    print('selected_columns',selected_columns)
    X = data.loc[:, selected_columns]
    return X,y

def model_training(prior_mu_array,scaled_lam_array,media_variables,
                   control_variables,X_train_sorted,y_train_sorted,independent_variable,date_variable,
                   adstock=4,seasonality=10):
    my_model_config = {'beta_channel': {'dist': 'InverseGamma',"kwargs":{"mu":prior_mu_array, "sigma": 0.3}},
                   'lam': {'dist': 'Gamma',"kwargs":{"mu":scaled_lam_array, "sigma": 2}},
                    "likelihood": {"dist": "Normal","kwargs":{"sigma": {'dist': 'HalfNormal', 'kwargs': {'sigma': 4}}},
                                'intercept': {'dist': 'Normal', 'kwargs': {'mu': 0, 'sigma': 2}},#tvp
 'alpha': {'dist': 'Beta', 'kwargs': {'alpha': 1, 'beta': 3}},
 'lam': {'dist': 'Gamma', 'kwargs': {'alpha': 3, 'beta': 1}},
 'gamma_control': {'dist': 'Normal', 'kwargs': {'mu': 0, 'sigma': 2}},
 'gamma_fourier': {'dist': 'Laplace', 'kwargs': {'mu': 0, 'b': 1}}}
                                }
    
    sampler_configuration = {"progressbar": True}

    mmm_6 = DelayedSaturatedMMM(
            model_config = my_model_config,
            sampler_config = sampler_configuration,
            date_column=date_variable,
            channel_columns=media_variables,
            control_columns=control_variables,
            adstock_max_lag=adstock,
            yearly_seasonality=seasonality 
        )

    #with pm.Model() as model_6:
    trace_6 = mmm_6.fit(X=X_train_sorted,
                        y=y_train_sorted[independent_variable],
                        chains=1)
    return mmm_6,trace_6
    
def train_test_split(date_variable,X,y,split=0.75):
    split_index = int(split * len(X))

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    X_train_sorted = X_train.sort_values(by=date_variable).reset_index()
    y_train_sorted = y_train.sort_values(by=date_variable).reset_index()
    X_test_sorted = X_test.sort_values(by=date_variable).reset_index()
    y_test_sorted = y_test.sort_values(by=date_variable).reset_index()
    return X_train_sorted,y_train_sorted,X_test_sorted,y_test_sorted

def share_percentage(df, mmm_6, spend_variables,media_variables, original_scale=True):
  
  """
  Processes contributions, considering spend_variables for filtering and calculating ratios.

  Args:
      mmm_6: The model.
      spend_variables: A list of variables to consider from volume_contribution.
      original_scale: Whether to use the original scale (default: True).

  Returns:
      A dictionary containing 'spend_ratio' and 'effective_ratio' for each filtered column.
  """
  volume_contribution = mmm_6.compute_mean_contributions_over_time(original_scale=original_scale)
  volume_contribution.to_csv('outcsvi.csv')
  total_added_values = volume_contribution[media_variables].sum(axis=0)
  total_spend = df[spend_variables].sum(axis=0)
  spend_ratio = {col: (total_spend[col] / total_spend.sum()) * 100 for col in spend_variables}
  effective_ratio = {col: (total_added_values[col] / total_added_values.sum()) * 100 for col in media_variables}

    # Convert dictionaries to DataFrames
  spend_ratio_df = pd.DataFrame(spend_ratio.items(), columns=['Column', 'Spend'])
  effective_ratio_df = pd.DataFrame(effective_ratio.items(), columns=['Column', 'Effective'])
  spend_ratio_df['Effective'] = effective_ratio_df['Effective']
  out = spend_ratio_df.set_index('Column').to_dict(orient='index')
    # Create a new dictionary to store the modified format
  modified_out = {}

    # Iterate over the items in the original dictionary
  for key, value in out.items():
        spend = value['Spend']
        effective = value['Effective']
        if isinstance(spend, (int, float)):
            spend = round(spend, 10)  # Round to avoid scientific notation
        if isinstance(effective, (int, float)):
            effective = round(effective, 10)  # Round to avoid scientific notation
        modified_out[key] = [spend, effective]
  return modified_out, volume_contribution

def create_response_df(volume_contribution, spend_df, spend_variables, media_variables):
  """
  Creates a new DataFrame 'response_df' with columns 'channel' and 'date',
  populated from the input DataFrame 'df' and filtered based on 'spend_variables'.
  Adds a 'target_total' column from a separate 'data' DataFrame.

  Args:
      df (pandas.DataFrame): The input DataFrame containing spend data.
      spend_variables (list): A list of channel names (spend variables) to include.
      data (pandas.DataFrame): Another DataFrame containing target data.

  Returns:
      pandas.DataFrame: The new DataFrame 'response_df' with the desired columns.
  """
  volume_contribution.reset_index(inplace=True)
    # Get the date columns (assuming they're the first columns)
  date_col = next((col for col in volume_contribution.columns if 'date' in col.lower()), None)
  date_col_2 = next((col for col in spend_df.columns if 'date' in col.lower()), None)
  spend_df[date_col_2] = pd.to_datetime(spend_df[date_col_2])
    # Melt the DataFrames
  melted_df = volume_contribution.melt(id_vars=date_col, var_name="channel", value_name="target")
  melted_df_2 = spend_df.melt(id_vars=date_col_2, var_name="channel", value_name="spend")
    # Filter channels
  filtered_df = melted_df[melted_df['channel'].isin(media_variables)]
  filtered_df_2 = melted_df_2[melted_df_2['channel'].isin(media_variables)]
  filtered_df_2.rename(columns={date_col_2: "date"}, inplace=True)
  filtered_df['date'] = pd.to_datetime(filtered_df['date'])
  filtered_df_2['date'] = pd.to_datetime(filtered_df_2['date'])
  merged_df = pd.merge(filtered_df, filtered_df_2, how='inner', on=['date', 'channel'])
  return merged_df

def sort_response_curve(data):
    sorted_data = data.copy()
    for curve in sorted_data:
        sorted_values = sorted(curve["values"], key=lambda x: (x["spend"], x["target"]))
        curve["values"] = sorted_values
    return sorted_data

def get_multi_line_chart_data2(Response_df):
    """Saving the response curves for the trained model as json dump
    Args:
        Response_df: A Pandas DataFrame for reponse curves for trained and selected model
        filename: path where json needs to be saved
    Returns: json for response curve for trained model
    """  
    multi_line_chart_json = Response_df.to_dict('records')
    multi_line_chart_data2 = []
    old_key = ""
    multi_line_chart_obj = {}
    predictions_spend_obj = {}
    for index, obj in enumerate(multi_line_chart_json, start=0):
        dimension_key = obj['channel']
        if old_key != dimension_key:
            if bool(multi_line_chart_obj) is True:
                multi_line_chart_data2.append(multi_line_chart_obj)
            multi_line_chart_obj = {}
            multi_line_chart_obj["name"] = dimension_key
            multi_line_chart_obj["values"] = []
        predictions_spend_obj = {}
    
        predictions_spend_obj["spend"] = obj["spend"]
        predictions_spend_obj["target"] = obj["target"]
        multi_line_chart_obj["values"].append(predictions_spend_obj)
        if index == len(multi_line_chart_json)-1:
            multi_line_chart_data2.append(multi_line_chart_obj)
        old_key = dimension_key
    multi_line_chart_data2 = sort_response_curve(multi_line_chart_data2)
    return multi_line_chart_data2

def predict_metrics(mmm,X_train_sorted,y_train_sorted,X_test_sorted,y_test_sorted,data,contribution_percentage,target):
    out_train=mmm.sample_posterior_predictive(X_pred=X_train_sorted, extend_idata=False)
    out_test=mmm.sample_posterior_predictive(X_pred=X_test_sorted, extend_idata=False)
    output_train=out_train["y"].to_series().groupby("date").mean().values
    output_test=out_test["y"].to_series().groupby("date").mean().values
    r2_train=r2_score(y_train_sorted[target].to_list(), output_train)
    r2_test=r2_score(y_test_sorted[target].to_list(), output_test)

    n_test=X_test_sorted.shape[0]
    p=len(data['media_variables'])
    n_train=X_train_sorted.shape[0]
    if(n_train!= 1 + p):
        adjusted_r2_train = 1 - (1 - r2_train) * (n_train - 1) / (n_train - p - 1)
    else:
        adjusted_r2_train = "It is comming as Infinity"
    if(n_test!= 1 + p):
        adjusted_r2_test = 1 - (1 - r2_test) * (n_test - 1) / (n_test - p - 1)
    else:
        adjusted_r2_test = "It is comming as Infinity"
    
    if(not isinstance(adjusted_r2_train,str) and (not isinstance(adjusted_r2_test,str))):
        diff_adjusted_r2=adjusted_r2_train-adjusted_r2_test
    else:
        diff_adjusted_r2="One of the adj r2 is infinite"

    rmse_train = np.sqrt(mean_squared_error(y_train_sorted[target].to_list(), output_train))
    rmse_test = np.sqrt(mean_squared_error(y_test_sorted[target].to_list(), output_test))

    nrmse_train = rmse_train / (np.max(output_train) - np.min(output_train))
    nrmse_test = rmse_test / (np.max(output_test) - np.min(output_test))

    #rssd_ = sqrt(sum((effect_share-spend_share)^2))
    add = [(value[0] - value[1]) ** 2 for value in contribution_percentage.values()]
    rssd = math.sqrt(sum(add))
    mape_train=mean_absolute_percentage_error(y_train_sorted[target].to_list(), output_train)
    mape_test=mean_absolute_percentage_error(y_test_sorted[target].to_list(), output_test)

    return {'r2_train':r2_train,
            'r2_test':r2_test,
            'adjusted_r2_train':adjusted_r2_train,
            'adjusted_r2_test':adjusted_r2_test,
            'rmse_train':rmse_train,
            'rmse_test':rmse_test,
            'nrmse_train':nrmse_train,
            'nrmse_test':nrmse_test,
            'rssd':rssd,
            'mape_train':mape_train,
            'mape_test':mape_test,
            'diff_adjusted_r2': diff_adjusted_r2
            }, output_train, output_test
    
def graphs(output_train, train_date, train_target, r_squared, adjusted_r_squared, rmse, nrmse, mape, filename,
          model_label="Model", actual_label="Actual",  
          model_color="blue", actual_color="green",  
          line_styles=("--", "-")):  
    train_date  = pd.to_datetime(train_date)  # Allow customization of date column name   
    # Create a DataFrame for easy plotting
    plot_data = pd.DataFrame({
        'Date': train_date,
        model_label: output_train,
        actual_label: train_target
    })
    plot_data.set_index('Date', inplace=True)
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(plot_data.index, plot_data[model_label], label=model_label, color=model_color, linestyle=line_styles[0])
    plt.plot(plot_data.index, plot_data[actual_label], label=actual_label, color=actual_color, linestyle=line_styles[1])

    # Formatting the plot
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title(f'Model vs Actual Data Comparison\nR-squared: {r_squared}\nAdjusted R-squared: {adjusted_r_squared}\nRMSE: {rmse}\nNRMSE: {nrmse}\nMAPE: {mape}')
    plt.legend()
    plt.grid(True)

    # Formatting the x-axis for date
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=90)

    # Formatting the y-axis to be more readable
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    filename = os.path.splitext(filename)[0]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename_for_saving = f"metrics_train_{filename}_{timestamp}.png"
    curr_path = os.getcwd()
    save_path = os.path.join(curr_path, "Graph")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename_for_saving))

    
def test_graphs(output_test,test_date, test_target, r_squared, adjusted_r_squared, rmse, nrmse, mape, filename,
          model_label="Model", actual_label="Actual",  
          model_color="blue", actual_color="green",  
          line_styles=("--", "-")):  

    test_date  = pd.to_datetime(test_date)  # Allow customization of date column name
    
    # Create a DataFrame for easy plotting
    plot_data = pd.DataFrame({
        'Date': test_date,
        model_label: output_test,
        actual_label: test_target
    })
    plot_data.set_index('Date', inplace=True)
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(plot_data.index, plot_data[model_label], label=model_label, color=model_color, linestyle=line_styles[0])
    plt.plot(plot_data.index, plot_data[actual_label], label=actual_label, color=actual_color, linestyle=line_styles[1])

    # Formatting the plot
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title(f'Model vs Actual Data Comparison\nR-squared: {r_squared}\nAdjusted R-squared: {adjusted_r_squared}\nRMSE: {rmse}\nNRMSE: {nrmse}\nMAPE: {mape}')
    plt.legend()
    plt.grid(True)

    # Formatting the x-axis for date
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=90)

    # Formatting the y-axis to be more readable
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    filename = os.path.splitext(filename)[0]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename_for_saving = f"metrics_test_{filename}_{timestamp}.png"
    curr_path = os.getcwd()
    save_path = os.path.join(curr_path, "Graph")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename_for_saving))



