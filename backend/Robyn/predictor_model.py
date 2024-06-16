from itertools import dropwhile
import pandas as pd
import numpy as np
from pandas.api.types import is_categorical_dtype
from pandas.api.types import is_numeric_dtype, is_float_dtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import datetime
import copy
import json
from PIL import Image
from pathlib import Path
import warnings
from push_gcs import write_json_to_gcs
warnings.filterwarnings('ignore')

# Load in python libraries to use R
from rpy2.robjects.packages import importr
import rpy2.interactive as r
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import rpy2.robjects.numpy2ri as rpyn
import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter
pandas2ri.activate()
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
base = importr("base")
utils = importr("utils")
Robyn = importr("Robyn")
lubridate = importr("lubridate")
dplyr = importr("dplyr")
timeDate = importr("timeDate")
ggplot2 = importr("ggplot2")
lares = importr("lares")
parallel = importr("parallel")
doParallel = importr("doParallel")

def predictor_user_options(df, user_params):
    """Getting options for various user selections for predictor frontend 
        Args:
            df: A Pandas DataFrame
            user_param: user selection from explorer page
        Returns: dictionary containing user selections for predictor
    """
    country_list = ['ARGENTINA', 'ARUBA', 'AUSTRALIA', 'AUSTRIA', 'BANGLADESH', 'BELGIUM', 
                    'BRAZIL', 'CANADA', 'CHILE', 'CHINA', 'COLOMBIA', 'CROATIA', 'CZECHIA', 
                    'DENMARK', 'DOMINICAN REPUBLIC', 'EGYPT', 'ESTONIA', 'FINLAND', 'GERMANY', 
                    'GREECE', 'HONG KONG', 'HUNGARY', 'ICELAND', 'INDIA', 'INDONESIA', 'IRELAND', 
                    'ISRAEL', 'ITALY', 'KENYA', 'LITHUANIA', 'LUXEMBOURG', 'MALAYSIA', 'MEXICO', 
                    'NETHERLANDS', 'NEW ZEALAND', 'NICARAGUA', 'NIGERIA', 'NORWAY', 'PAKISTAN', 
                    'PARAGUAY', 'PERU', 'PHILIPPINES', 'POLAND', 'PORTUGAL', 'RUSSIA', 'SINGAPORE', 
                    'SLOVAKIA', 'SLOVENIA', 'SOUTH AFRICA', 'SOUTH KOREA', 'SPAIN', 'SWEDEN', 
                    'SWITZERLAND', 'THAILAND', 'UNITED KINGDOM', 'UNITED STATES OF AMERICA', 'VIETNAM']
    data_var = list(df.columns)
    for var in user_params['paid_media_spends']:
        media_name_check = [media_name for media_name in data_var if var.split('_')[0].lower() in media_name.lower()]
        media_var_imp = [media_var for media_var in media_name_check if 'impression'.lower() in media_var.lower()]
        media_var_click = [media_var for media_var in media_name_check if 'click'.lower() in media_var.lower()]

    context_vars = list(df.columns)
    context_vars = list(set(context_vars) - set([user_params['date_var']] + 
                                           [user_params['dep_var']] + 
                                           user_params['paid_media_spends'] +
                                           media_var_imp +
                                           media_var_click))
    for col in context_vars:
        if (df[col].dtypes == object) or (df[col].dtypes == str) or (df[col].dtypes == bool) or isinstance(df[col], datetime.date) or pd.api.types.is_datetime64_dtype(df[col]) or (col == user_params['dep_var']):
            context_vars = list(set(context_vars) - set([col]))
    
    adstock_list = ['Exponential Fixed Decay', 'Flexible Delay with No Lag', 'Flexible Delay with Lag']

    user_options = {'country' : country_list,
                    'context_vars' : context_vars,
                    'adstock' : adstock_list}
    
    return user_options

def feature_importance(df, user_input): 
    """Getting feature importance of the variables in the dataset, for users to make decision while choosing the contextual variables
        Args:
            df: A Pandas DataFrame
            user_param: user selection from explorer page
        Returns: DataFrame containing feature importance
    """   
    var_list = []
    for col in df.columns:
        if (df[col].dtypes == object) or (df[col].dtypes == str) or (df[col].dtypes == bool) or isinstance(df[col], datetime.date) or pd.api.types.is_datetime64_dtype(df[col]) or (col == user_input['dep_var']):
            continue
        else:
            var_list = var_list + [col]

    X = df[[x for x in df.columns if x in var_list]]
    y = df[user_input['dep_var']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    feat_importances = round(pd.Series(rf.feature_importances_, index=X.columns)*100, 2)
    feat_importances = feat_importances.sort_values(ascending=False)
    feat_importances_df = pd.DataFrame(feat_importances).reset_index(drop=False)
    feat_importances_df = feat_importances_df.rename({'index':'Variable', 0:'Importance'}, axis=1)

    return feat_importances_df
    

class predictor:
    def __init__(self, df):
        """
        initialization of the class
        date: date variable in the data
        channel_spend: list of channels and their repective spends
        target: dependent variable in the data
        target_type: select target type as revenue or conversion
        granularity_selected: granulaity of the uploaded dataset
        """
        self.df = df

        self.adstock_list = {'exponential fixed decay' : 'geometric_traditional', 
                             'flexible delay with no lag' : 'weibull_cdf_no_lagged', 
                             'flexible delay with lag' : 'weibull_pdf_all_shapes'}
        
    def get_country_code(self, user_params):
        """Getting the country code for the country selected by the user, as robyn only takes country code as input
        Args:
            df: A Pandas DataFrame
            user_param: user selection from predictor page, to update the dictionary
        Returns: dictionary containing updated user selection from predictor page
        """  
        robyn_country_dict = {'ARGENTINA': 'AR', 'ARUBA': 'AW', 'AUSTRALIA': 'AU', 'AUSTRIA': 'AT', 'BANGLADESH': 'BD', 'BELGIUM': 'BE', 
                            'BRAZIL': 'BR', 'CANADA': 'CA', 'CHILE': 'CL', 'CHINA': 'CN', 'COLOMBIA': 'CO', 'CROATIA': 'HR', 
                            'CZECHIA': 'CZ', 'DENMARK': 'DK', 'DOMINICAN REPUBLIC': 'DO', 'EGYPT': 'EG', 'ESTONIA': 'EE', 
                            'FINLAND': 'FI', 'GERMANY': 'DE', 'GREECE': 'GR', 'HONG KONG': 'HK', 'HUNGARY': 'HU', 'ICELAND': 'IS', 
                            'INDIA': 'IN', 'INDONESIA': 'ID', 'IRELAND': 'IE', 'ISRAEL': 'IL', 'ITALY': 'IT', 'KENYA': 'KE', 
                            'LITHUANIA': 'LT', 'LUXEMBOURG': 'LU', 'MALAYSIA': 'MY', 'MEXICO': 'MX', 'NETHERLANDS': 'NL', 
                            'NEW ZEALAND': 'NZ', 'NICARAGUA': 'NI', 'NIGERIA': 'NG', 'NORWAY': 'NO', 'PAKISTAN': 'PK', 
                            'PARAGUAY': 'PY', 'PERU': 'PE', 'PHILIPPINES': 'PH', 'POLAND': 'PL', 'PORTUGAL': 'PT', 
                            'RUSSIA': 'RU', 'SINGAPORE': 'SG', 'SLOVAKIA': 'SK', 'SLOVENIA': 'SI', 'SOUTH AFRICA': 'ZA', 
                            'SOUTH KOREA': 'KR', 'SPAIN': 'ES', 'SWEDEN': 'SE', 'SWITZERLAND': 'CH', 'THAILAND': 'TH', 
                            'UNITED KINGDOM': 'GB', 'UNITED STATES OF AMERICA': 'US', 'VIETNAM': 'VN'}
        country_check = user_params['prophet_country'].upper().strip()
        if country_check in robyn_country_dict:
            user_params['prophet_country'] = robyn_country_dict[country_check]
        else:
            raise Exception("Country not found")
        
        return user_params
    
    def get_multi_line_chart_data2(self, Response_df, filename):
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
        # with open(filename+'Response_Curves.json', 'w', encoding='utf-8') as f:
        #         json.dump(multi_line_chart_data2, f, ensure_ascii=False, indent=4)
        return multi_line_chart_data2

    def get_variable_list(self, df, user_params):
        """Get the list of the variables from dataset which are already selected by user in explorer
        Args:
            df: A Pandas DataFrame 
            user_params: dictionary of user inputs selected by user in explorer
        Returns: list containing the user inputs
        """  
        check_var = ['date_var', 'dep_var', 'paid_media_spends', 'paid_media_vars', 'organic_vars']
                
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
            
    def user_params_rules(self, df, user_params):
        """Defining the rules for robyn MMM based on user inputs
        Args:
            df: A Pandas DataFrame 
            user_params: dictionary of user inputs selected by user in explorer and predictor
        Returns: list containing the user inputs
        """ 
        data_var = list(df.columns)
        #Calling user inputs list in explorer
        var_used = self.get_variable_list(df, user_params)
        
        # adding holiday effect for robyn based on user input
        if user_params['include_holiday'] == True:
            user_params['prophet_vars'] = user_params['prophet_vars']+['holiday']
        del user_params['include_holiday']
        
        # adding granularity seasonality for robyn based on dataset uploaded
        if user_params['granularity'] == 'daily':
            user_params['prophet_vars'] = user_params['prophet_vars']+['weekday']
        elif user_params['granularity'] == 'weekly':
            user_params['prophet_vars'] = user_params['prophet_vars']+['monthly']
        elif user_params['granularity'] == 'monthly':
            user_params['prophet_vars'] = user_params['prophet_vars']+['monthly']
        del user_params['granularity']
    
        if user_params['prophet_country'] == None:
            user_params['prophet_country'] = 'US'
            if 'holiday' in user_params['prophet_vars']:
                user_params['prophet_vars'].remove('holiday')
        
        # defining media variables apart from spend if available, impressions or clicks are considered for each channel
        # if impression or clicks have null+missing datapoints less than 10% or null+missing percentage of spend then only these variables are considered
        # variable are selected based media spend variable names
        # if required variable is unavailable or doesn't meet the criteria for a particular channel, it's spend variable is considered
        if user_params['paid_media_vars'] == None:
            user_params['paid_media_vars'] = []
            
            for var in user_params['paid_media_spends']:
                media_name_check = [media_name for media_name in data_var if var.split('_')[0].lower() in media_name.lower()]
                media_var_check = [media_var for media_var in media_name_check if 'impression'.lower() in media_var.lower()]
                if not media_var_check:
                    media_var_check = [media_var for media_var in media_name_check if 'click'.lower() in media_var.lower()]

                if media_var_check:
                    col_check = media_var_check[0]
                    null_percentage = df[col_check].isnull().sum() / len(df) * 100
                    zero_percentage = (df[col_check] == 0).sum() / len(df) * 100
                    check_percentage = null_percentage + zero_percentage
                    col_check = var
                    null_percentage = df[col_check].isnull().sum() / len(df) * 100
                    zero_percentage = (df[col_check] == 0).sum() / len(df) * 100
                    threshold_cal = null_percentage + zero_percentage
                    threshold_cal = min(10, threshold_cal)
                    if check_percentage>threshold_cal:
                        media_var_check = [var]
                else:
                    media_var_check = [var]
                    
                user_params['paid_media_vars'] = user_params['paid_media_vars'] + [media_var_check[0]]
            
        if user_params['paid_media_vars'] == None:
            user_params['paid_media_vars'] = user_params['paid_media_spends']
            
        if 'factor_vars' not in user_params:
            user_params['factor_vars'] = []

        # Preparing contextual variables for robyn if user doesn't provide any input    
        if user_params['context_vars'] == None:
            user_params['context_vars'] = data_var
        params_to_exclude = list(set(var_used + user_params['paid_media_vars']))
        if isinstance(params_to_exclude, list):
            user_params['context_vars'] = list(set(user_params['context_vars']).difference(set(params_to_exclude)))
        else:
            user_params['context_vars'].remove(params_to_exclude)

        # Getting the robyn adstock based on the user selection            
        if 'geometric' in user_params['adstock'].lower():
            user_params['adstock']='geometric'
        elif 'cdf' in user_params['adstock'].lower():
            user_params['adstock']='weibull_cdf'
        elif 'pdf' in user_params['adstock'].lower():
            user_params['adstock']='weibull_pdf'

        # Checking for organic variable in contextual variable and adjusting the inputs    
        if user_params['organic_vars'] == None:
            user_params['organic_vars'] = [org_var for org_var in user_params['context_vars'] if 'organic'.lower() in org_var.lower()]
            user_params['context_vars'] = list(set(user_params['context_vars']) - set(user_params['organic_vars']))
        
        # Checking the contextual variable inputs if it contaians any variables with non-numerical type
        con_var_list = []
        for col in user_params['context_vars']:
            if ((df[col].dtypes == object) or (df[col].dtypes == str) or 
            (df[col].dtypes == bool) or isinstance(df[col], datetime.date) or 
            pd.api.types.is_datetime64_dtype(df[col]) or (col == user_params['dep_var'])):
                continue
            else:
                con_var_list = con_var_list + [col]
        user_params['context_vars'] = con_var_list
        
        return user_params
    
    def create_result_json(self, Response_df, output_dict, filename):
        """Creating json dump for trained model effect share
        Args:
            dictionary: effect share dictionary
            filename: path where file needs to be saved
        """ 
        Response_json = []
        Response_dict = {}
        for dim in Response_df['channel'].unique():
            Response_df_temp = Response_df[Response_df['channel']==dim]
            Response_df_temp = Response_df_temp.drop('channel', axis=1)
            Response_dict['name'] = dim
            Response_dict['values'] = Response_df_temp.to_dict(orient='records')
            Response_json = Response_json + [Response_dict]

        # with open(filename+'Response_Curves.json', 'w', encoding='utf-8') as f:
        #     json.dump(Response_json, f, ensure_ascii=False, indent=4)

        # with open(filename+'Output_Table.json', 'w', encoding='utf-8') as f:
        #     json.dump(output_dict, f, ensure_ascii=False, indent=4)
        return output_dict

    def save_response_curve_snippet(self, filename):
        """Saving snippet of response curve as PNG from one-pager
        Args:
            filename: path where file needs to be saved
        """ 
        image = Image.open(filename+"one_pager.png")
        left = 50
        upper = 5750
        right = left + 3400
        lower = upper + 1675
        cropped_image = image.crop((left, upper, right, lower))
        cropped_image.save(filename+"response_curve_snippet.png")
        image.close()

    # Define the R code with placeholders for the R list elements
    ro.r("""
    # Function to generate inputs for robyn based on user inputs
    prepare_robyn_input_var<-function(df_, robyn_input){
        
        date_var <- robyn_input$date_var
        dep_var<- robyn_input$dep_var
        dep_var_type<- robyn_input$dep_var_type
        prophet_vars<- unlist(robyn_input$prophet_vars)
        prophet_country<- robyn_input$prophet_country
        context_vars<- unlist(robyn_input$context_vars)
        paid_media_spends<- unlist(robyn_input$paid_media_spends)
        paid_media_vars<- unlist(robyn_input$paid_media_vars)
        organic_vars<- unlist(robyn_input$organic_vars)
        factor_vars<- unlist(robyn_input$factor_vars)
        window_start <- min(df_[[date_var]])
        window_end <- max(df_[[date_var]])
        adstock<- robyn_input$adstock
        
        InputCollect <- robyn_inputs(
        dt_input = df_,
        dt_holidays = dt_prophet_holidays,
        date_var = date_var,
        dep_var = dep_var,
        dep_var_type = dep_var_type,
        prophet_vars = prophet_vars,
        prophet_country = prophet_country,
        context_vars = context_vars,
        paid_media_spends = paid_media_spends,
        paid_media_vars = paid_media_vars,
        organic_vars = organic_vars,
        factor_vars = factor_vars,
        window_start = window_start,
        window_end = window_end,
        adstock = adstock
        )
        
        return(InputCollect)
    }

    # Function to generate hyperparameters for various media types for saturation effect (Hill function)
    generate_hyperparameters_saturation_effect <- function(media_types) {
        
        # Define the list of media types and parameter ranges
        alpha_range <- c(0.5, 3)
        gamma_range <- c(0.3, 1)
        
        hyperparameters <- list()
        
        for (media in media_types) {
            range <- c()
            if (grepl("alphas", media)){
                range <- alpha_range
            } else if (grepl("gammas", media)){
                range <- gamma_range
            } else{
                next
            }
            hyperparameters[[media]] <- range  
        }
        
        return(hyperparameters)
    }

    # Function to generate hyperparameters for various media types when adstock type selected is geometric
    generate_hyperparameters_geometric_ads <- function(media_types, hyperparameters) {
        
        # Define the list of media types and parameter ranges
        theta_range_tv <-  c(0.3, 0.8)
        theta_range_radio <- c(0.1, 0.4)
        theta_range_digital <- c(0, 0.3)
        
        ch_typ_tv_list <- c('tv', 'television')
        ch_typ_tv_list <- paste(ch_typ_tv_list, collapse = "|")
        
        ch_typ_radio_list <- c('radio', 'fm', 'newsletter', 'print', 'ooh', 'outdoor', 'out of home', 'newspaper', 'magazine', 'flyer', 'catalog', 'brochure', 'postcard')
        ch_typ_radio_list <- paste(ch_typ_radio_list, collapse = "|")
            
        for (media in media_types) {
            range <- c()
            if (grepl("thetas", media)){
                if (grepl(ch_typ_tv_list, media)){
                range <- theta_range_tv
                } else if (grepl(ch_typ_radio_list, media)){
                range <- theta_range_radio
                } else{
                range <- theta_range_digital
                }
            } else{
                next
            }
            hyperparameters[[media]] <- range  
        }
        
        return(hyperparameters)
    }

    # Function to generate hyperparameters for various media types when adstock type selected is weibull
    generate_hyperparameters_weibull_ads <- function(media_types, hyperparameters, adstock_type) {
        
        # Define the list of media types and parameter ranges
        shape_range_CDF <- c(0.0001, 2)
        shape_range_PDF_AllShapes <- c(0.0001, 10)
        shape_range_PDF_StrongLagged <- c(2.0001, 10)
        scale_range <- c(0, 0.1)
            
        for (media in media_types) {
            range <- c()
            if (grepl("shapes", media)){
            
                if (adstock_type == 'weibull_cdf_no_lagged'){
                    range <- shape_range_CDF
                } else if (adstock_type == 'weibull_pdf_all_shapes'){
                    range <- shape_range_PDF_AllShapes
                } else if (adstock_type == 'weibull_pdf_strong_lagged'){
                    range <- shape_range_PDF_StrongLagged
                }
            } else if (grepl("scales", media)){
                range <- scale_range
            } else{
                next
            }
            hyperparameters[[media]] <- range  
        }
        
        return(hyperparameters)
    }

    # Model Building for Robyn MMM
    robyn_model_run<-function(InputCollect, iterations, trials){

        OutputModels <- robyn_run(
            InputCollect = InputCollect,
            cores = NULL,
            iterations = iterations,
            trials = trials,
            ts_validation = TRUE,
            add_penalty_factor = FALSE # If using experimental feature
            )
            
        return(OutputModels)
        }
        
    # Model Result for Robyn MMM
    robyn_output<-function(InputCollect, OutputModels, trials){

        OutputCollect <- robyn_outputs(
            InputCollect, OutputModels,
            pareto_fronts = "auto",
            csv_out = "pareto",
            clusters = TRUE,
            plot_pareto = FALSE,
            plot_folder = "",
            export = FALSE
        )
        
        return(OutputCollect)
        }

    # Getting best candidate model
    best_candidate_model<-function(OutputFile){

        BestModel <- subset(OutputFile, top_sol == TRUE)
        
        columns_to_round <- c("rsq_test", "rsq_train", "nrmse_test", "decomp.rssd")
        BestModel[columns_to_round] <- lapply(BestModel[columns_to_round], function(x) signif(x, digits = 3))
        
        print("*****")
        print(unique(BestModel$rsq_test))
        print("*****")
        
        BestModel$rsq_diff <- abs(round((BestModel$rsq_test - BestModel$rsq_train), 3))
        
        print(BestModel$rsq_diff)
        
        BestModel <- subset(BestModel, rsq_test == max(BestModel$rsq_test))
        modelID_list <- unique(BestModel$solID)
        if(length(modelID_list) == 1){
            modelID <- modelID_list[1]
        }else{
            BestModel <- subset(BestModel, nrmse_test == min(BestModel$nrmse_test))
            modelID_list <- unique(BestModel$solID)
            if(length(modelID_list) == 1){
                modelID <- modelID_list[1]
            }else{
                BestModel <- subset(BestModel, decomp.rssd == min(BestModel$decomp.rssd))
                modelID_list <- unique(BestModel$solID)
                modelID <- modelID_list[1]
            }
        }
        
        BestModel <- subset(BestModel, solID == modelID)
        
        return(BestModel)
        }

    # Saving One-Pager for Best Candidate Model
    one_pager<-function(InputCollect, OutputCollect, modelID, filename){
        filename_temp <- paste0(filename, 'one_pager.png')
        myOnePager <- robyn_onepagers(InputCollect, OutputCollect, select_model = modelID , export = FALSE)
        ggsave(
            filename = filename_temp,
            plot = myOnePager[[modelID]], limitsize = FALSE,
            dpi = 400, width = 17, height = 19
            )
        print("** One-Pager Saved **")
        }
        
    # Saving prophet decomp
    # Note: Used directly from robyn code: https://github.com/facebookexperimental/Robyn/blob/main/R/R/plots.R (line 21-45)
    prophet_decomp_save <- function(InputCollect, OutputCollect, filename){
    
        #check_class
        x <- "robyn_outputs"
        object <- OutputCollect
        if (any(!x %in% class(object))) stop(sprintf("Input object must be class %s", x))

        #prophet_decomp
        pareto_fronts <- OutputCollect$pareto_fronts
        hyper_fixed <- OutputCollect$hyper_fixed
        temp_all <- OutputCollect$allPareto

        if (!hyper_fixed) {
            if (!is.null(InputCollect$prophet_vars) && length(InputCollect$prophet_vars) > 0 ||
            !is.null(InputCollect$factor_vars) && length(InputCollect$factor_vars) > 0) {
                
                dt_plotProphet <- InputCollect$dt_mod %>%
                select(c("ds", "dep_var", InputCollect$prophet_vars, InputCollect$factor_vars)) %>%
                tidyr::gather("variable", "value", -.data$ds) %>%
                mutate(ds = as.Date(.data$ds, origin = "1970-01-01"))
                pProphet <- ggplot(
                    dt_plotProphet, aes(x = .data$ds, y = .data$value)
                    ) +
                geom_line(color = "steelblue") +
                facet_wrap(~ .data$variable, scales = "free", ncol = 1) +
                labs(title = "Prophet decomposition", x = NULL, y = NULL) +
                theme_lares(background = "white", ) +
                scale_y_abbr()

                ggsave(
                    paste0(filename, "prophet_decomp.png"),
                    plot = pProphet, limitsize = FALSE,
                    dpi = 600, width = 12, height = 3 * length(unique(dt_plotProphet$variable))
                    )
                }
            }
        print("** prophet decomp Saved **")
        }

    # Getting response curves
    get_response_curves <- function(channel_list, InputCollect, OutputCollect, modelID){
        response_df = data.frame(matrix(ncol = 3, nrow = 0))
        colnames(response_df) = c('channel', 'spend', 'target')
        for (ch in channel_list){
            response_ch <- robyn_response(InputCollect = InputCollect,
                                        OutputCollect = OutputCollect,
                                        select_model = modelID,
                                        metric_name = ch)
            response_temp <- data.frame(channel = ch, spend = response_ch$input_total, target = response_ch$response_total)
            response_df <- rbind(response_df, response_temp)
            }
            
        response_df <- response_df[order(response_df$channel, response_df$spend), ]
        row.names(response_df) <- NULL

        return(response_df)
        }

    # main function for robyn r code         
    robyn_code<-function(df_, robyn_input, adstock_type, iterations, trials, filename){
        # install.packages('reticulate')
        library(reticulate)
        virtualenv_create('r-reticulate')
        use_virtualenv('r-reticulate', required = TRUE)
        Sys.setenv(RETICULATE_PYTHON = '/usr/local/bin/python3.10')
        py_install('nevergrad', pip = TRUE)
        py_install('numpy', pip = TRUE)  
        py_config()
        nevergrad <- import("nevergrad")
        cl <- makeCluster(4)
        registerDoParallel(cl)
        start_time <- Sys.time()
        print("**** start_time ****")
        print(start_time)
        Sys.setenv(R_FUTURE_FORK_ENABLE = "true")
        options(future.fork.enable = TRUE)
        
        create_files <- TRUE
        
        #print((df_[robyn_input$date_var][1]))
        #df_[robyn_input$date_var][1] <- as.Date(df_[robyn_input$date_var][1])
        
        data("dt_prophet_holidays")
        
        print("** Robyn Input **")
        InputCollect <- prepare_robyn_input_var(df_, robyn_input)
        print(InputCollect)
        
        print("** Generate Hyperparameter List **")
        hyper_names_list <- hyper_names(adstock = InputCollect$adstock, all_media = InputCollect$all_media)
        print(hyper_names_list)
        
        plot_adstock(plot = FALSE)
        plot_saturation(plot = FALSE)
        
        print("** Refernce Hyperparameter Bounds **")
        print(hyper_limits())
        
        print("** Saturation Effect Hyperparamaters **")
        # Generate hyperparameters based on the media types and ranges for saturation effect (Hill function)
        hyperparameters <- generate_hyperparameters_saturation_effect(hyper_names_list)
        
        print("** Adstock Effect Hyperparamaters **")
        if (robyn_input$adstock == 'geometric'){
            # Generate hyperparameters based on the media types and ranges when adstock is geometric
            hyperparameters <- generate_hyperparameters_geometric_ads(hyper_names_list, hyperparameters)
        } else{
            # Generate hyperparameters based on the media types and ranges when adstock is weibull cdf or weibull pdf
            hyperparameters <- generate_hyperparameters_weibull_ads(hyper_names_list, hyperparameters, adstock_type)
        }
        
        print("** Train-Test Hyperparamaters **")
        # Hyperparameter for training, validation and testing
        hyperparameters[['train_size']] = c(0.7, 0.8)
        
        print("** Robyn Input with Hyperparameters **")
        InputCollect <- robyn_inputs(InputCollect = InputCollect, hyperparameters = hyperparameters)
        
        print("** Robyn Model Running **")
        # Model Building for Robyn MMM
        OutputModels <- robyn_model_run(InputCollect, iterations, trials)
        print("** Robyn Model Output **")
        print(OutputModels)
        
        print("** Robyn Model Result **")
        # Model Output for Robyn MMM
        OutputCollect <- robyn_output(InputCollect, OutputModels)
        print("** Robyn Result **")
        #print(OutputCollect)
        
        print("** Generating Output File **")
        OutputFile <- OutputCollect$xDecompAgg
        #print(OutputFile)
        
        print("** Selecting Best Candidate Model **")
        BestModel <- best_candidate_model(OutputFile)
        modelID <- BestModel$solID[1]
        print(modelID)
        BestModel_df <- as.data.frame(BestModel)
        
        print("** Saving One-Pager for Selected Model **")
        ## one_pager(InputCollect, OutputCollect, modelID, filename)
        
        print("** Saving prophet decomp **")
        ## prophet_decomp_save(InputCollect, OutputCollect, filename)
        
        print("** Generating response curves **")
        channel_list <- robyn_input$paid_media_spends
        response_df <- get_response_curves(channel_list, InputCollect, OutputCollect, modelID)
        
        print("** Saving Model **")
        ## ExportedModel <- robyn_write(InputCollect, OutputCollect, modelID, export = TRUE, dir=filename)
        
        df_return <- list(df1 = BestModel_df, df2 = response_df)
        end_time <- Sys.time()
        print("****end time****")
        print(end_time)
        stopCluster(cl)
        return(df_return)
        }
        
    """)

    def execute(self, explorer_user_input, predictor_user_params, filename):
        """main function to execute robyn mmm and other functionalities
        Args: 
            explorer_user_input: explorer page user inputs
            predictor_user_params: predictor page user inputs
            file_path: file path where EDA report has to be saved
        Returns:
            DataFrame: Response_df_py: response curves for channels
            Dictionary: output_dict: effect share for channels and adj r2 for test
        """
        #filename = None
        df = self.df

        # # generating uniquw file path for each model
        # timestamp_ = str(datetime.datetime.now())
        # timestamp_ = str(timestamp_.replace("-","_").replace(":","").replace(".","").replace(" ",""))
        # filename = filename+"Model_"+timestamp_
        # Path(filename).mkdir(parents=True, exist_ok=True)
        # filename = filename+"\\"

        # defining seasonal and other variables for robyn
        user_params = {
            'prophet_vars': ['trend', 'season'],
            'paid_media_vars': None,
            'organic_vars': None
            }
        
        # merging inputs fron explorer and predictor
        user_params = {**predictor_user_params, **user_params}
        user_params = {**explorer_user_input, **user_params}

        # pre-processing country and adstock for robyn
        user_params['prophet_country'] = user_params['prophet_country'].upper()
        user_params['adstock'] = user_params['adstock'].lower()

        user_params['adstock']=self.adstock_list[user_params['adstock'].lower()]

        # defining model iterations and trials
        model_iterations = 2500
        model_trials = 5
        if user_params['adstock'] != 'geometric_traditional':
            model_iterations = 2500
            model_trials = 8
        
        # format the date column
        _format = "%Y-%m-%d"
        if pd.to_datetime(df[user_params['date_var']], format=_format, errors='coerce').notnull().all(): 
            df[user_params['date_var']] = pd.to_datetime(df[user_params['date_var']], format=_format)
        df[user_params['date_var']] = df[user_params['date_var']].astype(str)
        
        user_params_updated = copy.deepcopy(user_params)
        user_params_updated = self.get_country_code(user_params_updated)
        user_params_updated = self.user_params_rules(df, user_params_updated)

        df=df.fillna(0)
        
        # converting python dictionary into r code readable format
        r_user_params = robjects.ListVector(user_params_updated)
            
        adstock_type = user_params['adstock']

        robyn_code = ro.globalenv['robyn_code']

        # executing robyn code and saving response
        df_return = robyn_code(df, r_user_params, adstock_type, model_iterations, model_trials, filename)
        df1 = df_return[0]
        df2 = df_return[1]
        print("** Converting Dataframe: R to Python **")
        with localconverter(ro.default_converter + pandas2ri.converter):
            BestModel_df_py = ro.conversion.rpy2py(df1)
        with localconverter(ro.default_converter + pandas2ri.converter):
            Response_df_py = ro.conversion.rpy2py(df2)
        
        # checking difference between adj r2 for test and train
        check_rsq_diff = 0
        print("** Checking Adjusted R Squared Difference **")
        check_rsq_diff = np.floor(BestModel_df_py['rsq_diff'][0] * 100)
        if (check_rsq_diff) <= 10:
            check_rsq_diff = None
        else:
            check_rsq_diff = "Difference of adjusted r-squared between train and test is greater than 10%: " + str(check_rsq_diff) +"%"
        
        # getting adj r2 for test for the best model
        adj_rsq = round(BestModel_df_py['rsq_test'][0] * 100, 2)

        # getting effect share of the channels
        print("** Getting Effect Share of the channels **")
        effect_share_df = BestModel_df_py[['rn', 'effect_share']]
        effect_share_df = effect_share_df[effect_share_df['rn'].isin(user_params_updated['paid_media_spends'])].reset_index(drop=True)
        effect_share_df['effect_share'] = round(effect_share_df['effect_share']*100, 2)
        effect_share_df = effect_share_df.rename({'rn':'channel'}, axis=1)
        effect_share_dict = {v['channel']: v['effect_share'] for v in effect_share_df.T.to_dict().values()}

        # creating output json for results to be displayed
        output_dict = {
            'check_rsq_diff' : check_rsq_diff,
            'adj_rsq_test' : adj_rsq,
            'effect_share' : effect_share_dict}
        
        print("** Saving Results **")
        # saving model and it's response curves and effect share
        output_table = self.create_result_json(Response_df_py, output_dict, filename)
        write_json_to_gcs(BUCKET_NAME, filename, output_table,file_name_to_be_put_gcs='Output_Table.json')
        # self.save_response_curve_snippet(filename)
        response_curves_json = self.get_multi_line_chart_data2(Response_df_py, filename)
        write_json_to_gcs(BUCKET_NAME, filename, response_curves_json,file_name_to_be_put_gcs='Response_Curves.json')
        print("********************************************")
        ## clear the redis cache
        # Run the Celery task to clear the Redis cache
        return output_table,response_curves_json