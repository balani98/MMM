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
import logging
import os
import warnings
from google.cloud import storage
warnings.filterwarnings('ignore')
import configparser
Config = configparser.ConfigParser()
from push_gcs import write_json_to_gcs
import math
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
# os.environ["R_HOME"] = r"C:\Program Files\R\R-4.4.0"

# Load in python libraries to use R
from rpy2.robjects.packages import importr
import rpy2.interactive as r
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import rpy2.robjects.numpy2ri as rpyn
import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter
pandas2ri.activate()

#import nevergrad

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
class robyn_predictor:
    def __init__(self):
        """
        initialization of the class
        date: date variable in the data
        channel_spend: list of channels and their repective spends
        target: dependent variable in the data
        target_type: select target type as revenue or conversion
        granularity_selected: granulaity of the uploaded dataset
        """
        self.df = pd.DataFrame()

        self.adstock_list = {'exponential fixed decay' : 'geometric_traditional', 
                             'flexible decay with no lag' : 'weibull_cdf_no_lagged', 
                             'flexible decay with lag' : 'weibull_pdf_all_shapes'}
        
    def read_file(self,blob_name):
        client = storage.Client.from_service_account_json(SERVICE_ACCOUNT)
        bucket = client.bucket("global_mso_data_files")
        upload = client.bucket("upload_folder")
        blob = bucket.blob(blob_name)
        with blob.open("r") as data:
            df = pd.read_csv(data)
        return df

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
        country_check = user_params['country'].upper().strip()
        if country_check in robyn_country_dict:
            user_params['country'] = robyn_country_dict[country_check]
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
            
    def robyn_user_params_rules(self, user_params):
        """Defining the rules for robyn MMM based on user inputs
        Args:
            user_params: dictionary of user inputs selected by user in explorer and predictor
        Returns: list containing the user inputs
        """ 
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

        # if holiday effect is not be included and country is not provided
        if user_params['country'] == None:
            user_params['country'] = 'US'
            if 'holiday' in user_params['prophet_vars']:
                user_params['prophet_vars'].remove('holiday')
            
        if user_params['media_variables'] == None:
            user_params['media_variables'] = user_params['spend_variables']
            
        if 'factor_vars' not in user_params:
            user_params['factor_vars'] = []

        # Getting the robyn adstock based on the user selection            
        if 'geometric' in user_params['adstock'].lower():
            user_params['adstock']='geometric'
        elif 'cdf' in user_params['adstock'].lower():
            user_params['adstock']='weibull_cdf'
        elif 'pdf' in user_params['adstock'].lower():
            user_params['adstock']='weibull_pdf'
        
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

        with open(filename+'Output_Table.json', 'w', encoding='utf-8') as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=4)

    def save_best_robyn_model_output(self, BestModel_df_py, Model_Result_df_py, filename):
        try:
            BestModel_df_py.to_csv(filename+'BestModelRobynResult.csv', index=False)
            # Model_Result_df_py.to_csv(filename+'ModelResult.csv', index=False)
        except:
            print("CUSTOM WARNING: Results for Best Model for Robyn could not be saved")

    def save_response_curve_snippet(self, filename):
        """Saving snippet of response curve as PNG from one-pager
        Args:
            filename: path where file needs to be saved
        """ 
        try:
            image = Image.open(filename+"one_pager.png")
            left = 50
            upper = 5750
            right = left + 3400
            lower = upper + 1675
            cropped_image = image.crop((left, upper, right, lower))
            cropped_image.save(filename+"response_curve_snippet.png")
            image.close()
        except:
            print("CUSTOM WARNING: Response curve snippet could not be generated as one_pager does not exist")

    # Define the R code with placeholders for the R list elements
    ro.r("""
    # Function to generate inputs for robyn based on user inputs
    prepare_robyn_input_var<-function(df_, robyn_input){
        
        date_var <- robyn_input$date_variable
        dep_var<- robyn_input$target_variable
        dep_var_type<- robyn_input$target_variable_type
        prophet_vars<- unlist(robyn_input$prophet_vars)
        prophet_country<- robyn_input$country
        context_vars<- unlist(robyn_input$control_variables)
        paid_media_spends<- unlist(robyn_input$spend_variables)
        paid_media_vars<- unlist(robyn_input$media_variables)
        organic_vars<- unlist(robyn_input$organic_variables)
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

    # # Getting best candidate model
    # best_candidate_model<-function(OutputFile){

    #     BestModel <- subset(OutputFile, top_sol == TRUE)
        
    #     columns_to_round <- c("rsq_test", "rsq_train", "nrmse_test", "decomp.rssd")
    #     BestModel[columns_to_round] <- lapply(BestModel[columns_to_round], function(x) signif(x, digits = 3))
        
    #     print("*****")
    #     print(unique(BestModel$rsq_test))
    #     print("*****")
        
    #     BestModel$rsq_diff <- abs(round((BestModel$rsq_test - BestModel$rsq_train), 3))
        
    #     print(BestModel$rsq_diff)
        
    #     BestModel <- subset(BestModel, rsq_test == max(BestModel$rsq_test))
    #     modelID_list <- unique(BestModel$solID)
    #     if(length(modelID_list) == 1){
    #         modelID <- modelID_list[1]
    #     }else{
    #         BestModel <- subset(BestModel, nrmse_test == min(BestModel$nrmse_test))
    #         modelID_list <- unique(BestModel$solID)
    #         if(length(modelID_list) == 1){
    #             modelID <- modelID_list[1]
    #         }else{
    #             BestModel <- subset(BestModel, decomp.rssd == min(BestModel$decomp.rssd))
    #             modelID_list <- unique(BestModel$solID)
    #             modelID <- modelID_list[1]
    #         }
    #     }
        
    #     BestModel <- subset(BestModel, solID == modelID)
          
    #     return(BestModel)
    #     }
         

    # Getting best candidate model using RSSD, Difference of Train and Val & Max Adj Trian
    best_candidate_model <- function(OutputFile) {
        # Round certain columns to three significant digits
        columns_to_round <- c("rsq_val", "rsq_train", "decomp.rssd")
        OutputFile[columns_to_round] <- lapply(OutputFile[columns_to_round], function(x) signif(x, digits = 4))
        
        # Filter models where Decomposition RSSD < 0.15
        BestModel <- subset(OutputFile, decomp.rssd <= 0.15)
        
        # If there is no model with RSSD < 0.15, select the next best model
        if (nrow(BestModel) == 0) {
            BestModel <- subset(OutputFile, decomp.rssd == min(OutputFile$decomp.rssd))
            BestModel <- subset(BestModel, rsq_train == max(BestModel$rsq_train))
        } else {
            # Calculate the difference in R-squared
            BestModel$rsq_diff <- abs(BestModel$rsq_val - BestModel$rsq_train)
            
            # Filter models where the absolute difference in R-squared < 0.15
            BestModelTemp <- BestModel
            BestModel <- subset(BestModel, rsq_diff <= 0.15)
            
            # If there is no model with R-squared difference < 0.15, select the next best model
            if (nrow(BestModel) == 0) {
                BestModel <- subset(BestModelTemp, rsq_diff == min(BestModelTemp$rsq_diff))
                BestModel <- subset(BestModel, rsq_train == max(BestModel$rsq_train))
            } else {
                # Select the best model based on criteria
                BestModel <- subset(BestModel, rsq_train == max(BestModel$rsq_train))
            }
        }

        # Export the selected model
        return(BestModel)
    }

    # Saving One-Pager for Best Candidate Model
    one_pager<-function(InputCollect, OutputCollect, modelID, filename){
        filename_temp <- paste0(filename, 'one_pager.PNG')
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
                    paste0(filename, "prophet_decomp.PNG"),
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
    robyn_code<-function(df_, robyn_input, adstock_type, iterations, trials, train_size, filename){
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
                
        data("dt_prophet_holidays")
        
        print("** Robyn Input **")
        InputCollect <- prepare_robyn_input_var(df_, robyn_input)
        # print(InputCollect)
        
        print("** Generate Hyperparameter List **")
        hyper_names_list <- hyper_names(adstock = InputCollect$adstock, all_media = InputCollect$all_media)
        # print(hyper_names_list)
        
        plot_adstock(plot = FALSE)
        plot_saturation(plot = FALSE)
        
        print("** Refernce Hyperparameter Bounds **")
        # print(hyper_limits())
        
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
        hyperparameters[['train_size']] = train_size
        
        print("** Robyn Input with Hyperparameters **")
        
        InputCollect <- robyn_inputs(InputCollect = InputCollect, hyperparameters = hyperparameters)
        
        print("** Robyn Model Running **")
        # Model Building for Robyn MMM
        OutputModels <- robyn_model_run(InputCollect, iterations, trials)
        print("** Robyn Model Output **")
        # print(OutputModels)
        
        print("** Robyn Model Result **")
        # Model Output for Robyn MMM
        OutputCollect <- robyn_output(InputCollect, OutputModels)
        print("** Robyn Result **")
        #print(OutputCollect)
        
        print("** Generating Output File **")
        OutputFile <- OutputCollect$xDecompAgg
        # print(OutputFile)
        
        Outputtemp <- OutputCollect$xDecompVecCollect 
        #print(Outputtemp)

        print("** Selecting Best Candidate Model **")
        BestModel <- best_candidate_model(OutputFile)
        modelID <- BestModel$solID[1]
        print(modelID)
        BestModel_df <- as.data.frame(BestModel)
        Outputtemp <- subset(Outputtemp,solID = modelID)
        Outputtemp <- subset(Outputtemp, select = c("ds", "dep_var", "depVarHat"))
        Actualvsfitted_df <- as.data.frame(Outputtemp)
          
        # print("** Saving One-Pager for Selected Model **")
        tryCatch(
        {
            #one_pager(InputCollect, OutputCollect, modelID, filename)
        },
        error = function(e) {
                cat("CUSTOM WARNING: One-pager could not be generated, error occured. Message", e$message, "\n")
            }
        )
        
        print("** Saving prophet decomp **")
        tryCatch(
        {
            #prophet_decomp_save(InputCollect, OutputCollect, filename)
        },
        error = function(e) {
                cat("CUSTOM WARNING: Prophet Decomposed chart could not be generated, error occured. Message", e$message, "\n")
            }
        )
        
        print("** Generating response curves **")
        channel_list <- robyn_input$spend_variables
        response_df <- get_response_curves(channel_list, InputCollect, OutputCollect, modelID)
        
        print("** Saving Model **")
        tryCatch(
        {
            #ExportedModel <- robyn_write(InputCollect, OutputCollect, modelID, export = TRUE, dir=filename)
        },
        error = function(e) {
                cat("CUSTOM WARNING: Model could not be exported/saved, error occured. Message", e$message, "\n")
            }
        )
        
        df_return <- list(df1 = BestModel_df, df2 = response_df, df3 = Actualvsfitted_df, df4 = OutputFile)
        end_time <- Sys.time()
        print("****end time****")
        print(end_time)
        stopCluster(cl)
        return(df_return)
        }
        
    """)

    def get_r_squared(self, df_temp, type, train_size, val_test_size):
        train_rows = int(round(self.df.shape[0]*train_size))
        if type == 'train':
            df_smape = df_temp.head(train_rows)
        else:
            test_rows = int(round(self.df.shape[0]*(train_size+val_test_size)))
            df_smape = df_temp.iloc[train_rows:test_rows]
        y_true = df_smape['dep_var']
        y_pred = df_smape['depVarHat']
        y_mean = np.mean(y_true)
        ss_total = np.sum((y_true - y_mean) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)
    
    def convert_nrmse_to_rmse(self, nrmse, type, train_size, val_test_size, user_params):
        train_rows = int(round(self.df.shape[0]*train_size))
        if type == 'train':
            y = self.df[user_params['target_variable']].head(train_rows)
        else:
            test_rows = int(round(self.df.shape[0]*(train_size+val_test_size)))
            y = self.df[user_params['target_variable']].iloc[train_rows:test_rows]
        rmse = nrmse * (max(y) - min(y))
        return rmse
    
    def SMAPE(self, actual, pred):
        """calculating Std. Error of Residual
        Args:
            actual (series): target
            pred (series): predicted target
        Returns:
            float: SMAPE
        """
        smape = abs(actual-pred)/(actual + pred)
        smape = np.mean(smape[~smape.isna()])
        return smape
        
    def get_smape(self, df_temp, type, train_size, val_test_size):
        train_rows = int(round(self.df.shape[0]*train_size))
        if type == 'train':
            df_smape = df_temp.head(train_rows)
        else:
            test_rows = int(round(self.df.shape[0]*(train_size+val_test_size)))
            df_smape = df_temp.iloc[train_rows:test_rows]
        
        smape = self.SMAPE(df_smape['dep_var'], df_smape['depVarHat'])
        
        return smape
    
    def get_spend_and_effect_share(self, BestModel_df_py, user_params_updated):
        effect_share_df = BestModel_df_py[['rn', 'spend_share', 'effect_share']]
        effect_share_df = effect_share_df[effect_share_df['rn'].isin(user_params_updated['spend_variables'])].reset_index(drop=True)
        effect_share_df['spend_share'] = round(effect_share_df['spend_share']*100, 2)
        effect_share_df['effect_share'] = round(effect_share_df['effect_share']*100, 2)
        effect_share_df = effect_share_df.rename({'rn':'channel'}, axis=1)
        effect_share_dict = {v['channel']: [v['spend_share'], v['effect_share']] for v in effect_share_df.T.to_dict().values()}
        return effect_share_dict
    
    def convert_special_floats(self, obj):
        if isinstance(obj, float):
            if math.isinf(obj):
                return 'Infinity' if obj > 0 else '-Infinity'
        return obj

    def custom_encoder(self, obj):
        if isinstance(obj, dict):
            return {k: self.custom_encoder(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.custom_encoder(v) for v in obj]
        else:
            return self.convert_special_floats(obj)
    
    def get_metrics(self, BestModel_df_py, Actualvsfitted_df_py, train_size, val_test_size, user_params):
        metrics = {
                'rssd': BestModel_df_py['decomp.rssd'][0],
                'r2_train': self.get_r_squared(Actualvsfitted_df_py, "train", train_size, val_test_size),
                'r2_test': self.get_r_squared(Actualvsfitted_df_py, "test", train_size, val_test_size),
                'adjusted_r2_train': BestModel_df_py['rsq_train'][0],
                'adjusted_r2_test': BestModel_df_py['rsq_val'][0],
                'rmse_train': self.convert_nrmse_to_rmse(BestModel_df_py['nrmse_train'][0], "train", train_size, val_test_size, user_params),
                'rmse_test': self.convert_nrmse_to_rmse(BestModel_df_py['nrmse_val'][0], "test", train_size, val_test_size, user_params),
                'nrmse_train': BestModel_df_py['nrmse_train'][0],
                'nrmse_test': BestModel_df_py['nrmse_val'][0],
                'mape_train':self.get_smape(Actualvsfitted_df_py, "train", train_size, val_test_size),
                'mape_test':self.get_smape(Actualvsfitted_df_py, "test", train_size, val_test_size),
                'diff_adjusted_r2': abs(BestModel_df_py['rsq_train'][0]-BestModel_df_py['rsq_val'][0])
                }
        
        converted_metric = self.custom_encoder(metrics)
        return converted_metric

    def get_model_trials_and_iterations(self, user_params):
        model_iterations = 0
        model_trials = 0
        if user_params['adstock'] == 'geometric_traditional':
            if (user_params['model_iterations'] == None) | (user_params['model_iterations'] == 0):
                model_iterations = user_params['model_iterations'] = 5000 #5000
            if (user_params['model_trials'] == None) | (user_params['model_trials'] == 0):
                model_trials = user_params['model_trials'] = 5 #5
        else:
            if (user_params['model_iterations'] == None) | (user_params['model_iterations'] == 0):
                model_iterations = user_params['model_iterations'] = 10000
            if (user_params['model_trials'] == None) | (user_params['model_trials'] == 0):
                model_trials = user_params['model_trials'] = 5

        return user_params, model_iterations, model_trials
    
    def execute(self, robyn_user_params, filename):
        try:
            """main function to execute robyn mmm and other functionalities
            Args: 
                explorer_user_input: explorer page user inputs
                predictor_user_params: predictor page user inputs
                file_path: file path where EDA report has to be saved
            Returns:
                DataFrame: Response_df_py: response curves for channels
                Dictionary: output_dict: effect share for channels and adj r2 for test
            """
            # filename = ""

            if robyn_user_params["include_holiday"] == "True":
                robyn_user_params["include_holiday"] == True
            elif robyn_user_params["include_holiday"] == "False":
                robyn_user_params["include_holiday"] == False
            else:
                robyn_user_params["include_holiday"] == False
                
            # Create file handler for INFO logs
            logging.basicConfig(level=logging.INFO)
            info_handler = logging.FileHandler('robyn_info.log')
            info_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            info_handler.setFormatter(formatter)
            logging.getLogger('').addHandler(info_handler)
            logging.info("Start: Robyn")

            # Reading file from stored location
            self.df = self.read_file(robyn_user_params['filename'])
            logging.info("File has been read successfully")

            # Pre-Processing for robyn model inputs
            logging.info("Initiating Pre-processing for robyn model required inputs")
            # making copy of user input before altering it for the code
            user_params = copy.deepcopy(robyn_user_params)
            user_params['model_iterations'] = None
            user_params['model_trials'] = None

            # defining seasonal and other variables for robyn in user dictionary
            user_params['prophet_vars'] = ['trend', 'season']

            # pre-processing country and adstock for robyn
            user_params['country'] = user_params['country'].upper()
            user_params['adstock'] = user_params['adstock'].lower()
            user_params['adstock']=self.adstock_list[user_params['adstock'].lower()]
            adstock_type = user_params['adstock']
            # defining model iterations and trials
            user_params, model_iterations, model_trials = self.get_model_trials_and_iterations(user_params)

            # defining train, validation and test split
            train_size = 0.9
            val_test_size = (1-train_size)/2

            # format the date column
            _format = "%Y-%m-%d"
            if pd.to_datetime(self.df[user_params['date_variable']], format=_format, errors='coerce').notnull().all(): 
                self.df[user_params['date_variable']] = pd.to_datetime(self.df[user_params['date_variable']], format=_format)
            self.df[user_params['date_variable']] = self.df[user_params['date_variable']].astype(str)
            
            # defining user input rules for robyn and getting country code
            user_params = self.get_country_code(user_params)
            user_params = self.robyn_user_params_rules(user_params)

            # fill na with 0 values as robyn could not handle na values
            self.df=self.df.fillna(0)
            logging.info("Pre-processing successfull")
            
            # Converting Python objects to R objects for Robyn
            # converting python dictionary into r code readable format
            r_user_params = robjects.ListVector(user_params)
            robyn_code = ro.globalenv['robyn_code']
            # with localconverter(ro.default_converter + pandas2ri.converter):
            #     rdf = ro.conversion.py2rpy(df)
            logging.info("Convertion of Python objects to R objects successfull for Robyn")

            # executing robyn code and saving response - R Code
            logging.info("Initializing model training for Robyn in R code")
            df_return = robyn_code(self.df, r_user_params, adstock_type, model_iterations, model_trials, train_size, filename)
            df1 = df_return[0]
            df2 = df_return[1]
            df3 = df_return[2]
            df4 = df_return[3]
            logging.info("Model trained sucessfully")

            print("** Converting Dataframe: R to Python **")
            with localconverter(ro.default_converter + pandas2ri.converter):
                BestModel_df_py = ro.conversion.rpy2py(df1)
            with localconverter(ro.default_converter + pandas2ri.converter):
                Response_df_py = ro.conversion.rpy2py(df2)
            with localconverter(ro.default_converter + pandas2ri.converter):
                Actualvsfitted_df_py = ro.conversion.rpy2py(df3)
            with localconverter(ro.default_converter + pandas2ri.converter):
                Model_Result_df_py = ro.conversion.rpy2py(df4)
            logging.info("Convertion of R objects to Python objects successfull for Robyn Output")

            # getting effect share of the channels
            print("** Getting Effect Share of the channels **")
            effect_share_dict = self.get_spend_and_effect_share(BestModel_df_py, user_params)
            logging.info("Effect share computed successfully")
                    
            print("** Creating Output json and calculating metrics**")
            # saving model and it's response curves and effect share
            # self.create_result_json(Response_df_py, output_dict, filename)
            #self.save_best_robyn_model_output(BestModel_df_py, Model_Result_df_py, filename)
            #self.save_response_curve_snippet(filename)
            response_curve_json = self.get_multi_line_chart_data2(Response_df_py, filename)
            logging.info("Genearting response curve and saving model results completed")
            #write_json_to_gcs(BUCKET_NAME, filename, response_curve_json,file_name_to_be_put_gcs='Response_Curves.json')
            # creating output json for results to be displayed
            output_dict = {
                'metrics': self.get_metrics(BestModel_df_py, Actualvsfitted_df_py, train_size, val_test_size, user_params),
                'effective_share' : effect_share_dict,
                'response_curve': response_curve_json,
                'response_curve_stats': {
                    'spend': [Response_df_py['spend'].min(), Response_df_py['spend'].max()],
                    'target': [Response_df_py['target'].min(), Response_df_py['target'].max()]
                    }
                }
            logging.info("Metrics calculated successfully")
            logging.info("Initiating saving output to GCS")
            write_json_to_gcs(BUCKET_NAME, filename, output_dict, file_name_to_be_put_gcs='Output.json')
            logging.info("output saved to GCS")
            logging.info("Completed: Robyn")
            logging.info("5002")
        
            print("********************************************")
            return output_dict
        except Exception as error:
            logging.error("Failed: Robyn")
            logging.error("5003")