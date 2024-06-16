from itertools import dropwhile
import pandas as pd
import numpy as np
from pandas.api.types import is_categorical_dtype
from pandas.api.types import is_numeric_dtype, is_float_dtype
import datetime
import ydata_profiling as ydp
import json
import warnings
warnings.filterwarnings('ignore')


class explorer:
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
        self.date = None
        self.channel_spend = None
        self.target = None
        self.target_type = None
        self.granularity_selected = None

        if self.granularity_selected != None:
            self.granularity_selected = self.granularity_selected.lower()

        self.check_dict = {
        'daily': {'min': 365, 'max': ((365 * 5) + 2), 'type': 'days'},
        'weekly': {'min': 104, 'max': 260, 'type': 'weeks'},
        'monthly': {'min': 36, 'max': 60, 'type': 'months'}
        }

    def is_date_column(self, column):
        try:
            pd.to_datetime(column)
            return True
        except ValueError:
            return False

    def onSubmit_selection(self):
        """checking for different user selection types (date, channel spend, target)
        Args:
            df: A Pandas DataFrame.
        Returns: Dictionary containing the list for each user selection type
        """
        user_selection = {}

        check_date_cols = [col for col in self.df.columns if self.is_date_column(self.df[col])]
        check_date_cols = list(set(check_date_cols + [col for col in self.df.columns if isinstance(self.df[col], datetime.date)]))
        if not check_date_cols:
            check_date_cols = list(set(self.df.columns))
        check_date_cols_with_delimeter = [col for col in check_date_cols if ((self.df[col].astype(str).str.contains('-')).all() or (self.df[col].astype(str).str.contains('/')).all())]
        if not check_date_cols_with_delimeter:
            check_date_cols_with_delimeter = list(set(self.df.columns))
        user_selection['date'] = check_date_cols_with_delimeter

        check_numeric_cols = list(set(self.df.select_dtypes(include=['int', 'float', 'int32', 'float32', 'int64', 'float64']).columns))
        if not check_numeric_cols:
            check_numeric_cols = list(set(self.df.columns))
        user_selection['channel_spend'] = sorted([col for col in check_numeric_cols if (('spend' in col.lower()) or 
                                                                                 ('cost' in col.lower()) or 
                                                                                 ('investment' in col.lower()))])
        user_selection['target'] = sorted([col for col in check_numeric_cols if (('order' in col.lower()) or 
                                                                          ('sold' in col.lower()) or 
                                                                          ('visit' in col.lower()) or 
                                                                          ('target' in col.lower()) or 
                                                                          ('converstion' in col.lower()) or 
                                                                          ('revenue' in col.lower()) or 
                                                                          ('dependent' in col.lower()) or 
                                                                          ('install' in col.lower()) or 
                                                                          ('lead' in col.lower()))])

        return user_selection

    def target_numeric_check(self):
        """perform data validation for target column, null values
        Args:
            numeric_ (_type_): target
        Raises:
            Exception 5002: value error
            Exception 5003: type error
        """
        row_count = self.df.shape[0]

        null_val = self.df[self.target].isnull().sum()

        if null_val == row_count:
            raise Exception(5002)

        if not ((is_numeric_dtype(self.df[self.target])) | (is_float_dtype(self.df[self.target]))):
                raise Exception(5003)
        
    def date_check(self):
        """perform data audit for date format, null values
        Raises:
            Exception 5002: value error
            Exception 5004: date format error
        """
        _format = "%Y-%m-%d"

        null_val = self.df[self.date].isnull().sum()

        if not null_val == 0:
            raise Exception(5002)
        
        if ((is_numeric_dtype(self.df[self.date])) | (is_float_dtype(self.df[self.date]))):
            raise Exception(5002)
        
        try:
            self.df[self.date] = pd.to_datetime(self.df[self.date], format=_format)
        except:
            raise Exception(5004)
        
    def get_data_granularity(self):
        """Getting data granularity based on date column selected and dataset uploaded
        Args:
            df: A Pandas DataFrame.
        Returns: Variable containing the granularity: Daily, Weekly, Monthly, Yearly
        """
        _format = "%Y-%m-%d"
        if pd.to_datetime(self.df[self.date], format=_format, errors='coerce').notnull().all(): 
            self.df[self.date] = pd.to_datetime(self.df[self.date], format=_format)
        
        date_diffs = self.df[self.date].diff()
        date_diffs = date_diffs[1:]
        granularity = None
        if (date_diffs.dt.days == 1).all():
            granularity = 'daily'
        elif (date_diffs.dt.days == 7).all():
            granularity = 'weekly'
        elif (date_diffs.dt.days >= 28).all() and (date_diffs.dt.days <= 31).all():
            granularity = 'monthly'
        elif (date_diffs.dt.days >= 365).all() and (date_diffs.dt.days <= 366).all():
            granularity = 'yearly'

        self.granularity_selected = granularity
        if self.granularity_selected != None:
            self.granularity_selected = self.granularity_selected.lower()
            
        return granularity

    def channel_spend_check(self):

        """checking for null and data type in channel spends
        Raises:
            Exception 5002: value error
            Exception 5003: type error
        """
        null_val = self.df[self.channel_spend].isna().any(1).sum()

        row_count = self.df.shape[0]

        if null_val == row_count:
            raise Exception(5002)

        for chl in self.channel_spend:
            if not ((is_numeric_dtype(self.df[chl])) | (is_float_dtype(self.df[chl]))):
                raise Exception(5003)
            
    def target_type_check(self):

        """checking for null in target_type

        Raises:
            Exception 5005: Error: No option selected
        """
        if ((self.target_type == None) | (self.target_type == 0) | (self.target_type == "")): 
            raise Exception(5005)
        
    def granulaity_check(self):

        """checking for null in target_type

        Raises:
            Exception 5005: Error: No option selected
        """
        if ((self.granularity_selected == None) | (self.granularity_selected == 0) | (self.granularity_selected == "")): 
            raise Exception(5005)
        
    def create_eda_profiling(self):
        """Creates a Pandas profiling using the ydata_profiling library
        Args:
            df: A Pandas DataFrame.
            explorative: Whether to generate an exploratory report
        Returns: A Pandas profiling report
        """
        # Generate a Pandas profiling report
        eda_profile = ydp.ProfileReport(self.df, explorative=True, minimal=True,
                                        title="EDA Report",
                                        correlations={"pearson": {"calculate": True},
                                                        "spearman": {"calculate": True},
                                                        "auto": {"calculate": True}})

        # Return the report
        return eda_profile
    
    def UI_stats(self):
        """Getting total spend, target and number of channels based on dataset uploaded
        Args:
            df: A Pandas DataFrame.
        Returns: Dict for stats
        """
        spend_temp = 0
        for dim in self.channel_spend:
            spend_temp = spend_temp + sum(self.df[dim])

        UI_stats_json = {'total_spend' : int(round(spend_temp)),
            'total_target' : int(round(sum(self.df[self.target]))),
            'no_of_channel' : int(len(self.channel_spend))
        }

        return UI_stats_json
    
    def check_data_size(self):
        """Checks the shape of a CSV file based on the check dictionary input
        Args:
            check_dict: A dictionary containing the check criteria for each data type
            df: A Pandas DataFrame
            data_type: The specified data type
        Returns:
            A dictionary containing the data types that are out of range, and the reason why they are out of range
        """
        # Get the shape of the DataFrame.
        df_shape = self.df.shape

        # Get the check criteria for the specified data type
        criteria = self.check_dict[self.granularity_selected]

        output_dict = {}
        # Add the reason why the data type is out of range
        if df_shape[0] < criteria['min']:
            reason = "Less than minimum threshold of " + str(criteria['min']) + " " + criteria['type']
            output_dict['reason'] = reason
            output_dict['difference'] = criteria['min'] - df_shape[0]
        elif df_shape[0] > criteria['max']:
            reason = "More than maximum threshold of " + str(criteria['max']) + " " + criteria['type']
            output_dict['reason'] = reason
            output_dict['difference'] = df_shape[0] - criteria['max']

        # Check if the shape of the DataFrame is out of range for the data type
        # if ~((df_shape[0] < criteria['min']) or (df_shape[0] > criteria['max'])):
        #    output_dict = None

        # Return the output dictionary.
        return output_dict
    
    def eda_findings(self, var_list):
        """Creates a dictionary of variables with missing data, zero values, and combined missing and zero values
        Returns:
            A dictionary of variables with missing data, zero values, and combined missing and zero values
        """

        # Create a dictionary to store the results.
        self.validation_report["missing_datapoints"] = {}
        self.validation_report["zero_datapoints"] = {}
        self.validation_report["combined_datapoints"] = {}

        for variable in var_list:
            # Check for variables with p_missing > 0.1
            p_missing = var_list[variable].get("p_cells_missing", 0.0)
            if p_missing > 0.1:
                self.validation_report["missing_datapoints"][variable]=round(p_missing*100,2)

            # Check for variables with p_zeros > 0.1
            p_zeros = var_list[variable].get("p_zeros", 0.0)
            if p_zeros > 0.1:
                self.validation_report["zero_datapoints"][variable]=round(p_zeros*100,2)

            # Check for variables with p_combined > 0.15
            p_combined = p_missing + p_zeros
            # Only add the variable to the "combined_datapoints" key if it is not already present in the "missing_datapoints" or "zero_datapoints" keys
            if p_combined > 0.15 and variable not in (self.validation_report["missing_datapoints"] or self.validation_report["zero_datapoints"]):
                self.validation_report["combined_datapoints"][variable]=round(p_combined*100,2)
    
    def variance_check(self, var_list):
        """Checks if variance is available in report['variables']. If not, adds the variable name to variance_threshold
        Args:
            report: A dictionary containing the report information
        Returns: A dictionary of variable names for which variance is not available
        """
        self.validation_report["no_variance_var"] = []
        for variable in var_list:
            p_variance = var_list[variable].get("variance")
            # Check if the value of p_var[variable].get("variance") is numeric.
            if isinstance(p_variance, (int, float)):
                # If p_variance == 0, append variable to the variance_threshold["no_variance"] list.
                if p_variance == 0:
                    self.validation_report["no_variance_var"].append(variable)

    def outlier_analysis(self, var_list):
        """check outlier using z score in the list of variables
        Returns:
            dictionary: list with variables names and percentage data having outliers
        """
        df = self.df.copy()
        outlier_analysis = {}
        outlier_dic = {}
        skipped_var = []
        for col in var_list:
            if (df[col].dtypes == object) or (df[col].dtypes == str) or (df[col].dtypes == bool) or isinstance(df[col], datetime.date):
                skipped_var = skipped_var + [col]
                continue
            mean = np.mean(df[col])
            std = np.std(df[col])
            df["z_outlier_" + col] = np.where(
                abs((df[col] - mean) / std) > 3, 1, 0)
            outlier_per = df[df["z_outlier_" + col]==1].shape[0]/df.shape[0]
            if outlier_per>0:
                outlier_dic[col]=round(outlier_per*100, 2)
            df.drop(columns=["z_outlier_" + col], inplace=True)
            outlier_analysis['outliers'] = outlier_dic
            outlier_analysis['skipped_var'] = skipped_var

        return outlier_analysis
    
    def eda_report(self, profile, output_file):
        """Saves a EDA - Pandas profiling report
        Args:
            profile: A Pandas profiling report.
            output_file: The path to the output file.
        """
        output_file = output_file + "EDA_Report.html"
        # Save the report to a file.
        profile.to_file(output_file)

    def get_sample_report(self, eda_profile_report):
        """get sample report in json of variables  for UI (first spend variable selected by the user is considered)
        Returns:
            dictionary: sample report of the variable
        """
        sample_report = {}
        overview_report = eda_profile_report['table']
        variable_report = eda_profile_report['variables'][self.channel_spend[0]]
        var_name = {"variable_name": self.channel_spend[0]}
        variable_report = {**var_name, **variable_report}
        sample_report['overview'] = overview_report
        sample_report['variable'] = variable_report

        return sample_report
            
    def execute(self, file_path):
        """main function to execute all the insights page functionality
        Args: file_path: file path where EDA report has to be saved
        Returns:
            dictionary:
              validation_report: variance analysis, outlier analysis, datapoint validation
              sample_report: overview and one sample variable stats for UI
              user_inputs: explorer user inputs to be passed to predictor
              UI_stats_json: high level stats for uploaded dataset
        """
        self.validation_report = {}
        
        # format the date column
        _format = "%Y-%m-%d"
        if pd.to_datetime(self.df[self.date], format=_format, errors='coerce').notnull().all(): 
            self.df[self.date] = pd.to_datetime(self.df[self.date], format=_format)
        
        # rename the columns to standard names to be used in predict/optimize/goal seek tabs
        # self.df = self.df.rename(
        #     columns={self.target: "target", self.date: "date"}
        # )

        # Getting eda report through Pandas profiling
        eda_profile = self.create_eda_profiling()

        # Read the CSV file into a Pandas DataFrame.
        eda_profile_report = json.loads(eda_profile.to_json())

        # Getting sample report for one variable for UI
        sample_report = self.get_sample_report(eda_profile_report)

        report_var_list = eda_profile_report['variables']

        # Check data size based on granulaity
        self.validation_report['data_size_validation'] = self.check_data_size()

        # Find the variables with missing data, zero values, and combined missing and zero values
        self.eda_findings(report_var_list)

        # Check variance of the variables in the dataset
        self.variance_check(report_var_list)

        # Check which variables contain outliers
        outlier_report = self.outlier_analysis(report_var_list)
        self.validation_report['outliers'] = outlier_report['outliers']

        # Converting and saving pandas profile eda report to HTML
        self.eda_report(eda_profile, file_path)

        # For MMM model inputs
        user_inputs = {
            'date_variable' : self.date,
            'target_variable': self.target,
            'target_variable_type': self.target_type,
            'granularity': self.granularity_selected,
            'spend_variables': self.channel_spend,
            'filename':""
        }

        UI_stats_json = self.UI_stats()

        return self.validation_report, sample_report, user_inputs, UI_stats_json