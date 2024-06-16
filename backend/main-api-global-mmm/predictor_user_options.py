import pandas as pd
from pandas.api.types import is_categorical_dtype
from pandas.api.types import is_numeric_dtype, is_float_dtype
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import datetime
import warnings
warnings.filterwarnings('ignore')

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
    for var in user_params['spend_variables']:
        media_name_check = [media_name for media_name in data_var if var.split('_')[0].lower() in media_name.lower()]
        media_var_imp = [media_var for media_var in media_name_check if 'impression'.lower() in media_var.lower()]
        media_var_click = [media_var for media_var in media_name_check if 'click'.lower() in media_var.lower()]

    context_vars = list(df.columns)
    context_vars = list(set(context_vars) - set([user_params['date_variable']] + 
                                           [user_params['target_variable']] + 
                                           user_params['spend_variables'] +
                                           media_var_imp +
                                           media_var_click))
    for col in context_vars:
        if (df[col].dtypes == object) or (df[col].dtypes == str) or (df[col].dtypes == bool) or isinstance(df[col], datetime.date) or pd.api.types.is_datetime64_dtype(df[col]) or (col == user_params['target_variable']):
            context_vars = list(set(context_vars) - set([col]))
    
    adstock_list = ['Exponential Fixed Decay', 'Flexible Decay with No Lag', 'Flexible Decay with Lag']

    user_options = {'country' : country_list,
                    'control_variables' : context_vars,
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
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X, y)
    feat_importances = round(pd.Series(rf.feature_importances_, index=X.columns)*100, 2)
    feat_importances = feat_importances.sort_values(ascending=False)
    feat_importances_df = pd.DataFrame(feat_importances).reset_index(drop=False)
    feat_importances_df = feat_importances_df.rename({'index':'Variable', 0:'Importance'}, axis=1)

    return feat_importances_df