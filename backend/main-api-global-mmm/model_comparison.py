import json
from flask import Flask, request, jsonify

def convert_json(frontend_data):
  input_data = {}
  input_data["include_holiday"] = frontend_data["include_holiday"]
  input_data["prophet_country"] = frontend_data["prophet_country"]
  input_data["date_variable"] = frontend_data["date_variable"]
  input_data["target_variable"] = frontend_data["target_variable"]
  input_data["organic_variables"] = frontend_data["organic_variables"]
  input_data["adstock"] = frontend_data["adstock"]
  input_data["filename"] = frontend_data["filename"]

  # Handle paid_media_spends transformation (assuming empty lists are desired)
  input_data["media_variables"] = []
  input_data["spend_variables"] = []

  # Control variables can be directly assigned (assuming context_vars are empty)
  input_data["control_variables"] = frontend_data.get("context_vars", [])

  return input_data

def compare_model_result(output_robyn, output_pymc, tolerance=1.0/100):
    """Compares Robyn and PyMC model results and returns the better model as a dictionary.

    Args:
        output_robyn (dict): Output dictionary from Robyn model.
        output_pymc (dict): Output dictionary from PyMC model.
        tolerance (float): The tolerance range for metric comparisons.

    Returns:
        dict: Dictionary containing metrics of the better model.
    """

    # Helper function to determine if two values are within the tolerance range
    def within_tolerance(value1, value2, tol):
        # print((value1, value2), abs(value1 - value2), abs(value1 - value2) <= tol)
        return abs(value1 - value2) <= tol

    # Track the best model and its metrics
    best_model = None
    best_metrics = None

    # Helper function to check if a value is infinite
    def is_infinite(value):
        if isinstance(value, str) and any(keyword in value.lower() for keyword in ['infinity', '-infinity', 'infinite', 'inf']):
            return True
        if isinstance(value, (int, float)) and (value == float('inf') or value == float('-inf')):
            return True
        return False
    
    output_pymc['rssd'] = output_pymc['rssd']/100

    # Check for infinity in adjusted R²
    robyn_infinite_adjr2 = is_infinite(output_robyn['adjusted_r2_train'])
    pymc_infinite_adjr2 = is_infinite(output_pymc['adjusted_r2_train'])

    # Check for infinity in mape
    robyn_infinite_mape = is_infinite(output_robyn['mape_train'])
    pymc_infinite_mape = is_infinite(output_pymc['mape_train'])

    # Checking for scenarios:
    #    when both adj r2 and mape are inf
    #    when adj r2 is inf and mape is within tolerance
    #    when only adj r2 is inf

    # if adj r2 robyn is inf and pymc is not
    if robyn_infinite_adjr2 and not pymc_infinite_adjr2:
        return 'pymc'
    # if adj r2 pymc is inf and robyn is not
    elif pymc_infinite_adjr2 and not robyn_infinite_adjr2:
        return 'robyn'
    # if adj r2 for robyn and pymc is inf
    elif robyn_infinite_adjr2 and pymc_infinite_adjr2:
        # # if mape robyn is inf and pymc is not
        if robyn_infinite_mape and not pymc_infinite_mape:
            return 'pymc'
        # if mape pymc is inf and robyn is not
        elif pymc_infinite_mape and not robyn_infinite_mape:
            return 'robyn'
        # if mape for robyn and pymc is inf or # MAPE is within tolerance, compare RSSD
        elif (robyn_infinite_mape and pymc_infinite_mape) or within_tolerance(output_robyn['mape_train'], output_pymc['mape_train'], tolerance):
            if output_robyn['rssd'] < output_pymc['rssd']:
                return 'robyn'
            else:
                return 'pymc'
        else:
            # MAPE is not within tolerance, choose the lower MAPE
            if output_robyn['mape_train'] < output_pymc['mape_train']:
                return 'robyn'
            else:
                return 'pymc'
            

    # Checking for scenario:
    #    when adj r2 is within tolerance and mape is is inf

    # Compare MAPE if R² or Adjusted R² is within tolerance
    if within_tolerance(output_robyn['adjusted_r2_train'], output_pymc['adjusted_r2_train'], tolerance):
        if robyn_infinite_mape and not pymc_infinite_mape:
            return 'pymc'
        elif pymc_infinite_mape and not robyn_infinite_mape:
            return 'robyn'
        elif robyn_infinite_mape and pymc_infinite_mape:
            # Both mape are infinite, compare RSSD
            if output_robyn['rssd'] < output_pymc['rssd']:
                return 'robyn'
            else:
                return 'pymc'
            

    # Checking for normal scenarios and tolerance, when neither adj r2 or mape is inf

    # Compare R² or Adjusted R² within tolerance
    if within_tolerance(output_robyn['adjusted_r2_train'], output_pymc['adjusted_r2_train'], tolerance):
        # Compare MAPE if R² or Adjusted R² is within tolerance
        if within_tolerance(output_robyn['mape_train'], output_pymc['mape_train'], tolerance):
            # Compare RSSD if MAPE is also within tolerance
            if within_tolerance(output_robyn['rssd'], output_pymc['rssd'], tolerance):
                # All metrics are within tolerance, choose based on lowest RSSD
                if output_robyn['rssd'] < output_pymc['rssd']:
                    best_model = 'robyn'
                    best_metrics = output_robyn
                else:
                    best_model = 'pymc'
                    best_metrics = output_pymc
            else:
                # Choose based on lowest RSSD if MAPE is tied
                if output_robyn['rssd'] < output_pymc['rssd']:
                    best_model = 'robyn'
                    best_metrics = output_robyn
                else:
                    best_model = 'pymc'
                    best_metrics = output_pymc
        else:
            # Choose based on lowest MAPE if R² or Adjusted R² is tied
            if output_robyn['mape_train'] < output_pymc['mape_train']:
                best_model = 'robyn'
                best_metrics = output_robyn
            else:
                best_model = 'pymc'
                best_metrics = output_pymc
    else:
        # Choose based on highest R² or Adjusted R² if not within tolerance
        if output_robyn['adjusted_r2_train'] > output_pymc['adjusted_r2_train']:
            best_model = 'robyn'
            best_metrics = output_robyn
        else:
            best_model = 'pymc'
            best_metrics = output_pymc

    # Add the name of the best model to the metrics as the first entry
    best_metrics = {'best_model': best_model, **best_metrics}

    return best_model

def compare_models(robyn_data, pymc_data):
    # Get data from the request body
    # data = request.get_json()
    # if not data or 'output_robyn' not in data or 'output_pymc' not in data:
    #   return jsonify({'error': 'Missing required data'}), 400
    print("robyn",robyn_data)
    print("pymc",pymc_data)
    output_robyn = robyn_data["metrics"]
    output_pymc = pymc_data["metrics"]

    # Compare models and get best metrics
    best_model = compare_model_result(output_robyn, output_pymc)

    # Return JSON response with best model metrics
    return best_model