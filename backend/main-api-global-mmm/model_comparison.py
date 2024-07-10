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


def compare_model_result(output_robyn, output_pymc, tolerance=1.0):
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
        return abs(value1 - value2) <= tol

    # Track the best model and its metrics
    best_model = None
    best_metrics = None

    # Compare R² or Adjusted R² within tolerance
    if within_tolerance(output_robyn['adj_rsq'], output_pymc['adj_rsq'], tolerance):
        # Compare MAPE if R² or Adjusted R² is within tolerance
        if within_tolerance(output_robyn['mape'], output_pymc['mape'], tolerance):
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
            if output_robyn['mape'] < output_pymc['mape']:
                best_model = 'robyn'
                best_metrics = output_robyn
            else:
                best_model = 'pymc'
                best_metrics = output_pymc
    else:
        # Choose based on highest R² or Adjusted R² if not within tolerance
        if output_robyn['adj_rsq'] > output_pymc['adj_rsq']:
            best_model = 'robyn'
            best_metrics = output_robyn
        else:
            best_model = 'pymc'
            best_metrics = output_pymc

    # Add the name of the best model to the metrics as the first entry
    best_metrics = {'best_model': best_model, **best_metrics}

    return best_metrics

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