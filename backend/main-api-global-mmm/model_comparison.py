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



def compare_model_result(output_robyn, output_pymc):
  """Compares Robyn and PyMC model results and returns the better model as a dictionary.

  Args:
      output_robyn (dict): Output dictionary from Robyn model.
      output_pymc (dict): Output dictionary from PyMC model.

  Returns:
      dict: Dictionary containing metrics of the better model.
  """

  # Thresholds for rssd and adjusted R-squared difference
  rssd_threshold = 0.30
  adj_rsq_diff_threshold = 0.15

  # Track the best model and its metrics
  best_model = None
  best_metrics = None

  # Compare rssd
  if output_robyn['rssd'] > rssd_threshold and output_pymc['rssd'] > rssd_threshold:
    best_model = 'robyn' if output_robyn['rssd'] < output_pymc['rssd'] else 'pymc'
    best_metrics = output_robyn if best_model == 'robyn' else output_pymc
  elif output_robyn['rssd'] <= rssd_threshold and output_pymc['rssd'] > rssd_threshold:
    best_model = 'robyn'
    best_metrics = output_robyn
  elif output_robyn['rssd'] > rssd_threshold and output_pymc['rssd'] <= rssd_threshold:
    best_model = 'pymc'
    best_metrics = output_pymc

  # Compare adjusted R-squared difference if no winner yet
  if not best_model:
    if (output_robyn['diff_adj_rsq_train_test'] > adj_rsq_diff_threshold and
        output_pymc['diff_adj_rsq_train_test'] > adj_rsq_diff_threshold):
      best_model = 'robyn' if output_robyn['diff_adj_rsq_train_test'] < output_pymc['diff_adj_rsq_train_test'] else 'pymc'
      best_metrics = output_robyn if best_model == 'robyn' else output_pymc
    elif (output_robyn['diff_adj_rsq_train_test'] < adj_rsq_diff_threshold and
          output_pymc['diff_adj_rsq_train_test'] > adj_rsq_diff_threshold):
      best_model = 'robyn'
      best_metrics = output_robyn
    elif (output_robyn['diff_adj_rsq_train_test'] > adj_rsq_diff_threshold and
          output_pymc['diff_adj_rsq_train_test'] < adj_rsq_diff_threshold):
      best_model = 'pymc'
      best_metrics = output_pymc

  # Compare adjusted R-squared train if no winner yet (optional)
    if not best_model:
      if output_robyn['train-adrsq'] < output_pymc['train-adrsq']:
        best_model = 'robyn'
        best_metrics = output_robyn
      else:
        best_model = 'pymc'
        best_metrics = output_pymc

   # Add the name of the best model to the metrics as the first entry
    if best_model:
        best_metrics = {'best_model': best_model}
    
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