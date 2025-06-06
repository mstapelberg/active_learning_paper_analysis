#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
from skopt import Optimizer
from skopt.space import Integer, Categorical
# from skopt.utils import use_named_args # Not strictly needed for this script structure

# --- Configuration ---
CSV_PATH = "model_summary_analysis.csv"
HPO_ID_COL = "hpo_id" # Column containing the hyperparameter string
METRIC_ERROR_COL = "mean_force_rmse"
METRIC_THROUGHPUT_COL = "mean_katom_steps_per_s_log"
METRIC_TIMESTEPS_COL = "mean_timesteps_per_s_log"
N_SUGGESTIONS = 10

# Define the hyperparameter search space
# Order matters here and must match parse_hpo_id and later reconstruction
param_space = [
    Integer(1, 2, name='num_layers'),
    Integer(1, 3, name='l_max'),
    Categorical([64, 128, 256], name='num_scalar_features'),
    Categorical([32, 64, 128], name='num_tensor_features'),
    Categorical([256, 512, 1024], name='mlp_width')
]
param_names = [s.name for s in param_space]

# Weights for the objective function (score to maximize)
# score = (w_throughput * throughput) + (w_timesteps * timesteps) - (w_error * error)
# Optimizer will minimize -score
WEIGHT_ERROR = 2.0
WEIGHT_THROUGHPUT = 1.0
WEIGHT_TIMESTEPS = 1.0

# Regular expression to parse HPO ID strings
# Example: hpo_099_num_layers-2_l_max-2_num_scalar_features-128_num_tensor_features-16_mlp_width-128
hpo_pattern = re.compile(
    r"num_layers-(\d+)_l_max-(\d+)_num_scalar_features-(\d+)_num_tensor_features-(\d+)_mlp_width-(\d+)"
)

def parse_hpo_id(hpo_id_str):
    """Parses hyperparameter values from an HPO ID string."""
    match = hpo_pattern.search(hpo_id_str)
    if match:
        try:
            return [
                int(match.group(1)), # num_layers
                int(match.group(2)), # l_max
                int(match.group(3)), # num_scalar_features
                int(match.group(4)), # num_tensor_features
                int(match.group(5))  # mlp_width
            ]
        except ValueError:
            print(f"Warning: Could not parse integer from HPO ID: {hpo_id_str}")
            return None
    else:
        # Try a simpler parse if the main one fails, for ids like "hpo_001" if they don't have params
        if hpo_id_str.startswith("hpo_") and "_" not in hpo_id_str[4:]: #e.g. hpo_001
             pass # This is just an ID, no params to parse from it in this format
        else:
            print(f"Warning: HPO ID string does not match expected pattern: {hpo_id_str}")
        return None

def calculate_objective_score(error, throughput, timesteps):
    """Calculates the objective score. Optimizer minimizes -score."""
    score_to_maximize = (
        (WEIGHT_THROUGHPUT * throughput) + 
        (WEIGHT_TIMESTEPS * timesteps) - 
        (WEIGHT_ERROR * error)
    )
    return -score_to_maximize # Optimizer minimizes

def load_and_process_data(csv_path):
    """Loads data, parses HPs, and calculates objective scores."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return [], []

    required_cols = [HPO_ID_COL, METRIC_ERROR_COL, METRIC_THROUGHPUT_COL, METRIC_TIMESTEPS_COL]
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in CSV.")
            return [], []

    X_observed = []
    y_observed = []

    for _, row in df.iterrows():
        hpo_id_str = str(row[HPO_ID_COL])
        params = parse_hpo_id(hpo_id_str)

        if params:
            # Ensure all param values are within the defined categorical space if they are to be used
            # This is more for validation if we expect all historical data to conform
            # For skopt.Optimizer.tell, it's generally fine if historical points are outside
            # the *new* search space, but they must be valid values for their types.
            
            # Example check (can be expanded or made more robust):
            # if not (params[2] in param_space[2].categories and \\
            #         params[3] in param_space[3].categories and \\
            #         params[4] in param_space[4].categories):
            #     print(f"Warning: Parameters for {hpo_id_str} have values outside defined categorical space. Skipping.")
            #     continue


            try:
                error = float(row[METRIC_ERROR_COL])
                throughput = float(row[METRIC_THROUGHPUT_COL])
                timesteps = float(row[METRIC_TIMESTEPS_COL])
            except ValueError:
                print(f"Warning: Could not convert metrics to float for HPO ID: {hpo_id_str}. Skipping.")
                continue
            
            if pd.isna(error) or pd.isna(throughput) or pd.isna(timesteps):
                print(f"Warning: NaN metric value found for HPO ID: {hpo_id_str}. Skipping.")
                continue

            X_observed.append(params)
            y_observed.append(calculate_objective_score(error, throughput, timesteps))

    if not X_observed:
        print("No valid historical data points with parsable hyperparameters found to inform the optimizer.")
        print("Suggestions will be based on the initial random search strategy of the optimizer.")
        print("Proceeding without observed data.")

    return X_observed, y_observed

def main():
    print("Attempting to generate new hyperparameter suggestions using Bayesian Optimization...")
    print("Please ensure scikit-optimize is installed (pip install scikit-optimize)")
    print("\\n--- Hyperparameter Search Space ---")
    for p in param_space:
        print(p)
    print(f"\\n--- Objective Weights ---")
    print(f"  Error Weight (to minimize): {WEIGHT_ERROR}")
    print(f"  Throughput Weight (to maximize): {WEIGHT_THROUGHPUT}")
    print(f"  Timesteps/s Weight (to maximize): {WEIGHT_TIMESTEPS}")

    X_observed, y_observed = load_and_process_data(CSV_PATH)

    optimizer = Optimizer(
        dimensions=param_space,
        random_state=42,  # For reproducibility
        # acq_func="gp_hedge", # Default, good general choice
        # base_estimator="GP"   # Default is Gaussian Process Regressor
    )

    if X_observed and y_observed:
        print(f"\\nInforming optimizer with {len(X_observed)} existing data point(s).")
        try:
            optimizer.tell(X_observed, y_observed)
        except Exception as e:
            print(f"Error telling optimizer about observed data: {e}")
            print("This might be due to a mismatch between parsed parameters and the defined space,")
            print("or issues with the observed values (e.g., NaNs not caught, non-finite values).")
            print("Proceeding without observed data.")


    print(f"\\nAsking optimizer for {N_SUGGESTIONS} new hyperparameter suggestions...")
    suggested_hyperparameters_list = optimizer.ask(n_points=N_SUGGESTIONS)

    print(f"\\n--- {N_SUGGESTIONS} Suggested Hyperparameter Sets ---")
    if not suggested_hyperparameters_list:
        print("Optimizer did not return any suggestions.")
    for i, params_values in enumerate(suggested_hyperparameters_list):
        suggestion = dict(zip(param_names, params_values))
        print(f"Set {i+1}: {suggestion}")

if __name__ == "__main__":
    main() 