import numpy as np
import pandas as pd
import re
import scipy.stats as stats
from statsmodels.stats.power import TTestIndPower
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('../data/hpo_analysis/new_allegro/wandb_export_2025-05-17T11_11_30.422-04_00.csv')

print(f"Original columns: {df.columns}")
# print("Original DataFrame head:")
# print(df.head())

# Function to parse the 'Name' column
def parse_name(name_str):
    params = {}
    match = re.search(r'hpo_(\d+)', name_str)
    if match:
        params['hpo_run_id'] = int(match.group(1))
    else:
        params['hpo_run_id'] = None

    core_hyperparameters = [
        "num_layers", "l_max", "num_scalar_features", 
        "num_tensor_features", "mlp_width"
    ]
    
    for key in core_hyperparameters:
        match = re.search(rf'{key}-([\d\.]+)', name_str)
        if match:
            val_str = match.group(1)
            try:
                params[key] = int(val_str)
            except ValueError:
                params[key] = float(val_str)
        else:
            params[key] = np.nan

    match = re.search(r'fold(\d+)', name_str)
    if match:
        params['fold'] = int(match.group(1))
    else:
        params['fold'] = np.nan

    match = re.search(r'seed(\d+)', name_str)
    if match:
        params['seed'] = int(match.group(1))
    else:
        params['seed'] = np.nan
        
    config_values = [params.get(key) for key in core_hyperparameters]
    params['hpo_config_signature'] = tuple(config_values)
        
    return pd.Series(params)

parsed_df = df['Name'].apply(parse_name)
df = pd.concat([df, parsed_df], axis=1)

print("\nDataFrame after parsing 'Name' column (first 5 rows):")
print(df.head())

core_hpo_params_for_cleaning = ["num_layers", "l_max", "num_scalar_features", "num_tensor_features", "mlp_width"]
df.dropna(subset=['hpo_config_signature'] + core_hpo_params_for_cleaning, how='any', inplace=True)

for col in core_hpo_params_for_cleaning:
    if col in df.columns:
        # Check if df[col] is a DataFrame (due to duplicate column names)
        target_data = df[col]
        if isinstance(target_data, pd.DataFrame):
            # If it's a DataFrame, we might have duplicate columns.
            # A common strategy is to use the first instance if the values are expected to be the same,
            # or average, or handle based on specific knowledge.
            # Here, we'll try to convert the first column, assuming it's the primary one.
            # If they could be different and meaningful, a more complex de-duplication might be needed earlier.
            print(f"Warning: Column '{col}' is a DataFrame, likely due to duplicates. Using the first instance for numeric conversion.")
            df[col] = pd.to_numeric(target_data.iloc[:, 0], errors='coerce')
        elif isinstance(target_data, pd.Series):
            df[col] = pd.to_numeric(target_data, errors='coerce')
        else:
            # Fallback for other unexpected types
            try:
                df[col] = pd.to_numeric(target_data, errors='coerce')
            except TypeError:
                print(f"Warning: Could not convert column '{col}' of type {type(target_data)} to numeric.")


df.dropna(subset=core_hpo_params_for_cleaning, inplace=True)

test_metric_columns = [col for col in df.columns if col.startswith('test') and pd.api.types.is_numeric_dtype(df[col])]
print(f"\nTest metric columns: {test_metric_columns}")

# Ensure 'hpo_config_signature' is a 1D Series before use.
if 'hpo_config_signature' in df.columns:
    if isinstance(df['hpo_config_signature'], pd.DataFrame):
        print("Warning: 'hpo_config_signature' column was a DataFrame. Modifying to use the first instance.")
        df['hpo_config_signature'] = df['hpo_config_signature'].iloc[:, 0]
    
    unique_configs = df['hpo_config_signature'].unique()
    print(f"\nFound {len(unique_configs)} unique HPO configurations after cleaning.")
    print("\nNumber of runs per HPO configuration (top 10):")
    print(df.groupby('hpo_config_signature').size().sort_values(ascending=False).head(10))
else:
    print("Error: 'hpo_config_signature' column not found. Cannot calculate unique configs or group.")
    unique_configs = np.array([])

# --- Helper function for pairwise statistical comparison ---
def perform_pairwise_comparison(data1, data2, alpha_corrected):
    """
    Performs Mann-Whitney U test, calculates Cohen's d, and post-hoc power
    for two independent groups.
    
    Args:
        data1 (array-like): Data for group 1.
        data2 (array-like): Data for group 2.
        alpha_corrected (float): Corrected significance level for the test.
        
    Returns:
        dict: Contains p_value, cohen_d, power, and significance status.
    """
    if len(data1) < 2 or len(data2) < 2:
        return {
            'p_value': np.nan, 'cohen_d': np.nan, 'power': np.nan,
            'significant': False, 'error': 'Insufficient data'
        }

    # Mann-Whitney U test
    try:
        u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
    except ValueError as e: # Handles cases like all values being identical
         return {
            'p_value': 1.0 if np.array_equal(data1, data2) else np.nan, # Or handle as appropriate
            'cohen_d': 0.0 if np.array_equal(data1, data2) else np.nan,
            'power': np.nan,
            'significant': False,
            'error': f'Mann-Whitney U error: {e}'
        }


    # Effect Size: Cohen's d
    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    n1, n2 = len(data1), len(data2)
    
    if n1 + n2 - 2 <= 0: # Should be caught by len(data) < 2 but as an extra safe guard
        cohen_d = np.nan
    else:
        s_pooled = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        if s_pooled == 0:
            cohen_d = np.inf if mean1 != mean2 else 0
        else:
            cohen_d = (mean1 - mean2) / s_pooled
            
    # Post-hoc Power
    power = np.nan
    if not np.isnan(cohen_d) and n1 > 0 and n2 > 0 : # Check if cohen_d is valid
        power_analysis = TTestIndPower()
        try:
            power = power_analysis.power(effect_size=abs(cohen_d), 
                                         nobs1=n1, 
                                         ratio=n2/n1,
                                         alpha=alpha_corrected)
        except Exception: # Catch any error from power calculation
            power = np.nan

    significant = p_value < alpha_corrected
    
    return {
        'p_value': p_value,
        'cohen_d': cohen_d,
        'power': power,
        'significant': significant,
        'n1': n1,
        'n2': n2,
        'error': None
    }

# --- Main Statistical Analysis Loop ---
alpha = 0.05 # Overall significance level

for metric in test_metric_columns:
    print(f"\n\n--- Analyzing Metric: {metric} ---")
    
    metric_df = df.dropna(subset=[metric]).copy() # Use .copy() to avoid SettingWithCopyWarning
    
    if metric_df.empty or metric_df['hpo_config_signature'].nunique() < 2:
        print(f"Skipping {metric} due to insufficient data or unique groups after NaN removal.")
        continue

    # Kruskal-Wallis needs groups with at least one observation,
    # but for meaningful comparison, more are needed.
    # The perform_pairwise_comparison handles groups with < 2 observations.
    grouped_data_for_kruskal = [
        group[metric].values for _, group in metric_df.groupby('hpo_config_signature') if len(group[metric].values) > 0
    ]
    
    if len(grouped_data_for_kruskal) < 2:
        print(f"Skipping Kruskal-Wallis for {metric} as fewer than 2 groups have data.")
        # Still attempt to show boxplot
        if not metric_df.empty:
            plt.figure(figsize=(12, 6))
            # Ensure hpo_config_signature_str is created for plotting if not already
            if 'hpo_config_signature_str' not in metric_df.columns:
                 metric_df['hpo_config_signature_str'] = metric_df['hpo_config_signature'].astype(str)
            
            sorted_configs_plot = metric_df.groupby('hpo_config_signature_str')[metric].median().sort_values().index
            sns.boxplot(x='hpo_config_signature_str', y=metric, data=metric_df, order=sorted_configs_plot)
            plt.title(f'Distribution of {metric} by HPO Configuration')
            plt.xticks(rotation=90, ha='right', fontsize=8)
            plt.tight_layout()
            plt.show()
        continue
        
    kruskal_stat, kruskal_p_value = stats.kruskal(*grouped_data_for_kruskal)
    print(f"\nKruskal-Wallis test for {metric}:")
    print(f"  Statistic: {kruskal_stat:.4f}, P-value: {kruskal_p_value:.4f}")

    # Visualization
    plt.figure(figsize=(15, 7))
    metric_df['hpo_config_signature_str'] = metric_df['hpo_config_signature'].astype(str)
    sorted_configs_plot = metric_df.groupby('hpo_config_signature_str')[metric].median().sort_values().index
    sns.boxplot(x='hpo_config_signature_str', y=metric, data=metric_df, order=sorted_configs_plot)
    plt.title(f'Distribution of {metric} by HPO Configuration (Sorted by Median)')
    plt.xlabel('HPO Configuration Signature')
    plt.ylabel(metric)
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.tight_layout()
    plt.show()

    if kruskal_p_value < alpha:
        print(f"  Significant difference found among HPO configurations for {metric} (p < {alpha}). Performing pairwise comparisons.")
        
        configs = list(metric_df['hpo_config_signature'].unique())
        n_configs = len(configs)
        # n_comparisons calculation should only count valid pairs for Bonferroni
        # For simplicity, we'll apply Bonferroni based on actual comparisons made.
        
        pairwise_results_list = []
        
        # Count actual comparisons to be made for Bonferroni
        num_actual_comparisons = 0
        temp_pairs_to_compare = []
        for i in range(n_configs):
            for j in range(i + 1, n_configs):
                config1_sig_temp = configs[i]
                config2_sig_temp = configs[j]
                data1_temp = metric_df[metric_df['hpo_config_signature'] == config1_sig_temp][metric].dropna().values
                data2_temp = metric_df[metric_df['hpo_config_signature'] == config2_sig_temp][metric].dropna().values
                if len(data1_temp) >=2 and len(data2_temp) >=2:
                    num_actual_comparisons +=1
                    temp_pairs_to_compare.append((config1_sig_temp, config2_sig_temp))
        
        if num_actual_comparisons == 0:
            print("  No valid pairs for pairwise comparison (e.g. all groups have < 2 data points).")
            continue
            
        corrected_alpha = alpha / num_actual_comparisons
        print(f"  Using Bonferroni corrected alpha for pairwise tests: {corrected_alpha:.5f} ({alpha}/{num_actual_comparisons})")

        for config1_sig, config2_sig in temp_pairs_to_compare:
            data1 = metric_df[metric_df['hpo_config_signature'] == config1_sig][metric].dropna().values
            data2 = metric_df[metric_df['hpo_config_signature'] == config2_sig][metric].dropna().values
            
            # Already checked len(data1) >=2 and len(data2) >=2
            comparison_stats = perform_pairwise_comparison(data1, data2, corrected_alpha)
            
            if comparison_stats['error']:
                print(f"  Skipping {config1_sig} vs {config2_sig}: {comparison_stats['error']}")
                continue

            print(f"  Comparison: {str(config1_sig)} vs {str(config2_sig)}")
            print(f"    Mann-Whitney U p-value: {comparison_stats['p_value']:.4f} "
                  f"(Significant w/ Bonferroni: {comparison_stats['significant']})")
            print(f"    Cohen's d: {comparison_stats['cohen_d']:.3f}")
            print(f"    Post-hoc Power (approx, alpha={corrected_alpha:.4f}): {comparison_stats['power']:.3f}")
            print(f"    N1: {comparison_stats['n1']}, N2: {comparison_stats['n2']}")
            
            pairwise_results_list.append({
                'metric': metric,
                'config1': str(config1_sig),
                'config2': str(config2_sig),
                'p_value': comparison_stats['p_value'],
                'cohen_d': comparison_stats['cohen_d'],
                'power': comparison_stats['power'],
                'significant_bonferroni': comparison_stats['significant'],
                'n1': comparison_stats['n1'],
                'n2': comparison_stats['n2'],
                'corrected_alpha': corrected_alpha
            })

        if pairwise_results_list:
            pairwise_df = pd.DataFrame(pairwise_results_list)
            print(f"\n  Summary of Pairwise Comparisons for {metric} (Bonferroni applied):")
            # Show all significant pairs, or pairs with large effect/high power
            significant_or_notable = pairwise_df[
                (pairwise_df['significant_bonferroni']) | 
                (pairwise_df['power'].fillna(0) > 0.8) | 
                (abs(pairwise_df['cohen_d'].fillna(0)) > 0.5)
            ]
            if not significant_or_notable.empty:
                print(significant_or_notable.to_string(index=False, float_format="%.3f"))
            else:
                print("    No pairs met the criteria for 'significant or notable' in the summary.")
    else:
        print(f"  No significant overall difference found among HPO configurations for {metric} (p >= {alpha}). Skipping pairwise comparisons.")

print("\n\nAnalysis Complete.")
print("Post-hoc power is descriptive. Effect sizes (Cohen's d) provide context to p-values.")

# --- Example: How to use perform_pairwise_comparison for two specific configurations ---
# This is a conceptual example. You'll need to pick actual config signatures from your data.

# First, ensure you have at least one metric processed and unique_configs populated
if test_metric_columns and 'hpo_config_signature' in df.columns and len(df['hpo_config_signature'].unique()) >=2 :
    example_metric = test_metric_columns[0] # Take the first test metric
    print(f"\n\n--- Example of Specific Pairwise Comparison for metric: {example_metric} ---")
    
    # Get all unique configurations for this metric after NaN removal
    example_metric_df = df.dropna(subset=[example_metric])
    available_configs_for_example = list(example_metric_df['hpo_config_signature'].unique())

    if len(available_configs_for_example) >= 2:
        config_A_sig = available_configs_for_example[0]
        config_B_sig = available_configs_for_example[1]

        print(f"Comparing Config A: {config_A_sig}")
        print(f"with Config B: {config_B_sig}")

        data_A = example_metric_df[example_metric_df['hpo_config_signature'] == config_A_sig][example_metric].dropna().values
        data_B = example_metric_df[example_metric_df['hpo_config_signature'] == config_B_sig][example_metric].dropna().values
        
        # For a standalone comparison, you might use the original alpha or a corrected one
        # depending on context (e.g., if this is one of many planned comparisons).
        # Here, using original alpha for a single, ad-hoc comparison example.
        standalone_alpha = 0.05 
        
        specific_comparison_results = perform_pairwise_comparison(data_A, data_B, alpha_corrected=standalone_alpha)

        if specific_comparison_results['error']:
             print(f"Could not compare: {specific_comparison_results['error']}")
        else:
            print(f"  Metric: {example_metric}")
            print(f"  Mann-Whitney U p-value: {specific_comparison_results['p_value']:.4f}")
            print(f"  Cohen's d: {specific_comparison_results['cohen_d']:.3f}")
            print(f"  Power (alpha={standalone_alpha}): {specific_comparison_results['power']:.3f}")
            print(f"  Significant at alpha={standalone_alpha}: {specific_comparison_results['significant']}")
            print(f"  N_A: {specific_comparison_results['n1']}, N_B: {specific_comparison_results['n2']}")
    else:
        print("Not enough unique configurations with data for the example metric to perform a specific pairwise comparison.")
else:
    print("\nSkipping example of specific pairwise comparison due to lack of processed metrics or configurations.")
