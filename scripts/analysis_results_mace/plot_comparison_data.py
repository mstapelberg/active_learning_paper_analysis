import pandas as pd
import matplotlib.pyplot as plt
import os
import ast
import numpy as np

def parse_cell_param_diffs(cell_param_str):
    """Parses the cell_param_diffs string into a dictionary of floats."""
    try:
        # Clean the string: remove np.float64, extraneous quotes, etc.
        cleaned_str = cell_param_str.replace('np.float64(', '').replace(')', '')
        # The string looks like a dictionary, so ast.literal_eval should work
        return ast.literal_eval(cleaned_str)
    except Exception as e:
        print(f"Error parsing cell_param_diffs string: {cell_param_str}")
        print(f"Error: {e}")
        # Return a dictionary with NaNs if parsing fails, to maintain structure
        return {'da': np.nan, 'db': np.nan, 'dc': np.nan, 'dalpha': np.nan, 'dbeta': np.nan, 'dgamma': np.nan}

def plot_data(data_type, df):
    """
    Generates and saves bar plots for the given DataFrame.

    Args:
        data_type (str): The type of data (e.g., 'perfect', 'neb', 'vacancy').
        df (pd.DataFrame): The DataFrame containing the data to plot.
    """
    if df.empty:
        print(f"No data to plot for {data_type}.")
        return

    output_dir = f"plots/{data_type}"
    os.makedirs(output_dir, exist_ok=True)

    # Columns to plot directly
    direct_plot_cols = ['energy_diff', 'cell_rmse', 'mean_atomic_displacement', 'max_atomic_displacement']
    
    # --- Plotting direct columns ---
    for col in direct_plot_cols:
        if col not in df.columns:
            print(f"Column {col} not found in DataFrame for {data_type}. Skipping.")
            continue
        plt.figure(figsize=(12, 7))
        try:
            # Ensure the column is numeric, converting if necessary
            numeric_col_data = pd.to_numeric(df[col], errors='coerce')
            if numeric_col_data.isnull().all():
                print(f"All values in column {col} are NaN for {data_type}. Skipping plot.")
                plt.close()
                continue

            plt.bar(df['composition'], numeric_col_data)
            plt.xlabel("Composition")
            plt.ylabel(col.replace('_', ' ').title())
            plt.title(f"{col.replace('_', ' ').title()} for {data_type.title()} Structures")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{col}_comparison.png"))
            plt.close()
            print(f"Saved plot: {output_dir}/{col}_comparison.png")
        except Exception as e:
            print(f"Could not generate plot for {col} in {data_type}: {e}")
            plt.close()

    # --- Plotting cell_param_diffs ---
    if 'cell_param_diffs' in df.columns:
        # Parse the 'cell_param_diffs' column
        parsed_cell_params = df['cell_param_diffs'].apply(parse_cell_param_diffs)
        cell_param_df = pd.DataFrame(parsed_cell_params.tolist(), index=df.index)

        # Ensure 'composition' column is available for x-axis
        if 'composition' in df.columns:
            cell_param_df['composition'] = df['composition']
        else:
            print(f"Composition column not found for {data_type} during cell_param_diffs plotting. Using index.")
            cell_param_df['composition'] = cell_param_df.index


        for param_col in cell_param_df.columns:
            if param_col == 'composition': # Skip the composition column itself
                continue
            plt.figure(figsize=(12, 7))
            try:
                 # Ensure the column is numeric, converting if necessary
                numeric_param_data = pd.to_numeric(cell_param_df[param_col], errors='coerce')
                if numeric_param_data.isnull().all():
                    print(f"All values in cell param diff {param_col} are NaN for {data_type}. Skipping plot.")
                    plt.close()
                    continue

                plt.bar(cell_param_df['composition'], numeric_param_data)
                plt.xlabel("Composition")
                plt.ylabel(f"Cell Parameter Difference ({param_col})")
                plt.title(f"Cell Parameter {param_col.upper()} Differences for {data_type.title()} Structures")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"cell_param_{param_col}_comparison.png"))
                plt.close()
                print(f"Saved plot: {output_dir}/cell_param_{param_col}_comparison.png")
            except Exception as e:
                print(f"Could not generate plot for cell_param_diffs ({param_col}) in {data_type}: {e}")
                plt.close()
    else:
        print(f"Column 'cell_param_diffs' not found in DataFrame for {data_type}. Skipping these plots.")


def main():
    #base_results_dir = "analysis_results_mace"
    base_results_dir = "."
    data_folders = ["perfect", "neb", "vacancy"]
    
    # Columns that might contain np.float64() syntax and need conversion
    # For 'cell_param_diffs', a custom parser is used.
    # 'atomic_displacements_all' is a list of floats, usually fine with pd.read_csv if formatted correctly.
    # However, if it's like "[0.00835...]" string, it might need ast.literal_eval too.
    # For now, we focus on the requested plots.

    all_data = {}

    for folder in data_folders:
        file_path = os.path.join(base_results_dir, folder, f"summary_{folder}_comparisons.csv")
        if os.path.exists(file_path):
            print(f"Reading data from: {file_path}")
            try:
                df = pd.read_csv(file_path)
                
                # Clean column names (remove leading/trailing spaces)
                df.columns = [col.strip() for col in df.columns]
                
                # Basic check for expected columns
                print(f"Columns in {file_path}: {df.columns.tolist()}")
                if 'composition' not in df.columns:
                    print(f"Critical: 'composition' column missing in {file_path}. Check CSV format.")
                    continue # Skip this file if composition is missing

                all_data[folder] = df
                plot_data(folder, df)
            except pd.errors.EmptyDataError:
                print(f"Warning: {file_path} is empty. Skipping.")
            except Exception as e:
                print(f"Error reading or processing {file_path}: {e}")
        else:
            print(f"File not found: {file_path}. Skipping.")

    # --- Example: Combined plotting (if needed) ---
    # This part is conceptual. You'd need to define how you want to combine
    # and compare across 'perfect', 'neb', 'vacancy' for each material.
    # For instance, comparing 'energy_diff' for 'MaterialX' across all three types.

    # print("\n--- Placeholder for Combined Plots ---")
    # if all_data:
    #     # Example: Iterate through unique compositions across all loaded data
    #     all_compositions = set()
    #     for df_type in all_data.values():
    #         if 'composition' in df_type.columns:
    #             all_compositions.update(df_type['composition'].unique())
        
    #     for comp in sorted(list(all_compositions)):
    #         plt.figure(figsize=(15, 8))
    #         plot_idx = 1
            
    #         # Metrics to compare
    #         metrics_to_compare = ['energy_diff', 'cell_rmse', 'mean_atomic_displacement'] # Add more as needed

    #         for metric in metrics_to_compare:
    #             ax = plt.subplot(1, len(metrics_to_compare), plot_idx)
    #             data_points = []
    #             labels = []
                
    #             for folder_type, df in all_data.items():
    #                 if 'composition' in df.columns and metric in df.columns:
    #                     material_data = df[df['composition'] == comp]
    #                     if not material_data.empty:
    #                         value = material_data[metric].iloc[0] # Assuming one entry per composition
    #                         # Convert to numeric, coercing errors
    #                         numeric_value = pd.to_numeric(value, errors='coerce')
    #                         if pd.notna(numeric_value):
    #                             data_points.append(numeric_value)
    #                             labels.append(folder_type.title())
    #                         else:
    #                             print(f"Warning: Could not convert {metric} value '{value}' to numeric for {comp} in {folder_type}")
                
    #             if data_points:
    #                 ax.bar(labels, data_points)
    #                 ax.set_title(f"{metric.replace('_', ' ').title()}\nfor {comp}")
    #                 ax.set_ylabel(metric.replace('_', ' ').title())
    #                 ax.tick_params(axis='x', rotation=45)
    #             else:
    #                 ax.text(0.5, 0.5, 'No data', ha='center', va='center')
    #                 ax.set_title(f"{metric.replace('_', ' ').title()}\nfor {comp}")

    #             plot_idx += 1
            
    #         if plot_idx > 1: # Only save if there was something to plot
    #             plt.suptitle(f"Comparison for Material: {comp}", fontsize=16)
    #             plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    #             combined_plot_dir = "plots/combined"
    #             os.makedirs(combined_plot_dir, exist_ok=True)
    #             plt.savefig(os.path.join(combined_plot_dir, f"comparison_{comp.replace('/', '_')}.png"))
    #             print(f"Saved combined plot: {combined_plot_dir}/comparison_{comp.replace('/', '_')}.png")
    #         plt.close()

    print("\nScript finished.")

if __name__ == "__main__":
    main() 