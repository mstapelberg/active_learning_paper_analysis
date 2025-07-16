import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler

def parse_loss_functions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features from the 'loss_functions' string column.
    Extracts coefficients for major components and flags for loss types.
    """
    df = df.copy()
    if 'loss_functions' not in df.columns:
        return df

    # --- Extract Coefficients ---
    # Use regex to find the component and its coefficient
    df['loss_coeff_energy'] = df['loss_functions'].str.extract(r'per_atom_energy_.*?\(coeff=([\d.]+)\)').astype(float)
    df['loss_coeff_forces'] = df['loss_functions'].str.extract(r'forces_.*?\(coeff=([\d.]+)\)').astype(float)
    df['loss_coeff_stress'] = df['loss_functions'].str.extract(r'stress_.*?\(coeff=([\d.]+)\)').astype(float)

    # --- Extract Loss Types (Binary Flags) ---
    df['loss_has_huber'] = df['loss_functions'].str.contains('huber', case=False, na=False).astype(int)
    df['loss_has_mse'] = df['loss_functions'].str.contains('mse', case=False, na=False).astype(int)
    df['loss_has_mae'] = df['loss_functions'].str.contains('mae', case=False, na=False).astype(int)
    df['loss_has_stratified'] = df['loss_functions'].str.contains('stratified', case=False, na=False).astype(int)
    df['loss_has_focal'] = df['loss_functions'].str.contains('focal', case=False, na=False).astype(int)

    return df

def create_llm_summary(row: pd.Series) -> str:
    """Creates a natural language summary for a single model run."""
    summary_parts = []
    
    # Performance
    force_rmse = row.get('val0_epoch/forces_rmse_metrics', float('nan'))
    stress_rmse = row.get('val0_epoch/stress_rmse_metrics', float('nan'))
    summary_parts.append(f"Achieved force RMSE of {force_rmse:.4f} and stress RMSE of {stress_rmse:.4f}.")
    if row.get('is_pareto_optimal'):
        summary_parts.append("This is a Pareto-optimal model.")

    # Key Hyperparameters
    r_max = row.get('r_max', 'N/A')
    num_layers = row.get('num_layers', 'N/A')
    poly_p = row.get('poly_p', 'N/A')
    summary_parts.append(f"Trained with r_max={r_max}, {num_layers} layers, and polynomial degree p={poly_p}.")

    # Loss Function
    loss_summary = []
    if row.get('loss_has_huber'): loss_summary.append('Huber')
    if row.get('loss_has_mse'): loss_summary.append('MSE')
    if row.get('loss_has_stratified'): loss_summary.append('stratified')
    if loss_summary:
        summary_parts.append(f"Used a {'/'.join(loss_summary)} loss.")

    return " ".join(summary_parts)

def main():
    """
    Loads the merged data, processes it for LLM analysis, and saves the result.
    """
    print("Loading merged_data.csv...")
    try:
        data = pd.read_csv('merged_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: 'merged_data.csv' not found.")
        print("Please ensure you have run the main notebook to generate this file.")
        return

    # 1. Select Best Epoch per Run
    print("Selecting best epoch for each run...")
    performance_col = 'val0_epoch/forces_rmse_metrics'
    if performance_col not in data.columns:
        print(f"Error: Performance column '{performance_col}' not found. Cannot determine best epoch.")
        return
        
    # Drop rows where performance metric is NaN before finding the best
    data.dropna(subset=[performance_col], inplace=True)
    best_epochs = data.loc[data.groupby('run_uid')[performance_col].idxmin()]
    print(f"Filtered from {len(data)} rows to {len(best_epochs)} best-performing epochs.")

    llm_df = best_epochs.copy()

    # 2. Engineer Performance Features
    print("Engineering performance features...")
    force_rmse = llm_df['val0_epoch/forces_rmse_metrics']
    stress_rmse = llm_df['val0_epoch/stress_rmse_metrics']

    # Identify Pareto frontier
    pareto_mask = pd.Series(True, index=llm_df.index)
    for i in llm_df.index:
        is_dominated = ((force_rmse < force_rmse[i]) & (stress_rmse < stress_rmse[i])).any()
        if is_dominated:
            pareto_mask[i] = False
    llm_df['is_pareto_optimal'] = pareto_mask.astype(int)

    # Normalized performance score (higher is better)
    scaler = MinMaxScaler()
    normalized_force = 1 - scaler.fit_transform(llm_df[[performance_col]])
    normalized_stress = 1 - scaler.fit_transform(llm_df[['val0_epoch/stress_rmse_metrics']])
    llm_df['performance_score'] = (normalized_force + normalized_stress) / 2

    # 3. Deconstruct Complex Features
    print("Engineering features from loss functions...")
    llm_df = parse_loss_functions(llm_df)
    
    # 4. Encode Categorical Data
    print("One-hot encoding categorical hyperparameters...")
    categorical_cols = [col for col in ['model_dtype', 'scalar_embed_mlp_nonlinearity'] if col in llm_df.columns]
    if categorical_cols:
        llm_df = pd.get_dummies(llm_df, columns=categorical_cols, prefix=categorical_cols, dummy_na=True)

    # 5. Create LLM Summary Column
    print("Creating natural language summaries...")
    llm_df['llm_summary'] = llm_df.apply(create_llm_summary, axis=1)

    # 6. Final Cleanup
    print("Cleaning up final dataframe...")
    # Select relevant columns for the final dataset
    hyperparam_cols = [
        'r_max', 'num_layers', 'l_max', 'poly_p', 'num_bessels', 'mlp_depth', 'mlp_width',
        'seed', 'avg_num_neighbors'
    ]
    performance_features = ['performance_score', 'is_pareto_optimal', 'val0_epoch/forces_rmse_metrics', 'val0_epoch/stress_rmse_metrics']
    loss_features = [col for col in llm_df.columns if col.startswith('loss_')]
    other_info = ['run_uid', 'dataset_hash', 'llm_summary']
    
    # Include one-hot encoded columns
    encoded_cols = [col for col in llm_df.columns if any(cat_col in col for cat_col in categorical_cols)]
    
    final_cols = other_info + performance_features + hyperparam_cols + loss_features + encoded_cols
    
    # Ensure all selected columns exist in the dataframe
    final_cols_exist = [col for col in final_cols if col in llm_df.columns]
    llm_df_final = llm_df[final_cols_exist]

    # Save to CSV
    output_path = 'llm_analysis_data.csv'
    llm_df_final.to_csv(output_path, index=False)
    print(f"\n✅ Successfully created LLM-ready dataset at: {output_path}")
    print(f"Final dataset has {llm_df_final.shape[0]} rows and {llm_df_final.shape[1]} columns.")

if __name__ == "__main__":
    main() 