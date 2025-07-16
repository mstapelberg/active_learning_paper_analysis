import pandas as pd
from pathlib import Path
import sys

def normalize_run_uid(run_uid: str) -> str:
    """
    Normalizes a run_uid string to a consistent format.
    - Splits 'project/run'
    - Converts project to lowercase
    - Replaces underscores with hyphens in project
    - Rejoins and returns
    """
    if '/' not in run_uid:
        return run_uid
    
    project, run_name = run_uid.split('/', 1)
    normalized_project = project.lower().replace('_', '-')
    return f"{normalized_project}/{run_name}"

def main():
    """
    Enriches the LLM analysis data with new dataset hashes and metrics.
    """
    llm_data_path = 'llm_analysis_data.csv'
    enriched_data_path = 'llm_analysis_data_enriched.csv'
    info_dir = Path('dataset_hashed_data')

    # 1. Load the new dataset information files
    print("Loading new dataset information files...")
    info_files = list(info_dir.glob('*_info.csv'))
    if not info_files:
        print(f"Error: No info CSV files found in {info_dir}")
        sys.exit(1)

    info_dfs = [pd.read_csv(f) for f in info_files]
    info_df = pd.concat(info_dfs, ignore_index=True)
    info_df.drop_duplicates(subset=['run_uid'], inplace=True)
    print(f"Combined {len(info_df)} unique run entries from info files.")

    # 2. Load the main LLM analysis data
    print(f"Loading {llm_data_path}...")
    try:
        llm_df = pd.read_csv(llm_data_path)
    except FileNotFoundError:
        print(f"Error: '{llm_data_path}' not found.")
        print("Please ensure you have run 'prepare_for_llm.py' to generate this file.")
        return

    # 3. Normalize run_uid in both DataFrames for a consistent join key
    print("Normalizing run_uid for consistent merging...")
    llm_df['run_uid'] = llm_df['run_uid'].apply(normalize_run_uid)
    info_df['run_uid'] = info_df['run_uid'].apply(normalize_run_uid)

    # 4. Merge DataFrames
    print("Merging dataframes...")
    enriched_df = pd.merge(llm_df, info_df, on='run_uid', how='left', suffixes=('', '_new'))

    # 5. Update dataset_hash and add new metrics
    print("Updating dataset_hash column...")
    # Update 'dataset_hash' with the new hash where it exists, otherwise keep the old one
    if 'updated_dataset_hash' in enriched_df.columns:
        enriched_df['dataset_hash'] = enriched_df['updated_dataset_hash'].fillna(enriched_df['dataset_hash'])
        enriched_df.drop(columns=['updated_dataset_hash'], inplace=True)


    # Add new metric columns, keeping original values if new ones are not available
    for col in ['avg_force_magnitude', 'avg_stress_magnitude']:
        if col + '_new' in enriched_df.columns:
            # Use .get() for the fillna value to avoid error if the column doesn't exist in the original df
            enriched_df[col] = enriched_df[col + '_new'].fillna(enriched_df.get(col, pd.NA))
            enriched_df.drop(columns=[col + '_new'], inplace=True)

    # 6. Save the final enriched file
    enriched_df.to_csv(enriched_data_path, index=False)
    print(f"\n✅ Successfully created enriched dataset at: {enriched_data_path}")
    print(f"Final dataset has {enriched_df.shape[0]} rows and {enriched_df.shape[1]} columns.")

if __name__ == "__main__":
    main() 