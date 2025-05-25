import os
import argparse

# Import the workflows from the neb_analysis_tools package
from neb_analysis_tools import workflows 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Allegro vs. VASP comparison analyses.")
    parser.add_argument("--data_dir", type=str, default="../data/post_vasp", 
                        help="Base directory for VASP data (e.g., containing MRS folder).")
    parser.add_argument("--model_path", type=str, default="../data/potentials/gen-7-05-19_compiled_model.nequip.pt2",
                        help="Path to the compiled model file.")
    parser.add_argument("--model_type", type=str, default="allegro", choices=["allegro", "mace"],
                        help="Type of machine learning potential (allegro or mace).")
    parser.add_argument("--results_dir", type=str, default="analysis_results_refactored",
                        help="Main directory to save all analysis results.")
    
    parser.add_argument("--skip_perfect", action='store_true', help="Skip perfect system analysis.")
    parser.add_argument("--skip_vacancy", action='store_true', help="Skip vacancy system analysis.")
    parser.add_argument("--skip_neb", action='store_true', help="Skip NEB comparison analysis.")

    args = parser.parse_args()

    DATA_DIR = args.data_dir
    MODEL_PATH = args.model_path
    MODEL_TYPE = args.model_type
    RESULTS_MAIN_DIR = args.results_dir
    
    os.makedirs(RESULTS_MAIN_DIR, exist_ok=True)

    # Define paths for sub-results directories
    perfect_results_dir = os.path.join(RESULTS_MAIN_DIR, "perfect")
    vacancy_results_dir = os.path.join(RESULTS_MAIN_DIR, "vacancy")
    neb_results_dir = os.path.join(RESULTS_MAIN_DIR, "neb")

    # --- Validations ---
    model_ok = True
    data_ok = True
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Allegro model not found at {MODEL_PATH}. Please check the path.")
        model_ok = False
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data directory not found at {DATA_DIR}. Please check the path.")
        data_ok = False
    
    if not model_ok or not data_ok:
        print("Exiting due to missing model or data directory.")
        exit()
    
    # --- Execute Analyses ---
    if not args.skip_perfect:
        print("\nStarting Perfect System Analysis...")
        workflows.analyze_perfect_systems(DATA_DIR, MODEL_PATH, MODEL_TYPE, results_base_dir=perfect_results_dir)
    else:
        print("Skipping Perfect System Analysis.")

    if not args.skip_vacancy:
        print("\nStarting Vacancy System Analysis...")
        # Vacancy analysis depends on results from perfect analysis (relaxed perfect structures)
        if not os.path.isdir(perfect_results_dir) or not any(f.endswith('.xyz') for r,ds,fs in os.walk(perfect_results_dir) for f in fs):
            print(f"Warning: Perfect systems results directory ({perfect_results_dir}) is empty or does not contain .xyz files.")
            if not args.skip_perfect: # If perfect wasn't skipped but still no results, it might have failed.
                 print("Perfect analysis might have failed or produced no output. Vacancy analysis might be incomplete or fail.")
            else: # If perfect was skipped, we must run it if vacancy analysis is requested and dir is missing.
                print("Perfect analysis was skipped, but its results are needed for vacancy analysis. Consider running perfect analysis first or ensuring results are present.")
                # Decide: either exit, or run perfect analysis now if it was skipped.
                # For now, just warn and proceed, it might fail gracefully within analyze_vacancy_systems.
        workflows.analyze_vacancy_systems(DATA_DIR, MODEL_PATH, MODEL_TYPE,
                                      perfect_ml_relaxed_structures_dir=perfect_results_dir, 
                                      results_base_dir=vacancy_results_dir)
    else:
        print("Skipping Vacancy System Analysis.")

    if not args.skip_neb:
        print("\nStarting NEB Comparison Analysis...")
        # NEB analysis also depends on perfect system results
        if not os.path.isdir(perfect_results_dir) or not any(f.endswith('.xyz') for r,ds,fs in os.walk(perfect_results_dir) for f in fs):
            print(f"Warning: Perfect systems results directory ({perfect_results_dir}) is empty or does not contain .xyz files.")
            if not args.skip_perfect: # If perfect wasn't skipped but still no results.
                 print("Perfect analysis might have failed or produced no output. NEB analysis might be incomplete or fail.")
            else:
                print("Perfect analysis was skipped, but its results are needed for NEB analysis. Consider running perfect analysis first or ensuring results are present.")
        workflows.analyze_neb_comparisons(DATA_DIR, MODEL_PATH, MODEL_TYPE,
                                        perfect_ml_relaxed_structures_dir=perfect_results_dir, 
                                        results_base_dir=neb_results_dir)
    else:
        print("Skipping NEB Comparison Analysis.")

    print("\n--- All selected analyses complete. ---")
