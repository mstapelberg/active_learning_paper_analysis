# High-level analysis workflows
import os
import pandas as pd
import torch
from ase.io import read, write
import numpy as np # For DataFrame creation and NaN handling
import re

# Import from other modules in this package
from . import io_utils
from . import ase_simulation
from . import structure_modification
from . import comparison
from . import plotting

def analyze_perfect_systems(data_base_path, model_path, model_type="allegro", results_base_dir="results/perfect"):
    print("--- Analyzing Perfect Systems ---")
    os.makedirs(results_base_dir, exist_ok=True)

    vasp_initial_perf_path = os.path.join(data_base_path, "MRS/post_mrs_nostatic/mrs_perf")
    vasp_final_static_perf_path = os.path.join(data_base_path, "MRS/post_mrs_static/mrs_perf_static")

    if not os.path.isdir(vasp_initial_perf_path):
        print(f"Error: VASP initial perfect structures path not found: {vasp_initial_perf_path}")
        return
    if not os.path.isdir(vasp_final_static_perf_path):
        print(f"Error: VASP final static perfect structures path not found: {vasp_final_static_perf_path}")
        # We can proceed without this but warn that final comparison will be skipped.
        # For now, let's make it a critical check for this workflow part.
        # return # Or allow partial analysis

    composition_folders = [f for f in os.listdir(vasp_initial_perf_path) if os.path.isdir(os.path.join(vasp_initial_perf_path, f))]
    
    all_comparison_data = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        ml_calc = ase_simulation.get_calculator(model_path, model_type, device)
    except Exception as e:
        print(f"Error loading {model_type} model from {model_path}: {e}. Aborting perfect systems analysis.")
        return

    for comp_folder in composition_folders:
        print(f"\nProcessing perfect system: {comp_folder}")
        comp_results_dir = os.path.join(results_base_dir, comp_folder)
        os.makedirs(comp_results_dir, exist_ok=True)

        # 1. Get initial VASP structure
        initial_vasp_atoms = io_utils.get_initial_vasp_structure(vasp_initial_perf_path, comp_folder)
        if not initial_vasp_atoms:
            print(f"Skipping {comp_folder} due to missing initial VASP structure.")
            continue
        write(os.path.join(comp_results_dir, "initial_vasp.xyz"), initial_vasp_atoms)

        # 2. Relax with Allegro
        print("Relaxing with Allegro...")
        allegro_traj_file = os.path.join(comp_results_dir, "allegro_relaxation.traj")
        relaxed_allegro_atoms = None
        try:
            relaxed_allegro_atoms = ase_simulation.relax(initial_vasp_atoms, ml_calc, relax_cell=True, fmax=0.01, trajectory_file=allegro_traj_file)
            write(os.path.join(comp_results_dir, "relaxed_allegro_perfect.xyz"), relaxed_allegro_atoms)
        except Exception as e:
            print(f"Allegro relaxation failed for {comp_folder}: {e}")
            # relaxed_allegro_atoms remains None

        # 3. Get final VASP static structure
        final_vasp_atoms = io_utils.get_final_vasp_static_structure(vasp_final_static_perf_path, comp_folder)
        if not final_vasp_atoms:
            print(f"Warning: No final VASP static structure found for {comp_folder}. Comparison of final states will be partial.")
        else:
            write(os.path.join(comp_results_dir, "final_vasp_static_perfect.xyz"), final_vasp_atoms)

        # 4. Compare final structures (ML relaxed vs VASP static)
        if relaxed_allegro_atoms and final_vasp_atoms:
            print("Comparing ML relaxed structure with VASP static structure...")
            final_struct_comparison_results = comparison.compare_structures(relaxed_allegro_atoms, final_vasp_atoms)
            final_comparison_df = pd.DataFrame([final_struct_comparison_results])
            final_comparison_df.to_csv(os.path.join(comp_results_dir, f"comparison_{model_type}_vs_vasp_static_perfect.csv"), index=False)
            print(final_comparison_df)
            
            # Run diagnostics if large displacements detected
            max_disp = final_struct_comparison_results.get('max_atomic_displacement', 0)
            if max_disp > 1.0:  # Threshold for "large" displacement
                print(f"🔍 Large displacement detected ({max_disp:.3f} Å). Running diagnostics...")
                comparison.diagnose_structure_mismatch(
                    relaxed_allegro_atoms, final_vasp_atoms, 
                    results_dir=comp_results_dir, 
                    name_prefix=f"{comp_folder}_perfect_{model_type}_vs_vasp"
                )
            
            all_comparison_data.append(final_comparison_df.assign(composition=comp_folder, type="perfect"))
        elif not relaxed_allegro_atoms:
            print(f"Skipping final structure comparison for {comp_folder} because ML relaxation failed or was skipped.")
        
        # 5. Compare relaxation trajectories (Allegro vs VASP)
        vasp_relax_outcar_main_path = io_utils.get_vasp_relaxation_trajectory_paths(vasp_initial_perf_path, comp_folder)
        if vasp_relax_outcar_main_path:
            vasp_traj_data, _ = io_utils.parse_vasp_relaxation_trajectory(vasp_relax_outcar_main_path)
        else:
            vasp_traj_data = []
            print(f"Skipping VASP trajectory part for {comp_folder} because VASP relaxation OUTCAR path is missing.")

        if (allegro_traj_file and os.path.exists(allegro_traj_file)) or vasp_traj_data:
            print("Comparing relaxation trajectories...")
            traj_comp_data_extracted = comparison.extract_relaxation_trajectory_data(allegro_traj_file, vasp_traj_data)
            
            plotting.plot_relaxation_trajectories(traj_comp_data_extracted, 
                                                  save_path_prefix=os.path.join(comp_results_dir, f"{comp_folder}_perfect_traj_comparison"))
            # Save trajectory comparison data to CSV
            pd.DataFrame({
                'allegro_E': pd.Series(traj_comp_data_extracted['allegro_energies']),
                'allegro_F': pd.Series(traj_comp_data_extracted['allegro_fmax']),
                'vasp_E': pd.Series(traj_comp_data_extracted['vasp_energies']),
                'vasp_F': pd.Series(traj_comp_data_extracted['vasp_fmax'])
            }).to_csv(os.path.join(comp_results_dir, "trajectory_comparison_data_perfect.csv"), index=False)
        else:
            print(f"Skipping full trajectory comparison for {comp_folder} due to missing Allegro trajectory and VASP trajectory data.")

    if all_comparison_data:
        summary_df = pd.concat(all_comparison_data, ignore_index=True)
        summary_df.to_csv(os.path.join(results_base_dir, "summary_perfect_comparisons.csv"), index=False)
    print("--- Finished Perfect Systems Analysis ---")


def analyze_vacancy_systems(data_base_path, model_path, model_type, perfect_ml_relaxed_structures_dir, results_base_dir="results/vacancy"):
    print("--- Analyzing Vacancy Systems ---")
    os.makedirs(results_base_dir, exist_ok=True)

    # Path to VASP vacancy data (nostatic for trajectories, static for final structures)
    vasp_vac_nostatic_base = os.path.join(data_base_path, "MRS/post_mrs_nostatic/mrs_vac")
    vasp_vac_static_base = os.path.join(data_base_path, "MRS/post_mrs_static/mrs_vac_static")

    if not os.path.isdir(perfect_ml_relaxed_structures_dir):
        print(f"Error: Directory for ML-relaxed perfect structures not found: {perfect_ml_relaxed_structures_dir}. Cannot proceed with vacancy analysis.")
        return
    if not os.path.isdir(vasp_vac_nostatic_base):
        print(f"Warning: VASP nostatic vacancy data path not found: {vasp_vac_nostatic_base}. Trajectory comparisons may be affected.")
    if not os.path.isdir(vasp_vac_static_base):
        print(f"Warning: VASP static vacancy data path not found: {vasp_vac_static_base}. Final structure comparisons may be affected.")

    # Iterate based on compositions for which VASP vacancy data exists
    # This assumes that the subfolder names in `vasp_vac_static_base` are the composition names.
    if not os.path.isdir(vasp_vac_static_base):
        print(f"Error: VASP static vacancy data directory not found: {vasp_vac_static_base}. Cannot determine compositions for vacancy analysis.")
        return
        
    composition_folders_with_vasp_vac_data = [f for f in os.listdir(vasp_vac_static_base) if os.path.isdir(os.path.join(vasp_vac_static_base, f))]

    all_vacancy_comparison_data = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        ml_calc = ase_simulation.get_calculator(model_path, model_type, device)
    except Exception as e:
        print(f"Error loading {model_type} model from {model_path}: {e}. Aborting vacancy systems analysis.")
        return

    for comp_folder in composition_folders_with_vasp_vac_data:
        print(f"\nProcessing vacancies for composition: {comp_folder}")
        
        ml_perf_relaxed_xyz = os.path.join(perfect_ml_relaxed_structures_dir, comp_folder, "relaxed_allegro_perfect.xyz")
        if not os.path.exists(ml_perf_relaxed_xyz):
            print(f"Skipping vacancy analysis for {comp_folder}: ML relaxed perfect structure not found at {ml_perf_relaxed_xyz}")
            continue
        try:
            perfect_ml_atoms = read(ml_perf_relaxed_xyz)
        except Exception as e:
            print(f"Error reading ML relaxed perfect XYZ {ml_perf_relaxed_xyz} for {comp_folder}: {e}")
            continue

        # Find corresponding VASP vacancy subfolders (e.g., End_Index_XXX, Start_Index_XXX)
        vasp_comp_vac_static_path = os.path.join(vasp_vac_static_base, comp_folder)
        if not os.path.isdir(vasp_comp_vac_static_path):
            print(f"No VASP static vacancy data found for {comp_folder} at {vasp_comp_vac_static_path}")
            continue
            
        index_subfolders = [d for d in os.listdir(vasp_comp_vac_static_path) if os.path.isdir(os.path.join(vasp_comp_vac_static_path, d)) and ('End_Index_' in d or 'Start_Index_' in d)]

        if not index_subfolders:
            print(f"No 'Index_' subfolders found for VASP static vacancies in {vasp_comp_vac_static_path} for {comp_folder}. Skipping.")
            continue

        for index_folder_name in index_subfolders:
            print(f"  Processing VASP vacancy index folder: {index_folder_name}")
            comp_results_vac_dir = os.path.join(results_base_dir, comp_folder, index_folder_name)
            os.makedirs(comp_results_vac_dir, exist_ok=True)

            # 1. Determine the atom index for vacancy creation from VASP folder name
            parsed_vac_idx_0_based = io_utils.parse_vacancy_index_from_folder(index_folder_name)
            if parsed_vac_idx_0_based is None:
                print(f"    Could not parse vacancy index from folder {index_folder_name}. Skipping this specific vacancy.")
                continue
            
            print(f"    Targeting vacancy at 0-based index: {parsed_vac_idx_0_based}")

            # 2. Create initial Allegro vacancy from perfect Allegro structure and relax it
            initial_allegro_vacancy_atoms = structure_modification.create_vacancy_from_perfect(
                perfect_ml_atoms, parsed_vac_idx_0_based, comp_results_vac_dir, 
                name_prefix=f"{comp_folder}_{index_folder_name}_allegro"
            )
            
            relaxed_allegro_vacancy_atoms = None
            allegro_vacancy_relax_traj_file = os.path.join(comp_results_vac_dir, f"{comp_folder}_{index_folder_name}_allegro_vacancy_relaxation.traj")
            if initial_allegro_vacancy_atoms:
                print(f"    Relaxing Allegro vacancy for {comp_folder}, index {parsed_vac_idx_0_based}...")
                try:
                    relaxed_allegro_vacancy_atoms = ase_simulation.relax(
                        initial_allegro_vacancy_atoms, ml_calc, 
                        relax_cell=False, fmax=0.01, trajectory_file=allegro_vacancy_relax_traj_file
                    )
                    write(os.path.join(comp_results_vac_dir, f"{comp_folder}_{index_folder_name}_allegro_vacancy_relaxed.xyz"), relaxed_allegro_vacancy_atoms)
                except Exception as e:
                    print(f"    Allegro vacancy relaxation failed for {comp_folder}, index {parsed_vac_idx_0_based}: {e}")
            else:
                print(f"    Skipping Allegro vacancy relaxation for {comp_folder}, index {parsed_vac_idx_0_based} due to creation failure.")

            # 3. Get corresponding final VASP static relaxed vacancy structure
            # `get_vasp_vacancy_structure_and_path` now looks inside comp_folder/index_folder_name
            # So, base_path should be vasp_vac_static_base, comp_folder is comp_folder, and target_subdir is index_folder_name.
            # This needs slight refactor of get_vasp_vacancy_structure_and_path or how it's called.
            # Let's adjust: get_vasp_vacancy_structure_and_path will search within comp_path for a subdir matching target_subdir_name_part.
            # Here, target_subdir_name_part will be index_folder_name itself.
            final_vasp_vacancy_atoms, _ = io_utils.get_vasp_vacancy_structure_and_path(vasp_vac_static_base, comp_folder, target_subdir_name_part=index_folder_name)
            
            if final_vasp_vacancy_atoms:
                write(os.path.join(comp_results_vac_dir, f"{comp_folder}_{index_folder_name}_vasp_vacancy_relaxed_static.xyz"), final_vasp_vacancy_atoms)
            else:
                print(f"    Warning: Could not load final VASP static relaxed vacancy for {comp_folder}, index folder {index_folder_name}. Comparison of final states will be skipped.")

            # 4. Compare final ML relaxed vacancy with VASP static relaxed vacancy
            if relaxed_allegro_vacancy_atoms and final_vasp_vacancy_atoms:
                print(f"    Comparing final ML relaxed vacancy with final VASP static relaxed vacancy for {index_folder_name}...")
                final_vac_struct_comp = comparison.compare_structures(relaxed_allegro_vacancy_atoms, final_vasp_vacancy_atoms)
                final_vac_df = pd.DataFrame([final_vac_struct_comp])
                final_vac_df.to_csv(os.path.join(comp_results_vac_dir, f"comparison_{model_type}_vs_vasp_vacancy_static.csv"), index=False)
                print(final_vac_df)
                
                # Run diagnostics if large displacements detected
                max_disp = final_vac_struct_comp.get('max_atomic_displacement', 0)
                if max_disp > 1.0:  # Threshold for "large" displacement
                    print(f"    🔍 Large displacement detected ({max_disp:.3f} Å). Running diagnostics...")
                    comparison.diagnose_structure_mismatch(
                        relaxed_allegro_vacancy_atoms, final_vasp_vacancy_atoms, 
                        results_dir=comp_results_vac_dir, 
                        name_prefix=f"{comp_folder}_{index_folder_name}_vacancy_{model_type}_vs_vasp"
                    )
                
                all_vacancy_comparison_data.append(final_vac_df.assign(composition=comp_folder, vacancy_details=index_folder_name, type="vacancy_static"))
            elif not relaxed_allegro_vacancy_atoms:
                print(f"    Skipping final vacancy structure comparison for {comp_folder}/{index_folder_name} as ML vacancy relaxation failed or was skipped.")

            # 5. Compare vacancy relaxation trajectories (Allegro vs VASP nostatic)
            # VASP nostatic vacancy data should also be in a similar subfolder structure.
            _, vasp_vac_relax_outcar_path = io_utils.get_vasp_vacancy_structure_and_path(vasp_vac_nostatic_base, comp_folder, target_subdir_name_part=index_folder_name)
            
            vasp_vac_traj_data_list = []
            if vasp_vac_relax_outcar_path:
                 vasp_vac_traj_data_list, _ = io_utils.parse_vasp_relaxation_trajectory(vasp_vac_relax_outcar_path)
            else:
                print(f"    No VASP nostatic OUTCAR found for trajectory for {comp_folder}/{index_folder_name}")

            if (allegro_vacancy_relax_traj_file and os.path.exists(allegro_vacancy_relax_traj_file)) or vasp_vac_traj_data_list:
                print(f"    Comparing vacancy relaxation trajectories for {comp_folder}/{index_folder_name}...")
                vac_traj_data_extracted = comparison.extract_relaxation_trajectory_data(allegro_vacancy_relax_traj_file, vasp_vac_traj_data_list)
                
                plotting.plot_relaxation_trajectories(vac_traj_data_extracted, 
                                                      save_path_prefix=os.path.join(comp_results_vac_dir, f"{comp_folder}_{index_folder_name}_vac_traj_comparison"))
                pd.DataFrame({
                    'allegro_E': pd.Series(vac_traj_data_extracted['allegro_energies']),
                    'allegro_F': pd.Series(vac_traj_data_extracted['allegro_fmax']),
                    'vasp_E': pd.Series(vac_traj_data_extracted['vasp_energies']),
                    'vasp_F': pd.Series(vac_traj_data_extracted['vasp_fmax'])
                }).to_csv(os.path.join(comp_results_vac_dir, "trajectory_comparison_data_vacancy.csv"), index=False)
            else:
                print(f"    Skipping vacancy trajectory comparison for {comp_folder}/{index_folder_name} due to missing Allegro trajectory and VASP trajectory data.")

    if all_vacancy_comparison_data:
        summary_vac_df = pd.concat(all_vacancy_comparison_data, ignore_index=True)
        summary_vac_df.to_csv(os.path.join(results_base_dir, "summary_vacancy_comparisons.csv"), index=False)
    print("--- Finished Vacancy Systems Analysis ---")

def analyze_neb_comparisons(data_base_path, model_path, model_type, perfect_ml_relaxed_structures_dir, results_base_dir="results/neb"):
    print("--- Analyzing NEB Comparisons ---")
    os.makedirs(results_base_dir, exist_ok=True)

    vasp_neb_static_base = os.path.join(data_base_path, "MRS/post_mrs_static/mrs_neb_static")

    if not os.path.isdir(perfect_ml_relaxed_structures_dir):
        print(f"Error: Directory for ML-relaxed perfect structures not found: {perfect_ml_relaxed_structures_dir}. Cannot proceed with NEB analysis.")
        return
    if not os.path.isdir(vasp_neb_static_base):
        print(f"Error: VASP static NEB data directory not found: {vasp_neb_static_base}. Skipping NEB analysis.")
        return

    neb_system_folders = [f for f in os.listdir(vasp_neb_static_base) if os.path.isdir(os.path.join(vasp_neb_static_base, f))]
    
    all_neb_summaries = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        ml_calc = ase_simulation.get_calculator(model_path, model_type, device)
    except Exception as e:
        print(f"Error loading {model_type} model from {model_path}: {e}. Aborting NEB analysis.")
        return

    for neb_sys_folder_name in neb_system_folders:
        print(f"\nProcessing NEB system: {neb_sys_folder_name}")        
        # Extract composition part from folder name like "Cr4Ti12V107WZr_101_to_97"
        comp_part_match = re.match(r"(.+?)_\d+_to_\d+", neb_sys_folder_name)
        if not comp_part_match:
            print(f"Could not parse composition from NEB folder name: {neb_sys_folder_name}. Skipping.")
            continue
        comp_folder = comp_part_match.group(1)

        sys_results_dir = os.path.join(results_base_dir, neb_sys_folder_name)
        os.makedirs(sys_results_dir, exist_ok=True)

        # 1. Load Allegro-relaxed perfect structure for this composition
        allegro_perf_relaxed_xyz = os.path.join(perfect_ml_relaxed_structures_dir, comp_folder, "relaxed_allegro_perfect.xyz")
        if not os.path.exists(allegro_perf_relaxed_xyz):
            print(f"Skipping NEB for {neb_sys_folder_name}: Allegro relaxed perfect structure for {comp_folder} not found at {allegro_perf_relaxed_xyz}")
            continue
        try:
            perfect_allegro_atoms = read(allegro_perf_relaxed_xyz)
        except Exception as e:
            print(f"Error reading Allegro relaxed perfect XYZ {allegro_perf_relaxed_xyz} for NEB system {neb_sys_folder_name}: {e}")
            continue

        # 2. Parse start and end atom indices for vacancy creation from NEB folder name
        neb_indices_0_based = io_utils.parse_neb_indices_from_folder(neb_sys_folder_name)
        if neb_indices_0_based is None:
            print(f"Could not parse NEB indices from folder {neb_sys_folder_name}. Skipping this NEB system.")
            continue
        start_idx_0_based, end_idx_0_based = neb_indices_0_based

        # 3. Create Allegro NEB start and end images from the perfect relaxed structure
        allegro_start_atoms, allegro_end_atoms = structure_modification.create_allegro_neb_endpoints(
            perfect_allegro_atoms, start_idx_0_based, end_idx_0_based, 
            sys_results_dir, system_name=f"{neb_sys_folder_name}_allegro_endpoints"
        )
        if not allegro_start_atoms or not allegro_end_atoms:
            print(f"Failed to create Allegro NEB endpoints for {neb_sys_folder_name}. Skipping.")
            continue

        # 4. Get VASP NEB path data (from static NEB results for converged comparison)
        # The function get_vasp_neb_endpoints_and_path_data returns the VASP *converged* start/end images from 00/OUTCAR and NN/OUTCAR
        # and the list of all VASP image energies/fmax along the converged path.
        vasp_conv_start_img, vasp_conv_end_img, vasp_energies, vasp_fmax, vasp_image_atoms_list = \
            io_utils.get_vasp_neb_endpoints_and_path_data(vasp_neb_static_base, neb_sys_folder_name)

        if not vasp_conv_start_img or not vasp_conv_end_img:
            print(f"Skipping NEB comparison for {neb_sys_folder_name} due to missing VASP converged start/end images.")
            continue
        if not vasp_energies or len(vasp_energies) < 2:
            print(f"Skipping NEB comparison for {neb_sys_folder_name} due to insufficient VASP NEB path data (energies). Need at least 2 images.")
            continue
        
        if vasp_image_atoms_list and all(img is not None for img in vasp_image_atoms_list):
            write(os.path.join(sys_results_dir, f"{neb_sys_folder_name}_vasp_neb_path_converged.xyz"), vasp_image_atoms_list)
        
        # 5. Run Allegro NEB using the *newly created* Allegro start and end points
        print(f"Running {model_type.upper()} NEB for {neb_sys_folder_name}...")
        num_intermediate_images_allegro = len(vasp_energies) - 2 # Match number of intermediate images in VASP path
        if num_intermediate_images_allegro < 1: num_intermediate_images_allegro = 1 # Ensure at least one intermediate
        
        allegro_neb_traj_dir = os.path.join(sys_results_dir, f"{model_type}_neb_optimization_trajectory")
        # os.makedirs(allegro_neb_traj_dir, exist_ok=True) # neb_relax will create it

        allegro_neb_obj = None
        try:
            allegro_neb_obj = ase_simulation.neb_relax(
                allegro_start_atoms, allegro_end_atoms, model_path, model_type,
                num_images=num_intermediate_images_allegro, 
                fmax=0.05, # Standard NEB fmax convergence
                steps=300, # Adjust NEB steps as needed
                trajectory_dir=allegro_neb_traj_dir, # This is for the band optimization trajectory
                system_name=neb_sys_folder_name
            )
            if allegro_neb_obj and allegro_neb_obj.images:
                 write(os.path.join(sys_results_dir, f"{neb_sys_folder_name}_{model_type}_neb_path_converged.xyz"), allegro_neb_obj.images)
        except Exception as e:
            print(f"{model_type.upper()} NEB calculation failed for {neb_sys_folder_name}: {e}")
            # allegro_neb_obj remains None

        # 6. Extract NEB comparison data
        # Since we simplified trajectory writing, the band trajectory file might not exist
        allegro_neb_band_opt_traj_file = os.path.join(allegro_neb_traj_dir, f"{neb_sys_folder_name}_neb_path.traj")
        if not os.path.exists(allegro_neb_band_opt_traj_file):
            allegro_neb_band_opt_traj_file = None  # Signal that no trajectory file is available
        
        neb_comparison_data = comparison.extract_neb_comparison_data(
            allegro_neb_obj, allegro_neb_band_opt_traj_file, 
            vasp_energies, vasp_fmax
        )
        
        plotting.plot_neb_comparison(neb_comparison_data, sys_results_dir, neb_sys_folder_name)
        
        # Save detailed data to CSV from extracted data
        max_images = max(neb_comparison_data['num_allegro_images'], neb_comparison_data['num_vasp_images'])
        df_data = {
            'image_index': list(range(max_images)),
            'allegro_energy_norm': list(neb_comparison_data['allegro_energies_norm']) + [np.nan]*(max_images - len(neb_comparison_data['allegro_energies_norm'])),
            'allegro_fmax_final': list(neb_comparison_data['allegro_fmax_final']) + [np.nan]*(max_images - len(neb_comparison_data['allegro_fmax_final'])),
            'vasp_energy_norm': list(neb_comparison_data['vasp_energies_norm']) + [np.nan]*(max_images - len(neb_comparison_data['vasp_energies_norm'])),
            'vasp_fmax_final': list(neb_comparison_data['vasp_fmax_final']) + [np.nan]*(max_images - len(neb_comparison_data['vasp_fmax_final']))
        }
        neb_data_df = pd.DataFrame(df_data)
        csv_filename = os.path.join(sys_results_dir, f"neb_data_converged_{neb_sys_folder_name}.csv")
        neb_data_df.to_csv(csv_filename, index=False)
        print(f"Saved converged NEB data to {csv_filename}")

        # Save summary of barriers
        summary_entry = {
            'system': neb_sys_folder_name,
            'allegro_barrier_fwd_eV': neb_comparison_data['allegro_barrier_fwd'],
            'vasp_barrier_fwd_eV': neb_comparison_data['vasp_barrier_fwd'],
            'barrier_diff_fwd_eV': neb_comparison_data['barrier_diff_fwd'],
            'allegro_barrier_rev_eV': neb_comparison_data['allegro_barrier_rev'],
            'vasp_barrier_rev_eV': neb_comparison_data['vasp_barrier_rev'],
            'barrier_diff_rev_eV': neb_comparison_data['barrier_diff_rev']
        }
        all_neb_summaries.append(pd.DataFrame([summary_entry]))

    if all_neb_summaries:
        summary_neb_df = pd.concat(all_neb_summaries, ignore_index=True)
        summary_neb_df.to_csv(os.path.join(results_base_dir, "summary_neb_comparisons.csv"), index=False)
    print("--- Finished NEB Comparisons Analysis ---")

# Add workflow functions here later 