# Core comparison logic for structures and trajectories
import numpy as np
import pandas as pd
from ase.io.trajectory import Trajectory # For reading Allegro trajectory in relaxation comparison
import os # For path checks in relaxation comparison
from ase.io import write

# Directly from neb_analysis.py, slightly adapted
def compare_structures(atoms_mlip, atoms_vasp):
    """Compares key properties of two ASE Atoms objects (MLIP vs VASP)."""
    results = {}
    if not atoms_mlip or not atoms_vasp:
        print("Warning: One or both structures are None in compare_structures. Skipping comparison.")
        return {
            'energy_diff': np.nan, 'mlip_energy_total': np.nan, 'vasp_energy_total': np.nan,
            'num_atoms': 0, 'cell_param_diffs': {k: np.nan for k in ['da','db','dc','dalpha','dbeta','dgamma']},
            'cell_rmse': np.nan, 'mean_atomic_displacement': np.nan, 'max_atomic_displacement': np.nan,
            'rms_atomic_displacement': np.nan, 'atomic_displacements_all': []
        }

    results['mlip_energy_total'] = atoms_mlip.get_potential_energy()
    results['vasp_energy_total'] = atoms_vasp.get_potential_energy()
    results['energy_diff'] = results['mlip_energy_total'] - results['vasp_energy_total']
    results['num_atoms'] = len(atoms_mlip)

    if atoms_mlip.cell is not None and atoms_vasp.cell is not None and atoms_mlip.pbc.all() and atoms_vasp.pbc.all():
        cell_mlip_params = np.array(atoms_mlip.cell.cellpar())
        cell_vasp_params = np.array(atoms_vasp.cell.cellpar())
        
        cell_param_diff_values = cell_mlip_params - cell_vasp_params
        results['cell_param_diffs'] = {
            'da': cell_param_diff_values[0], 'db': cell_param_diff_values[1], 'dc': cell_param_diff_values[2],
            'dalpha': cell_param_diff_values[3], 'dbeta': cell_param_diff_values[4], 'dgamma': cell_param_diff_values[5]
        }
        results['cell_rmse'] = np.sqrt(np.mean(cell_param_diff_values**2))
    else:
        print("Warning: Cell information missing or not periodic for one or both structures. Skipping cell comparison.")
        results['cell_param_diffs'] = {k: np.nan for k in ['da','db','dc','dalpha','dbeta','dgamma']}
        results['cell_rmse'] = np.nan

    if len(atoms_mlip) != len(atoms_vasp):
        print(f"Warning: Atom counts differ (MLIP: {len(atoms_mlip)}, VASP: {len(atoms_vasp)}). Atomic displacement calculation might be misleading or fail.")
        # Set displacement metrics to NaN and return what has been computed so far.
        results['mean_atomic_displacement'] = np.nan
        results['max_atomic_displacement'] = np.nan
        results['rms_atomic_displacement'] = np.nan
        results['atomic_displacements_all'] = []
        return results # Or raise ValueError depending on desired strictness

    pos_mlip = atoms_mlip.get_positions()
    pos_vasp = atoms_vasp.get_positions()
    
    # Use VASP cell for MIC if cells are different, assuming VASP is the reference for atomic positions.
    # If cells are very different, this comparison might be less meaningful.
    # Ensure the cell used for MIC is well-defined and atoms are scaled if necessary.
    # For now, direct subtraction with MIC based on VASP cell.
    # This assumes atoms are ordered identically.
    if atoms_vasp.cell is not None and atoms_vasp.pbc.all():
        vasp_cell_matrix = atoms_vasp.get_cell()
        displacement_vectors_cartesian = np.zeros_like(pos_vasp)
        for i in range(len(atoms_vasp)):
            raw_cartesian_diff = pos_mlip[i] - pos_vasp[i]
            # Apply Minimum Image Convention (MIC)
            # Convert raw Cartesian difference to fractional coordinates in VASP cell
            # Then wrap fractional coordinates to [-0.5, 0.5), then convert back to Cartesian.
            fractional_diff = np.linalg.solve(vasp_cell_matrix.T, raw_cartesian_diff)
            fractional_diff_mic = fractional_diff - np.round(fractional_diff)
            displacement_vectors_cartesian[i] = np.dot(vasp_cell_matrix.T, fractional_diff_mic)
        
        displacement_magnitudes = np.linalg.norm(displacement_vectors_cartesian, axis=1)
        
        results['mean_atomic_displacement'] = np.mean(displacement_magnitudes)
        results['max_atomic_displacement'] = np.max(displacement_magnitudes)
        results['rms_atomic_displacement'] = np.sqrt(np.mean(displacement_magnitudes**2))
        results['atomic_displacements_all'] = displacement_magnitudes.tolist() # Store as list for CSV/JSON
    else:
        print("Warning: VASP cell is not defined or not periodic. Skipping MIC for atomic displacements. Raw displacements will be used.")
        raw_displacements = pos_mlip - pos_vasp
        displacement_magnitudes = np.linalg.norm(raw_displacements, axis=1)
        results['mean_atomic_displacement'] = np.mean(displacement_magnitudes)
        results['max_atomic_displacement'] = np.max(displacement_magnitudes)
        results['rms_atomic_displacement'] = np.sqrt(np.mean(displacement_magnitudes**2))
        results['atomic_displacements_all'] = displacement_magnitudes.tolist()

    return results

def extract_relaxation_trajectory_data(allegro_traj_file, vasp_parsed_traj_data):
    """
    Extracts and prepares energy and fmax data from Allegro trajectory file 
    and parsed VASP trajectory data for comparison.
    
    Args:
        allegro_traj_file (str): Path to Allegro ASE trajectory file.
        vasp_parsed_traj_data (list of dict): Parsed VASP trajectory data from io_utils.parse_vasp_relaxation_trajectory.
    
    Returns:
        dict: Contains lists of energies and fmax for Allegro and VASP, and step counts.
    """
    comparison_data = {
        'allegro_energies': [], 'allegro_fmax': [], 'allegro_steps': 0,
        'vasp_energies': [], 'vasp_fmax': [], 'vasp_steps': 0,
    }
    # Parse Allegro trajectory
    if allegro_traj_file and os.path.exists(allegro_traj_file):
        try:
            allegro_ase_traj = Trajectory(allegro_traj_file, 'r')
            for atoms_step in allegro_ase_traj:
                comparison_data['allegro_energies'].append(atoms_step.get_potential_energy())
                forces = atoms_step.get_forces()
                comparison_data['allegro_fmax'].append(np.sqrt((forces**2).sum(axis=1).max()))
            comparison_data['allegro_steps'] = len(comparison_data['allegro_energies'])
        except Exception as e:
            print(f"Error reading Allegro trajectory {allegro_traj_file}: {e}")

    # Process VASP trajectory data (already parsed)
    if vasp_parsed_traj_data:
        for step_data in vasp_parsed_traj_data:
            comparison_data['vasp_energies'].append(step_data['energy'])
            comparison_data['vasp_fmax'].append(step_data['fmax'])
        comparison_data['vasp_steps'] = len(comparison_data['vasp_energies'])
    
    return comparison_data

def extract_neb_comparison_data(allegro_neb_obj, allegro_neb_band_traj_file, vasp_neb_energies, vasp_neb_fmax):
    """
    Extracts and calculates data for comparing NEB paths from Allegro and VASP.
    Does not perform plotting, only data extraction and barrier calculation.

    Args:
        allegro_neb_obj: The converged ASE DyNEB object from Allegro.
        allegro_neb_band_traj_file (str): Path to Allegro NEB band trajectory (evolution of the band).
        vasp_neb_energies (list): List of final energies for each VASP NEB image.
        vasp_neb_fmax (list): List of final fmax for each VASP NEB image.

    Returns:
        dict: Contains barrier information, and lists of energies/fmax for converged paths.
              Also includes Allegro band evolution data if trajectory file is provided.
    """
    if not allegro_neb_obj:
        print("Allegro NEB object missing. Cannot extract NEB comparison data.")
        return {
            'allegro_barrier_fwd': np.nan, 'vasp_barrier_fwd': np.nan, 'barrier_diff_fwd': np.nan,
            'allegro_barrier_rev': np.nan, 'vasp_barrier_rev': np.nan, 'barrier_diff_rev': np.nan,
            'allegro_energies_norm': [], 'allegro_fmax_final': [], 
            'vasp_energies_norm': [], 'vasp_fmax_final': [],
            'num_allegro_images': 0, 'num_vasp_images': 0,
            'allegro_band_energies_over_time': [], 'allegro_band_fmax_over_time': []
        }

    # Allegro NEB path properties (converged final state)
    allegro_images = allegro_neb_obj.images
    allegro_energies_final = [img.get_potential_energy() for img in allegro_images]
    allegro_fmax_final = [np.sqrt((img.get_forces()**2).sum(axis=1).max()) for img in allegro_images]
    
    # Normalize energies to the first image
    allegro_energies_norm = (np.array(allegro_energies_final) - allegro_energies_final[0]) if allegro_energies_final else np.array([])
    vasp_energies_norm = (np.array(vasp_neb_energies) - vasp_neb_energies[0]) if (vasp_neb_energies and not np.isnan(vasp_neb_energies[0])) else np.array(vasp_neb_energies)
    if not vasp_neb_energies: # Handle case of empty vasp_neb_energies
        vasp_energies_norm = np.array([])

    # Calculate barriers
    allegro_barrier_fwd = np.max(allegro_energies_norm) if allegro_energies_norm.size > 0 else np.nan
    allegro_barrier_rev = (np.max(allegro_energies_norm) - allegro_energies_norm[-1]) if allegro_energies_norm.size > 1 else np.nan
    
    vasp_barrier_fwd = np.max(vasp_energies_norm) if vasp_energies_norm.size > 0 and not np.all(np.isnan(vasp_energies_norm)) else np.nan
    vasp_barrier_rev = (np.max(vasp_energies_norm) - vasp_energies_norm[-1]) if vasp_energies_norm.size > 1 and not np.all(np.isnan(vasp_energies_norm)) else np.nan

    comparison_results = {
        'allegro_barrier_fwd': allegro_barrier_fwd,
        'vasp_barrier_fwd': vasp_barrier_fwd,
        'barrier_diff_fwd': allegro_barrier_fwd - vasp_barrier_fwd if not (np.isnan(allegro_barrier_fwd) or np.isnan(vasp_barrier_fwd)) else np.nan,
        'allegro_barrier_rev': allegro_barrier_rev,
        'vasp_barrier_rev': vasp_barrier_rev,
        'barrier_diff_rev': allegro_barrier_rev - vasp_barrier_rev if not (np.isnan(allegro_barrier_rev) or np.isnan(vasp_barrier_rev)) else np.nan,
        'allegro_energies_norm': allegro_energies_norm.tolist(), 
        'allegro_fmax_final': allegro_fmax_final,
        'vasp_energies_norm': vasp_energies_norm.tolist() if hasattr(vasp_energies_norm, 'tolist') else [],
        'vasp_fmax_final': vasp_neb_fmax if vasp_neb_fmax else [],
        'num_allegro_images': len(allegro_energies_final),
        'num_vasp_images': len(vasp_neb_energies) if hasattr(vasp_neb_energies, '__len__') else 0,
        'allegro_band_energies_over_time': [], 
        'allegro_band_fmax_over_time': [] 
    }

    # Process Allegro NEB optimization trajectory (evolution of the band)
    if allegro_neb_band_traj_file and os.path.exists(allegro_neb_band_traj_file):
        try:
            allegro_band_traj = Trajectory(allegro_neb_band_traj_file, 'r')
            for neb_step_images in allegro_band_traj:
                if not neb_step_images: continue
                step_energies = [img.get_potential_energy() for img in neb_step_images if img.calc]
                step_fmax_values = [np.sqrt((img.get_forces()**2).sum(axis=1).max()) for img in neb_step_images if img.calc]
                
                if step_energies:
                    comparison_results['allegro_band_energies_over_time'].append(step_energies)
                if step_fmax_values:
                    comparison_results['allegro_band_fmax_over_time'].append(np.max(step_fmax_values) if step_fmax_values else np.nan)
            
        except Exception as e:
            print(f"Error processing Allegro NEB band trajectory file {allegro_neb_band_traj_file}: {e}")
    else:
        print(f"ML NEB band trajectory file not found or not provided: {allegro_neb_band_traj_file}. Skipping band evolution analysis.")

    return comparison_results

def diagnose_structure_mismatch(atoms_mlip, atoms_vasp, results_dir=None, name_prefix="diagnosis"):
    """
    Provides detailed diagnostics for structure comparison mismatches.
    Helps identify issues like atom ordering, large displacements, etc.
    
    Args:
        atoms_mlip: ASE Atoms object from ML potential
        atoms_vasp: ASE Atoms object from VASP
        results_dir: Directory to save diagnostic outputs (optional)
        name_prefix: Prefix for output files
    
    Returns:
        dict: Detailed diagnostic information
    """
    if not atoms_mlip or not atoms_vasp:
        print("Warning: One or both structures are None. Cannot diagnose.")
        return {}
    
    print(f"\n=== STRUCTURE COMPARISON DIAGNOSTICS ===")
    print(f"ML structure: {len(atoms_mlip)} atoms")
    print(f"VASP structure: {len(atoms_vasp)} atoms")
    
    # Basic checks
    diagnostics = {
        'num_atoms_ml': len(atoms_mlip),
        'num_atoms_vasp': len(atoms_vasp),
        'atoms_match': len(atoms_mlip) == len(atoms_vasp),
        'composition_ml': {str(k): int(v) for k, v in zip(*np.unique(atoms_mlip.get_chemical_symbols(), return_counts=True))},
        'composition_vasp': {str(k): int(v) for k, v in zip(*np.unique(atoms_vasp.get_chemical_symbols(), return_counts=True))},
    }
    
    print(f"ML composition: {diagnostics['composition_ml']}")
    print(f"VASP composition: {diagnostics['composition_vasp']}")
    
    if len(atoms_mlip) != len(atoms_vasp):
        print("❌ CRITICAL: Atom counts differ!")
        return diagnostics
    
    # Check if compositions match
    comp_match = diagnostics['composition_ml'] == diagnostics['composition_vasp']
    diagnostics['composition_match'] = comp_match
    print(f"Compositions match: {comp_match}")
    
    if not comp_match:
        print("❌ WARNING: Compositions differ!")
    
    # Check atom ordering
    ml_symbols = atoms_mlip.get_chemical_symbols()
    vasp_symbols = atoms_vasp.get_chemical_symbols()
    ordering_match = ml_symbols == vasp_symbols
    diagnostics['ordering_match'] = ordering_match
    
    if not ordering_match:
        print("❌ WARNING: Atom ordering differs!")
        # Find mismatched positions
        mismatches = [(i, ml_symbols[i], vasp_symbols[i]) for i in range(len(ml_symbols)) if ml_symbols[i] != vasp_symbols[i]]
        print(f"First 10 mismatches: {mismatches[:10]}")
        diagnostics['ordering_mismatches'] = mismatches[:20]  # Save first 20
    else:
        print("✅ Atom ordering matches")
        diagnostics['ordering_mismatches'] = []
    
    # Cell comparison
    print(f"\nML cell parameters: {atoms_mlip.cell.cellpar()}")
    print(f"VASP cell parameters: {atoms_vasp.cell.cellpar()}")
    
    cell_diff = np.array(atoms_mlip.cell.cellpar()) - np.array(atoms_vasp.cell.cellpar())
    print(f"Cell differences: {cell_diff}")
    diagnostics['cell_differences'] = cell_diff.tolist()
    
    # Position analysis
    pos_ml = atoms_mlip.get_positions()
    pos_vasp = atoms_vasp.get_positions()
    
    # Raw displacements (no MIC)
    raw_displacements = pos_ml - pos_vasp
    raw_displacement_mags = np.linalg.norm(raw_displacements, axis=1)
    
    # MIC displacements
    vasp_cell = atoms_vasp.get_cell()
    mic_displacements = np.zeros_like(pos_vasp)
    for i in range(len(atoms_vasp)):
        raw_diff = pos_ml[i] - pos_vasp[i]
        frac_diff = np.linalg.solve(vasp_cell.T, raw_diff)
        frac_diff_mic = frac_diff - np.round(frac_diff)
        mic_displacements[i] = np.dot(vasp_cell.T, frac_diff_mic)
    
    mic_displacement_mags = np.linalg.norm(mic_displacements, axis=1)
    
    print(f"\nDisplacement Analysis:")
    print(f"Raw displacements - Max: {np.max(raw_displacement_mags):.3f} Å, Mean: {np.mean(raw_displacement_mags):.3f} Å")
    print(f"MIC displacements - Max: {np.max(mic_displacement_mags):.3f} Å, Mean: {np.mean(mic_displacement_mags):.3f} Å")
    
    # Find atoms with largest displacements
    large_disp_threshold = 1.0  # Å
    large_disp_indices = np.where(mic_displacement_mags > large_disp_threshold)[0]
    
    print(f"\nAtoms with displacements > {large_disp_threshold} Å:")
    for idx in large_disp_indices[:10]:  # Show first 10
        symbol = atoms_vasp[idx].symbol
        disp = mic_displacement_mags[idx]
        pos_ml_atom = pos_ml[idx]
        pos_vasp_atom = pos_vasp[idx]
        print(f"  Atom {idx} ({symbol}): {disp:.3f} Å")
        print(f"    ML pos:   [{pos_ml_atom[0]:.3f}, {pos_ml_atom[1]:.3f}, {pos_ml_atom[2]:.3f}]")
        print(f"    VASP pos: [{pos_vasp_atom[0]:.3f}, {pos_vasp_atom[1]:.3f}, {pos_vasp_atom[2]:.3f}]")
    
    diagnostics.update({
        'raw_disp_max': np.max(raw_displacement_mags),
        'raw_disp_mean': np.mean(raw_displacement_mags),
        'mic_disp_max': np.max(mic_displacement_mags),
        'mic_disp_mean': np.mean(mic_displacement_mags),
        'large_disp_count': len(large_disp_indices),
        'large_disp_atoms': [(int(idx), float(mic_displacement_mags[idx])) for idx in large_disp_indices[:20]]
    })
    
    # Save detailed outputs if directory provided
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        
        # Save diagnostic summary
        import json
        with open(os.path.join(results_dir, f"{name_prefix}_diagnostics.json"), 'w') as f:
            json.dump(diagnostics, f, indent=2)
        
        # Save displacement data
        disp_df = pd.DataFrame({
            'atom_index': range(len(atoms_vasp)),
            'element': atoms_vasp.get_chemical_symbols(),
            'raw_displacement': raw_displacement_mags,
            'mic_displacement': mic_displacement_mags,
            'ml_x': pos_ml[:, 0], 'ml_y': pos_ml[:, 1], 'ml_z': pos_ml[:, 2],
            'vasp_x': pos_vasp[:, 0], 'vasp_y': pos_vasp[:, 1], 'vasp_z': pos_vasp[:, 2]
        })
        disp_df.to_csv(os.path.join(results_dir, f"{name_prefix}_displacements.csv"), index=False)
        
        # Save structures for visualization
        write(os.path.join(results_dir, f"{name_prefix}_ml_structure.xyz"), atoms_mlip)
        write(os.path.join(results_dir, f"{name_prefix}_vasp_structure.xyz"), atoms_vasp)
        
        print(f"\n💾 Diagnostic files saved to: {results_dir}")
    
    return diagnostics

# Add comparison functions here later

if __name__ == "__main__":
    import argparse
    from ase.io import read
    
    parser = argparse.ArgumentParser(description="Run structure comparison diagnostics")
    parser.add_argument("ml_structure", help="Path to ML structure (.xyz or other ASE-readable format)")
    parser.add_argument("vasp_structure", help="Path to VASP structure (.xyz or other ASE-readable format)")
    parser.add_argument("--output_dir", "-o", help="Directory to save diagnostic outputs")
    parser.add_argument("--prefix", "-p", default="diagnosis", help="Prefix for output files")
    
    args = parser.parse_args()
    
    print("Loading structures...")
    try:
        ml_atoms = read(args.ml_structure)
        vasp_atoms = read(args.vasp_structure)
    except Exception as e:
        print(f"Error loading structures: {e}")
        exit(1)
    
    print(f"Running diagnostics between {args.ml_structure} and {args.vasp_structure}")
    
    diagnostics = diagnose_structure_mismatch(
        ml_atoms, vasp_atoms, 
        results_dir=args.output_dir, 
        name_prefix=args.prefix
    )
    
    print("\n=== SUMMARY ===")
    print(f"Max displacement: {diagnostics.get('mic_disp_max', 'N/A'):.3f} Å")
    print(f"Mean displacement: {diagnostics.get('mic_disp_mean', 'N/A'):.3f} Å")
    print(f"Atoms with large displacements: {diagnostics.get('large_disp_count', 0)}")
    print(f"Atom ordering matches: {diagnostics.get('ordering_match', False)}")
    print(f"Composition matches: {diagnostics.get('composition_match', False)}") 