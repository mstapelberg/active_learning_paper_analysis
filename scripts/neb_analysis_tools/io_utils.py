# IO Utilities for VASP parsing, structure reading/writing
import os
import glob
import re
import numpy as np
import pandas as pd
from ase.io import read, write
from ase.io.trajectory import Trajectory

def parse_vacancy_index_from_folder(folder_name: str) -> int | None:
    """
    Parses a 0-based atom index from vacancy folder names like 'Start_Index_NN' or 'End_Index_NN'.
    Example: 'Start_Index_14' -> 13 (converting to 0-based).
    Returns the 0-based index or None if parsing fails.
    """
    match = re.search(r'_Index_(\d+)', folder_name)
    if match:
        index = int(match.group(1))
        # Assuming VASP/folder name indices are 1-based, convert to 0-based for ASE
        return index - 1 
    else:
        # Try to parse from compositions like Cr4Ti12V107WZr_101_to_97 for single index if applicable (e.g. for vacancy creation at a "start" point of a migration)
        # This part might need more specific logic if single indices are embedded differently.
        # For now, prioritize the "_Index_DDD" format.
        print(f"Could not parse a single atom index like '_Index_DDD' from folder name: {folder_name}")
        return None

def parse_neb_indices_from_folder(folder_name: str) -> tuple[int, int] | None:
    """
    Parses 0-based start and end atom indices from NEB folder names like 'Comp_StartIDX_to_EndIDX'.
    Example: 'Cr4Ti12V107WZr_101_to_97' -> (100, 96).
    Returns a tuple (start_index_0_based, end_index_0_based) or None if parsing fails.
    """
    match = re.search(r'_(\d+)_to_(\d+)', folder_name)
    if match:
        start_idx_1_based = int(match.group(1))
        end_idx_1_based = int(match.group(2))
        # Convert to 0-based indices
        return start_idx_1_based - 1, end_idx_1_based - 1
    else:
        print(f"Could not parse NEB start/end indices from folder name: {folder_name}")
        return None

def parse_vasp_static_energy(outcar_path):
    """Parses the final energy from a VASP OUTCAR file for a static calculation."""
    energy = None
    try:
        with open(outcar_path, 'r') as f:
            for line in f:
                if "energy(sigma->0) =" in line: # For static calculations
                    energy = float(line.split()[-1])
        return energy
    except FileNotFoundError:
        print(f"Warning: OUTCAR file not found at {outcar_path}")
        return None
    except Exception as e:
        print(f"Error parsing {outcar_path}: {e}")
        return None

def parse_vasp_relaxation_trajectory(outcar_path):
    """
    Parses a VASP OUTCAR file to extract the trajectory of energies and fmax values
    during a relaxation.
    Returns:
        A list of dictionaries, where each dictionary is {'energy': E, 'fmax': F} for each ionic step.
        Returns an empty list if parsing fails or no trajectory data is found.
    """
    trajectory_data = []
    energies = []
    fmax_values = []
    atoms_frames = [] # To store ASE atoms objects for each step

    try:
        all_frames = read(outcar_path, index=":") # Reads all ionic steps
        
        for atoms_step in all_frames:
            energy = atoms_step.get_potential_energy() # This gets E0 for relaxations
            forces = atoms_step.get_forces()
            fmax = np.sqrt((forces**2).sum(axis=1).max())
            trajectory_data.append({'energy': energy, 'fmax': fmax})
            atoms_frames.append(atoms_step.copy())

    except FileNotFoundError:
        print(f"Warning: OUTCAR file not found at {outcar_path}")
        return [], []
    except Exception as e:
        print(f"ASE parsing failed for {outcar_path}: {e}. Attempting manual parse for energy/fmax.")
        trajectory_data = [] 
        atoms_frames = []
        try:
            with open(outcar_path, 'r') as f:
                lines = f.readlines()

            step_energies = []
            for line in lines:
                if "energy(sigma->0) =" in line:
                    try:
                        step_energies.append(float(line.split()[-1]))
                    except ValueError:
                        pass
            
            if step_energies:
                print(f"Manual parse: Found {len(step_energies)} energy steps for {outcar_path}.")
                for energy_val in step_energies:
                     trajectory_data.append({'energy': energy_val, 'fmax': np.nan}) 

        except Exception as manual_e:
            print(f"Manual parsing also failed for {outcar_path}: {manual_e}")
            return [], []
    return trajectory_data, atoms_frames

def parse_vasp_neb_trajectory(image_outcar_paths):
    """
    Parses VASP OUTCAR files from an NEB calculation.
    Args:
        image_outcar_paths (list of strings): Paths to final OUTCARs for each image (e.g., "00/OUTCAR", "01/OUTCAR").
    Returns:
        neb_results (list of dicts), image_final_energies, image_final_fmax, image_atoms_list
    """
    neb_results = []
    image_final_energies = []
    image_final_fmax = []
    image_atoms_list = [] 

    for i, outcar_path in enumerate(image_outcar_paths):
        try:
            atoms_image = read(outcar_path, index=-1) 
            energy = atoms_image.get_potential_energy()
            forces = atoms_image.get_forces()
            fmax = np.sqrt((forces**2).sum(axis=1).max())
            
            image_final_energies.append(energy)
            image_final_fmax.append(fmax)
            image_atoms_list.append(atoms_image)
            neb_results.append({'image_index': i, 'energy': energy, 'fmax': fmax, 'atoms': atoms_image})

        except FileNotFoundError:
            print(f"Warning: OUTCAR not found for NEB image at {outcar_path}")
            neb_results.append({'image_index': i, 'energy': np.nan, 'fmax': np.nan, 'atoms': None})
            image_final_energies.append(np.nan)
            image_final_fmax.append(np.nan)
            image_atoms_list.append(None)
        except Exception as e:
            print(f"Error parsing NEB OUTCAR {outcar_path}: {e}")
            neb_results.append({'image_index': i, 'energy': np.nan, 'fmax': np.nan, 'atoms': None})
            image_final_energies.append(np.nan)
            image_final_fmax.append(np.nan)
            image_atoms_list.append(None)
    return neb_results, image_final_energies, image_final_fmax, image_atoms_list

def get_initial_vasp_structure(base_path, composition_folder):
    """Reads the initial VASP structure (first frame of a relaxation)."""
    outcar_path = os.path.join(base_path, composition_folder, 'OUTCAR')
    atoms = None
    if os.path.exists(outcar_path):
        try:
            atoms = read(outcar_path, index=0)
        except Exception as e:
            print(f"Error reading initial OUTCAR {outcar_path}: {e}")
    else:
        outcar_0_path = os.path.join(base_path, composition_folder, 'OUTCAR-0')
        if os.path.exists(outcar_0_path):
            try:
                atoms = read(outcar_0_path, index=0)
            except Exception as e:
                print(f"Error reading initial OUTCAR-0 {outcar_0_path}: {e}")
        else:
            print(f"Warning: No OUTCAR or OUTCAR-0 found for initial structure in {os.path.join(base_path, composition_folder)}")
    return atoms

def get_final_vasp_static_structure(base_path, composition_folder):
    """Reads the final VASP structure from a static calculation (last frame)."""
    outcar_path = os.path.join(base_path, composition_folder, 'OUTCAR')
    atoms = None
    if os.path.exists(outcar_path):
        try:
            atoms = read(outcar_path, index=-1)
        except Exception as e:
            print(f"Error reading final static OUTCAR {outcar_path}: {e}")
    else:
        outcar_files = glob.glob(os.path.join(base_path, composition_folder, 'OUTCAR-*'))
        if outcar_files:
            highest_n = -1
            chosen_outcar = None
            for f_path in outcar_files:
                try:
                    n_match = re.search(r'OUTCAR-(\d+)', os.path.basename(f_path))
                    if n_match:
                        n = int(n_match.group(1))
                        if n > highest_n:
                            highest_n = n
                            chosen_outcar = f_path
                except ValueError:
                    continue
            if chosen_outcar:
                try:
                    atoms = read(chosen_outcar, index=-1)
                except Exception as e:
                    print(f"Error reading final static {chosen_outcar}: {e}")
            else:
                 print(f"Warning: No valid OUTCAR-N found in {os.path.join(base_path, composition_folder)}")       
        else:
            print(f"Warning: No OUTCAR or OUTCAR-* found for final static structure in {os.path.join(base_path, composition_folder)}")
    return atoms

def get_vasp_relaxation_trajectory_paths(base_path, composition_folder):
    """Gets paths to VASP OUTCARs that represent a relaxation trajectory."""
    outcar_path = os.path.join(base_path, composition_folder, 'OUTCAR')
    if os.path.exists(outcar_path):
        return outcar_path
    else:
        outcar_0_path = os.path.join(base_path, composition_folder, 'OUTCAR-0')
        if os.path.exists(outcar_0_path):
            return outcar_0_path
    print(f"Warning: No OUTCAR or OUTCAR-0 for VASP relaxation trajectory in {os.path.join(base_path, composition_folder)}")
    return None

def get_vasp_vacancy_structure_and_path(base_path, composition_folder, target_subdir_name_part=None):
    """
    Reads a VASP vacancy structure from subdirectories like Start_Index_XX or End_Index_XX.
    Returns the atoms object and the path to the OUTCAR used.
    """
    comp_path = os.path.join(base_path, composition_folder)
    if not os.path.isdir(comp_path):
        print(f"Warning: Composition folder {comp_path} not found for VASP vacancy.")
        return None, None

    possible_subdirs = [d for d in os.listdir(comp_path) if os.path.isdir(os.path.join(comp_path, d))]
    
    candidate_outcars = []
    # Prioritize subdirs matching target_subdir_name_part if provided
    priority_subdirs = []
    other_subdirs = []

    if target_subdir_name_part:
        for subdir in possible_subdirs:
            if target_subdir_name_part in subdir:
                priority_subdirs.append(subdir)
            else:
                other_subdirs.append(subdir)
        # Process priority subdirs first, then others if no match in priority
        ordered_subdirs = priority_subdirs + other_subdirs
    else:
        ordered_subdirs = possible_subdirs

    for subdir in ordered_subdirs:
        outcar_path = os.path.join(comp_path, subdir, 'OUTCAR')
        if os.path.exists(outcar_path):
            candidate_outcars.append(outcar_path)
        else:
            outcar_n_files = glob.glob(os.path.join(comp_path, subdir, 'OUTCAR-*'))
            if outcar_n_files:
                highest_n = -1
                chosen_outcar_n = None
                for f_path in outcar_n_files:
                    try:
                        n_match = re.search(r'OUTCAR-(\d+)', os.path.basename(f_path))
                        if n_match:
                            n = int(n_match.group(1))
                            if n > highest_n:
                                highest_n = n
                                chosen_outcar_n = f_path
                    except ValueError:
                        continue
                if chosen_outcar_n:
                    candidate_outcars.append(chosen_outcar_n)
    
    if not candidate_outcars:
        print(f"Warning: No OUTCAR files found in subdirectories of {comp_path} (matching '{target_subdir_name_part}' if specified).")
        return None, None

    chosen_outcar_for_structure = candidate_outcars[0] 
    print(f"Using VASP OUTCAR: {chosen_outcar_for_structure} for vacancy structure/trajectory.")
    try:
        atoms = read(chosen_outcar_for_structure, index=-1) 
        return atoms, chosen_outcar_for_structure 
    except Exception as e:
        print(f"Error reading VASP vacancy OUTCAR {chosen_outcar_for_structure}: {e}")
        return None, None

def get_vasp_neb_endpoints_and_path_data(base_vasp_neb_path, neb_system_folder):
    """ 
    Loads VASP NEB start and end structures, and the energy/fmax for all images along the path.
    Assumes base_vasp_neb_path points to a directory like "mrs_neb_static" or "mrs_neb".
    neb_system_folder is like "Cr4Ti12V107WZr_101_to_97".
    Returns: start_atoms, end_atoms, vasp_neb_energies, vasp_neb_fmax, vasp_image_atoms
    """
    system_path = os.path.join(base_vasp_neb_path, neb_system_folder)
    if not os.path.isdir(system_path):
        print(f"Warning: VASP NEB system folder not found: {system_path}")
        return None, None, [], [], []

    image_dirs = sorted([d for d in os.listdir(system_path) if os.path.isdir(os.path.join(system_path, d)) and d.isdigit()])
    if not image_dirs:
        print(f"Warning: No image directories (00, 01, ...) found in {system_path}")
        return None, None, [], [], []

    start_atoms_path = os.path.join(system_path, image_dirs[0], 'OUTCAR')
    end_atoms_path = os.path.join(system_path, image_dirs[-1], 'OUTCAR')

    start_atoms, end_atoms = None, None
    try:
        start_atoms = read(start_atoms_path, index=-1) 
    except Exception as e:
        print(f"Error reading VASP NEB start image {start_atoms_path}: {e}")
    try:
        end_atoms = read(end_atoms_path, index=-1) 
    except Exception as e:
        print(f"Error reading VASP NEB end image {end_atoms_path}: {e}")

    all_image_outcar_paths = [os.path.join(system_path, img_dir, 'OUTCAR') for img_dir in image_dirs]
    
    _, vasp_neb_energies, vasp_neb_fmax, vasp_image_atoms = parse_vasp_neb_trajectory(all_image_outcar_paths)

    return start_atoms, end_atoms, vasp_neb_energies, vasp_neb_fmax, vasp_image_atoms

# Add parsing functions here later 