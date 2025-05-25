# Functions for creating vacancies and NEB endpoints
import numpy as np
from ase.io import write, read
import os
import re # For parsing folder names if not done upstream

def create_vacancy_from_perfect(perfect_atoms_object, vacancy_atom_index_0_based, results_dir, name_prefix="vacancy"):
    """
    Creates a single vacancy in a copy of the perfect_atoms_object by removing an atom at the specified 0-based index.
    Saves the initial vacancy structure.
    Returns the vacancy_atoms object or None if index is invalid.
    """
    if not perfect_atoms_object:
        print("Error: Perfect atoms object is None. Cannot create vacancy.")
        return None
    if not (0 <= vacancy_atom_index_0_based < len(perfect_atoms_object)):
        print(f"Error: Invalid atom index {vacancy_atom_index_0_based} for structure with {len(perfect_atoms_object)} atoms.")
        return None

    vacancy_atoms = perfect_atoms_object.copy()
    removed_atom_symbol = vacancy_atoms[vacancy_atom_index_0_based].symbol
    print(f"Creating vacancy by removing atom index {vacancy_atom_index_0_based} (Symbol: {removed_atom_symbol}).")
    del vacancy_atoms[vacancy_atom_index_0_based]

    os.makedirs(results_dir, exist_ok=True)
    initial_vacancy_file = os.path.join(results_dir, f"{name_prefix}_initial_idx{vacancy_atom_index_0_based}.xyz")
    write(initial_vacancy_file, vacancy_atoms)
    print(f"Saved initial vacancy structure to {initial_vacancy_file}")
    
    return vacancy_atoms

def create_allegro_neb_endpoints(allegro_perfect_relaxed_atoms, start_atom_idx_0_based, end_atom_idx_0_based, results_dir, system_name="neb_system"):
    """
    Creates start and end point structures for an NEB calculation using the exact logic
    from the original create_neb_endpoints function to ensure proper atom ordering.
    
    This creates:
    1. Start structure: perfect structure with start_atom_idx removed
    2. End structure: perfect structure with end_atom moved to start_atom position, then start_atom removed
    
    This ensures consistent atom ordering for NEB interpolation.

    Args:
        allegro_perfect_relaxed_atoms (ase.Atoms): The Allegro-relaxed perfect supercell.
        start_atom_idx_0_based (int): 0-based index of the atom to remove for the NEB start point.
        end_atom_idx_0_based (int): 0-based index of the atom to move to start position for the NEB end point.
        results_dir (str): Directory to save the endpoint structures.
        system_name (str): Name for filenaming.

    Returns:
        tuple: (start_endpoint_atoms, end_endpoint_atoms) or (None, None) if an error occurs.
    """
    if not allegro_perfect_relaxed_atoms:
        print("Error: Allegro perfect relaxed atoms object is None.")
        return None, None
    
    num_atoms_perfect = len(allegro_perfect_relaxed_atoms)
    if not (0 <= start_atom_idx_0_based < num_atoms_perfect):
        print(f"Error: Invalid start_atom_idx_0_based {start_atom_idx_0_based} for structure with {num_atoms_perfect} atoms.")
        return None, None
    if not (0 <= end_atom_idx_0_based < num_atoms_perfect):
        print(f"Error: Invalid end_atom_idx_0_based {end_atom_idx_0_based} for structure with {num_atoms_perfect} atoms.")
        return None, None
    if start_atom_idx_0_based == end_atom_idx_0_based:
        print(f"Error: start_atom_idx_0_based ({start_atom_idx_0_based}) and end_atom_idx_0_based ({end_atom_idx_0_based}) are the same. NEB endpoints must be different.")
        return None, None

    # Create deep copies of the perfect structure
    start_endpoint = allegro_perfect_relaxed_atoms.copy()
    end_endpoint = allegro_perfect_relaxed_atoms.copy()
    
    # Get symbols for logging
    start_atom_symbol = start_endpoint[start_atom_idx_0_based].symbol
    end_atom_symbol = end_endpoint[end_atom_idx_0_based].symbol
    
    print(f"Creating NEB endpoints: moving atom {end_atom_idx_0_based} ({end_atom_symbol}) to position of atom {start_atom_idx_0_based} ({start_atom_symbol})")
    
    # Set the position of the end_index atom to the position of the start_index atom in the end structure
    end_endpoint.positions[end_atom_idx_0_based] = end_endpoint.positions[start_atom_idx_0_based].copy()

    # Remove the atom at start_index from both structures
    del start_endpoint[start_atom_idx_0_based]
    del end_endpoint[start_atom_idx_0_based]
    
    os.makedirs(results_dir, exist_ok=True)
    start_file = os.path.join(results_dir, f"{system_name}_allegro_start_idx{start_atom_idx_0_based}.xyz")
    end_file = os.path.join(results_dir, f"{system_name}_allegro_end_idx{end_atom_idx_0_based}_to_idx{start_atom_idx_0_based}.xyz")
    write(start_file, start_endpoint)
    write(end_file, end_endpoint)
    print(f"Saved Allegro NEB start endpoint to {start_file}")
    print(f"Saved Allegro NEB end endpoint to {end_file}")

    return start_endpoint, end_endpoint

# Add structure modification functions here later 