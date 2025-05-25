# Plotting utilities
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_relaxation_trajectories(trajectory_comparison_data, save_path_prefix="relaxation_comparison"):
    """
    Plots energy and fmax trajectories from Allegro and VASP relaxations.
    Saves the plot to a file.

    Args:
        trajectory_comparison_data (dict): Data from comparison.extract_relaxation_trajectory_data.
        save_path_prefix (str): Prefix for the output plot filename.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    if trajectory_comparison_data.get('allegro_energies'):
        plt.plot(trajectory_comparison_data['allegro_energies'], label='Allegro Energy', marker='o', markersize=5, linestyle='-')
    if trajectory_comparison_data.get('vasp_energies'):
        plt.plot(trajectory_comparison_data['vasp_energies'], label='VASP Energy', marker='x', markersize=5, linestyle='--')
    plt.xlabel('Step')
    plt.ylabel('Energy (eV)')
    plt.legend()
    plt.title('Energy Comparison during Relaxation')

    plt.subplot(1, 2, 2)
    if trajectory_comparison_data.get('allegro_fmax'):
        plt.plot(trajectory_comparison_data['allegro_fmax'], label='Allegro Fmax', marker='o', markersize=5, linestyle='-')
    if trajectory_comparison_data.get('vasp_fmax'):
        plt.plot(trajectory_comparison_data['vasp_fmax'], label='VASP Fmax', marker='x', markersize=5, linestyle='--')
    plt.xlabel('Step')
    plt.ylabel('Fmax (eV/Å)')
    plt.yscale('log') 
    plt.legend()
    plt.title('Fmax Comparison during Relaxation')
    
    plt.tight_layout()
    plot_filename = f"{save_path_prefix}.png"
    plt.savefig(plot_filename)
    print(f"Saved trajectory comparison plot to {plot_filename}")
    plt.close()

def plot_neb_comparison(neb_comparison_data, results_dir, system_name):
    """
    Plots converged NEB energy profiles and Fmax, and Allegro NEB convergence.

    Args:
        neb_comparison_data (dict): Data from comparison.extract_neb_comparison_data.
        results_dir (str): Directory to save plots.
        system_name (str): Name of the NEB system for file naming.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Plotting converged energy profiles
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    if neb_comparison_data.get('allegro_energies_norm') and len(neb_comparison_data['allegro_energies_norm']) > 0:
        allegro_barrier_fwd = neb_comparison_data.get('allegro_barrier_fwd', np.nan)
        plt.plot(neb_comparison_data['allegro_energies_norm'], label=f'Allegro (Barrier: {allegro_barrier_fwd:.3f} eV)', marker='o')
    if neb_comparison_data.get('vasp_energies_norm') and len(neb_comparison_data['vasp_energies_norm']) > 0 and not np.all(np.isnan(neb_comparison_data['vasp_energies_norm'])):
        vasp_barrier_fwd = neb_comparison_data.get('vasp_barrier_fwd', np.nan)
        plt.plot(neb_comparison_data['vasp_energies_norm'], label=f'VASP (Barrier: {vasp_barrier_fwd:.3f} eV)', marker='x')
    plt.xlabel('Image Index')
    plt.ylabel('Relative Energy (eV)')
    plt.legend()
    plt.title(f'Converged NEB Energy Profile: {system_name}')

    # Plotting Fmax per image (converged path)
    plt.subplot(1, 2, 2)
    if neb_comparison_data.get('allegro_fmax_final') and len(neb_comparison_data['allegro_fmax_final']) > 0:
        plt.plot(neb_comparison_data['allegro_fmax_final'], label='Allegro Fmax', marker='o')
    if neb_comparison_data.get('vasp_fmax_final') and len(neb_comparison_data['vasp_fmax_final']) > 0 and not np.all(np.isnan(neb_comparison_data['vasp_fmax_final'])):
        plt.plot(neb_comparison_data['vasp_fmax_final'], label='VASP Fmax', marker='x')
    plt.xlabel('Image Index')
    plt.ylabel('Fmax (eV/Å)')
    plt.yscale('log')
    plt.legend()
    plt.title(f'Converged NEB Fmax: {system_name}')

    plt.tight_layout()
    plot_filename = os.path.join(results_dir, f"neb_converged_comparison_{system_name}.png")
    plt.savefig(plot_filename)
    print(f"Saved NEB converged path comparison plot to {plot_filename}")
    plt.close()

    # Plotting Allegro NEB convergence (Fmax of the band over optimization steps)
    if neb_comparison_data.get('allegro_band_fmax_over_time') and len(neb_comparison_data['allegro_band_fmax_over_time']) > 0:
        plt.figure()
        plt.plot(neb_comparison_data['allegro_band_fmax_over_time'], marker='o')
        plt.xlabel("Allegro NEB Optimization Step")
        plt.ylabel("Max Fmax in Band (eV/Å)")
        plt.yscale('log')
        plt.title(f"Allegro NEB Convergence (Max Fmax): {system_name}")
        fmax_plot_filename = os.path.join(results_dir, f"allegro_neb_convergence_fmax_{system_name}.png")
        plt.savefig(fmax_plot_filename)
        print(f"Saved Allegro NEB Fmax convergence plot to {fmax_plot_filename}")
        plt.close()

# Add plotting functions here later 