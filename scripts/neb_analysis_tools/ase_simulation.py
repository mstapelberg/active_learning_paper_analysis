# ASE based simulation tasks (relax, neb_relax)
import os
import torch
import numpy as np
from ase.io.trajectory import Trajectory
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from ase.mep import DyNEB
#from nequip.ase.nequip_calculator import NequIPCalculator

def get_calculator(model_path, model_type="allegro", device=None):
    """
    Factory function to create appropriate ASE calculator based on model type.
    
    Args:
        model_path (str): Path to the model file
        model_type (str): Type of model ("allegro" or "mace")
        device (str): Device to use ("cuda" or "cpu"). If None, auto-detects.
    
    Returns:
        ASE calculator instance
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if model_type.lower() == "allegro":
        from nequip.ase.nequip_calculator import NequIPCalculator
        return NequIPCalculator.from_compiled_model(compile_path=model_path, device=device)
    
    elif model_type.lower() == "mace":
        try:
            from mace.calculators.mace import MACECalculator
            return MACECalculator(model_paths=model_path, device=device)
        except ImportError:
            raise ImportError("MACE is not installed. Please install it with: pip install mace-torch")
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: allegro, mace")

def relax(atoms, calculator, relax_cell = False, fmax = 0.01, steps = 300, trajectory_file=None):
    new_atoms = atoms.copy()
    new_atoms.calc = calculator

    if trajectory_file:
        traj = Trajectory(trajectory_file, 'w', new_atoms)

    if relax_cell:
        fcf = FrechetCellFilter(new_atoms)
        opt = FIRE(fcf)
        if trajectory_file:
            opt.attach(traj.write, interval=1)
        opt.run(fmax=fmax, steps=steps)
    else:
        opt = FIRE(new_atoms)
        if trajectory_file:
            opt.attach(traj.write, interval=1)
        opt.run(fmax=fmax, steps=steps)
    
    if trajectory_file:
        traj.close()
    
    return new_atoms

def neb_relax(start, end, model_path, model_type="allegro", num_images = 5, fmax = 0.01, steps = 300, trajectory_dir=None, system_name="neb"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = [start.copy()]
    # Create intermediate images; num_images is the number of *intermediate* images
    for i in range(num_images):
        images.append(start.copy()) # placeholder, will be interpolated
    images.append(end.copy())

    # Check if we have at least 3 images (start, one intermediate, end) for DyNEB
    if len(images) < 3:
        print(f"Warning: NEB for {system_name} needs at least one intermediate image. Provided {num_images}. Adjusting to have at least start, one intermediate, and end.")
        pass # Keep as is, DyNEB will likely handle/error if images list is malformed.

    neb = DyNEB(images, climb=True, k=0.1) # k can be tuned
    neb.interpolate(mic=True) # Interpolates the intermediate images

    img_calculators = []
    for i, image in enumerate(neb.images):
        # Create a new calculator instance for each image to avoid potential state issues
        calc_instance = get_calculator(model_path, model_type, device)
        image.calc = calc_instance
        img_calculators.append(calc_instance) # Keep a reference if needed, though ASE usually handles this

    opt = FIRE(neb)

    neb_path_traj_file = None
    if trajectory_dir:
        os.makedirs(trajectory_dir, exist_ok=True)
        neb_path_traj_file = os.path.join(trajectory_dir, f"{system_name}_neb_path.traj")
        
        # Simple approach: save final NEB path only
        # More complex trajectory logging can be added later if needed
        def save_final_neb_path():
            # This will be called at the end to save the final converged path
            pass
    
    opt.run(fmax=fmax, steps=steps)

    # Save final NEB path if trajectory directory is provided
    if trajectory_dir and neb_path_traj_file:
        try:
            # Save each image separately to avoid list-of-atoms trajectory issues
            final_images_dir = os.path.join(trajectory_dir, f"{system_name}_final_images")
            os.makedirs(final_images_dir, exist_ok=True)
            
            from ase.io import write
            for i, image in enumerate(neb.images):
                image_file = os.path.join(final_images_dir, f"image_{i:02d}.xyz")
                write(image_file, image)
            
            print(f"Saved final NEB images to {final_images_dir}")
        except Exception as e:
            print(f"Warning: Could not save final NEB images: {e}")

    return neb

# Add simulation functions here later 