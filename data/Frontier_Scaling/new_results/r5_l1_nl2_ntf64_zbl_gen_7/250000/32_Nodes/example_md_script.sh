#!/bin/bash
#SBATCH -N 32
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --gpu-bind=none
#SBATCH --cpus-per-task=7
#SBATCH -A mat293
#SBATCH -t 00:15:00
#SBATCH -q debug


#PATH_TO_BUILD="/lustre/orion/mat293/scratch/myless/25_03_10_AllegroConda_torch24"
PATH_TO_BUILD="/lustre/orion/mat293/scratch/myless/25_05_06_AllegroConda_torch27"

#eval "$(micromamba shell hook --shell bash)"
#micromamba activate
module load miniforge3/23.11.0-0
source activate ${PATH_TO_BUILD}/allegro_torch27

module load PrgEnv-amd
module load amd/6.3.1 rocm/6.3.1 gcc-native-mixed/14.2
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}

export MPICH_GPU_SUPPORT_ENABLED=1

#srun ${PATH_TO_BUILD}/lammps/build/lmp -sf kk -k on g 8 -pk kokkos newton on neigh full "$@"

srun ${PATH_TO_BUILD}/lammps/build/lmp -sf kk -k on g 8 -pk kokkos newton on neigh full -in new_input.lammps


