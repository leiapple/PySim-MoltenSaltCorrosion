#!/bin/bash

#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=8
#SBATCH --time=0-1:00:00
#SBATCH --error=slurm_logs/stderr_%x_%j.err
#SBATCH --output=slurm_logs/stdout_%x_%j.out

# Stop on first error
set -e
# Load modules
module purge
module load 2024 VASP6/6.5.1-foss-2024a

export OMP_NUM_THREADS=8

# Activate Python environment
source /home/amd20/PythonVenvs/msr_corr/bin/activate

# Run Python script with all arguments
python -u benchmark_vasp.py --vasp_command "mpirun --bind-to hwthread --map-by ppr:12:node:PE=8:spawn vasp_gam" --ncore 8 --species_idx 0 1 --kernel_truncation --overwrite
