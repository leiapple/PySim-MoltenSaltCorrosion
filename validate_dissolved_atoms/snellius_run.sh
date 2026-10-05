#!/bin/bash

#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --time=0-5:00:00
#SBATCH --error=slurm_logs/stderr_%x_%j.err
#SBATCH --output=slurm_logs/stdout_%x_%j.out

# Stop on first error
set -e
# Load modules
module purge
module load 2024 VASP6/6.5.1-foss-2024a

export OMP_NUM_THREADS=1

# Activate Python environment
source /home/amd20/PythonVenvs/msr_corr/bin/activate

# Run Python script with all arguments
python -u benchmark_vasp.py --vasp_command "mpirun --bind-to hwthread --map-by ppr:24:node:PE=1:spawn vasp_gam" --ncore 1 --overwrite --kernel_truncation --species $@
