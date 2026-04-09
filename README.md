# MD Simulation of Metal Alloys in Contact with Molten Salts and Impurities

## Overview

This repository contains code to perform molecular dynamics (MD) simulations of metal alloys in contact with molten salts using:

- [ASE](https://ase-lib.org/)
- The UMA-s-1p1 universal machine-learning interatomic potential (MLIP) from FAIR-Chem [FAIR-Chem](https://github.com/facebookresearch/fairchem)

The simulation workflow includes alloy preparation, molten salt generation, impurity insertion, interface construction, and NPT/NVT MD simulations:

**Alloy → Salt → Impurities → Equilibration → Interface → NPT → NVT**

This repository contains the source code for the simulations reported in [TODO REF](doi.org/TODO).

Other information:
- Folder `Scripts` contains the simulation parameters used in the paper.
- Folder `Postprocessing` contains the code used to extract information from the ase trajectory files.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Default full simulation

```bash
python run_simulation.py
```

### 2. Custom simulation (example)

Should be run from the folder containing the `run_simulation.py` file.

```python
from run_simulation import *

set_seed(42)

# The other parameters are set to the default values in the SimulationConfig dataclass
config = SimulationConfig(
    temperature_K=1400,  # K
    initial_density=1.5,  # g/cm^3
    npt_num_steps=1000000,  # number of MD steps
)

alloy = prepare_alloy(c=config)
salt = prepare_salt(alloy, config)
salt = add_impurities(salt, config)
salt = run_npt_salt(salt, config)
salt = add_oxygen_top(salt, config)

system = combine_alloy_salt(alloy, salt, config)
system = npt_equilibration(system, config)
system = nvt_simulation(system, config)
```

### 3. Salt-only simulation (NPT)

```python
from run_simulation import *

set_seed(42)

config = SimulationConfig()  # Default parameters

salt = prepare_salt(alloy=None, c=config)
salt = npt_equilibration(system_in=salt, c=config)
```

## Configuration

All parameters are controlled via `SimulationConfig`, including:

- Alloy composition (`M_composition`)
- Salt type (`salt_anions`, `salt_cations`) and initial density guess (`initial_density`)
- Impurity type (`impurity`: "oxygen", "water", "none")
- MD parameters (`temperature_K`, `npt_num_steps`, `nvt_num_steps`)
- Output and log filenames (`npt_pure_alloy_logfile`, `npt_pure_salt_logfile`, `main_npt_logfile`, `main_nvt_logfile`, `main_npt_trajfile`, `main_nvt_trajfile`)
- Calculator parameters (`model_name`, `task_name`, `device`)

## Outputs

- `*.traj` files: MD trajectories (ASE format)
- `log.*` files: thermodynamic data (energy, temperature, stress)
- `stdout`: step-wise diagnostics (pressure, temperature, stress)

## Notes

- GPU recommended. An NVT simulation of the full salt+alloy system with timestep 1 fs for total simulation of 500 ps takes ~40 hours on a single H100 GPU
- Water simulations use a smaller timestep (0.2 fs vs. 1.0 fs by default)
- Simulations are reproducible given a fixed random seed (`set_seed`) and identical software/hardware environment

## License

The code in this repository is licensed under the MIT license.
