# ============================================================
# Test run_simulation.py with minimal configuration so it can be run on a laptop
# ============================================================
import sys

sys.path.append("..")
from run_simulation import (
    SimulationConfig,
    add_impurities,
    add_oxygen_top,
    combine_alloy_salt,
    npt_equilibration,
    nvt_simulation,
    prepare_alloy,
    prepare_salt,
    run_npt_salt,
)

# Load the simulation config and override some defaults
config = SimulationConfig(
    fmax_fe_bulk=0.1,
    fmax_alloy_bulk=5000,
    npt_num_steps_for_alloy=1,
    npt_num_steps_for_salt=1,
    npt_num_steps=1,
    nvt_num_steps=5,
    trajectory_write_interval=1,
    trajectory_print_interval=1,
    trajectory_log_interval=1,
    device="cpu",
)
# Prepare the metal alloy
alloy = prepare_alloy(c=config)
# Prepare the molten salt
salt = prepare_salt(alloy=alloy, c=config)
# Add impurities
salt = add_impurities(salt=salt, c=config)
# Equilibrate the salt
salt = run_npt_salt(salt=salt, c=config)
# Add additional oxygen on the top layer of simulation cell
salt = add_oxygen_top(salt=salt, c=config)
# Stick the salt on top of the alloy
combined = combine_alloy_salt(alloy=alloy, salt=salt, c=config)
# NPT equilibration
system = npt_equilibration(system_in=combined, c=config)
# NVT simulation
system = nvt_simulation(system=system, c=config)
