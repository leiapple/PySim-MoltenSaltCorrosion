from run_simulation import *

set_seed(42)

# The other parameters are set to the default values in the SimulationConfig dataclass
config = SimulationConfig(
    M_composition={"Fe": 0.8, "Ni": 0.0, "Cr": 0.2},  # Fe80Cr20 alloy composition
    alloy_surface=fcc111,
    temperature_K=1400,  # K
    initial_density=1.86,  # g/cm^3
    npt_num_steps=1000000,  # number of MD steps
    salt_cations=[["Na"], [216]],  #
    salt_anions=[["F"], [216]],  # salt composition
    impurity="none",
)

alloy = prepare_alloy(c=config)
salt = prepare_salt(alloy, config)
salt = add_impurities(salt, config)
salt = run_npt_salt(salt, config)
salt = add_oxygen_top(salt, config)

system = combine_alloy_salt(alloy, salt, config)
system = npt_equilibration(system, config)
system = nvt_simulation(system, config)
