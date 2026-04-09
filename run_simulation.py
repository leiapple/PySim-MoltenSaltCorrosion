import os
import random
from dataclasses import dataclass, field
from random import shuffle
from typing import Dict, List, Tuple

import numpy as np
from ase import Atom, Atoms, units
from ase.build import bulk, fcc100, fcc110, fcc111, molecule
from ase.constraints import FixAtoms
from ase.data import atomic_masses, atomic_numbers
from ase.filters import StrainFilter
from ase.geometry import get_distances
from ase.io import Trajectory, write
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS, LBFGS
from fairchem.core import FAIRChemCalculator, pretrained_mlip
from scipy.spatial.distance import cdist


# ============================================================
# Global RNG to ensure reproducibility
# ============================================================
def set_seed(rng_seed=42):
    random.seed(rng_seed)
    np.random.seed(rng_seed)


# ============================================================
# Configuration dataclass
# ============================================================


@dataclass
class SimulationConfig:
    # ---------------------------
    # System parameters with meaningful defaults
    # ---------------------------
    M_composition: Dict[str, float] = field(
        default_factory=lambda: {
            "Fe": 0.46,
            "Ni": 0.3,
            "Cr": 0.24,
        }  # Fe46Ni30Cr24 = Alloy 800H
    )

    # Whether to apply periodic boundary conditions along the z-direction
    pbc_z_direction: bool = True
    # How many layers of the alloy to fix from the bottom
    n_fix_layers_alloy: int = 2  # Only valid if pbc_z_direction is True

    fcc_bulk_initial_a: float = 3.5  # Å, initial lattice constant for the FCC Fe bulk
    fmax_fe_bulk: float = 0.001  # Relax until max force < 0.001 eV/Å
    fmax_alloy_bulk: float = 0.01  # Relax until max force < 0.01 eV/Å

    # Surface to use for the alloy, can be fcc100, fcc110, fcc111
    alloy_surface: callable = fcc100
    alloy_size: Tuple[int, int, int] = field(init=False)

    # Cation/anion symbols and respective number of atoms
    salt_cations: List = field(default_factory=lambda: [["Na"], [216]])
    salt_anions: List = field(default_factory=lambda: [["Cl"], [216]])

    impurity: str = "oxygen"  # "oxygen", "water", "none"
    n_O2: int = field(default=40)  # Only applies if impurity is "oxygen"
    # Number of O atoms add to the top layer to prevent O2 in the salt adsorption on the top of simulation cell
    n_O_top: int = field(default=50)  # Only applies if impurity is "oxygen"
    n_H2O: int = field(default=40)  # Only applies if impurity is "water"
    # Height of the oxygen layer on top of the simulation cell
    oxygen_top_height: float = 2.5  # Å, only applies if impurity is "oxygen"

    salt_initial_a: float = (
        2  # Å, initial lattice constant for the rocksalt crystal structure
    )
    # 1.86 g/cm³ for NaF, or 1.38 g/cm³ for NaCl at 1400 K
    initial_density: float = 1.38  # g/cm³
    # Whether to construct the molten salt by random placement or from an initial rocksalt crystal structure
    initial_salt_str: str = "rocksalt"  # "rocksalt" or "random"
    # Whether to remove positions from the rocksalt structure randomly or just cut off higher indices
    random_removal: bool = True  # Only applies if initial_salt_str is "rocksalt"

    # Minimum allowed distance between any two atoms when the molten salt is constructed by random placement
    min_distance: float = 1.6  # Å
    # Maximum number of attempts to find a suitable position for an atom while fulfilling the minimum distance constraint
    max_attempts: int = int(1e6)  # Only applies if initial_salt_str is "random"

    # Distance between the atoms of the alloy surface and the lowest possible salt atoms
    alloy_salt_spacing: float = 1.0  # Å
    # Prevent the alloy atom get to the top through PBC
    alloy_shift: float = 0.5  # Å

    # ---------------------------
    # Simulation parameters
    # ---------------------------
    temperature_K: float = 1400

    taut_fs: float = 100
    taup_fs: float = 1000

    timestep_fs: float = field(init=False)

    pressure_bar: float = 1.01325
    compressibility_per_bar: float = 4.0e-5

    friction_per_fs: float = 0.001

    trajectory_write_interval: int = 100
    trajectory_print_interval: int = 100
    trajectory_log_interval: int = 100

    npt_num_steps_for_alloy: int = 10000
    npt_num_steps_for_salt: int = 10000
    npt_num_steps: int = 2000
    nvt_num_steps: int = 1000000

    # ---------------------------
    # Log and output files
    # ---------------------------
    npt_pure_alloy_logfile: str = "log.npt_pure_alloy"
    npt_alloy_trajfile: str = "alloy_npt.traj"
    npt_salt_trajfile: str = "salt_npt.traj"
    npt_pure_salt_logfile: str = "log.npt_pure_salt"
    main_npt_logfile: str = "log.main_npt"
    main_nvt_logfile: str = "log.main_nvt"
    main_npt_trajfile: str = "main_npt.traj"
    main_nvt_trajfile: str = "main_nvt.traj"

    # ---------------------------
    # Calculator parameters
    # ---------------------------
    model_name: str = "uma-s-1p1"  # "uma-s-1p1" or "uma-s-1p2"
    task_name: str = "oc20"  # "oc20", "oc22" or "oc25"
    device: str = "cuda"  # "cuda" or "cpu"
    _calculator: FAIRChemCalculator = field(default=None, init=False, repr=False)

    # ============================================================
    # Post-init logic (derived parameters)
    # ============================================================

    def __post_init__(self):
        # Validate composition
        if not np.isclose(sum(self.M_composition.values()), 1.0):
            raise ValueError("M_composition must sum to 1")

        # Define the size of the alloy slab times the unit cell: fcc100 - (8, 8, 14), fcc110 - (6, 8, 14), fcc111 - (8, 10, 12)
        # The unit cell is chosen to make the simulation size similar: 20x20x24 Å
        if self.alloy_surface == fcc100:
            self.alloy_size = (8, 8, 14)
        elif self.alloy_surface == fcc110:
            self.alloy_size = (7, 8, 16)
        elif self.alloy_surface == fcc111:
            self.alloy_size = (8, 10, 12)
        else:
            raise ValueError("Unsupported alloy surface")

        # Timestep logic
        if self.impurity == "water":
            self.timestep_fs = 0.2  # fs
        else:
            self.timestep_fs = 1.0  # fs

    @property
    def calculator(self) -> FAIRChemCalculator:
        """Lazy-load the calculator if not already initialized."""
        if self._calculator is None:
            predictor = pretrained_mlip.get_predict_unit(
                self.model_name, device=self.device
            )
            self._calculator = FAIRChemCalculator(predictor, task_name=self.task_name)
        return self._calculator


# ============================================================
# Prepare the metal alloy
# ============================================================


def prepare_alloy(
    c: SimulationConfig,
):
    """Construct the metal alloy and optimize the lattice constant for strain / forces.

    Args:
        c (SimulationConfig): Simulation configuration.

    Returns:
        Atoms: ASE atoms object of the equilibrated alloy.
    """
    # Generate an FCC bulk and optimize the lattice constant for strain / forces
    FCC_bulk = bulk("Fe", "fcc", a=c.fcc_bulk_initial_a, cubic=True)

    # Optimize the bulk for strain / forces
    FCC_bulk.calc = c.calculator
    opt = BFGS(
        StrainFilter(FCC_bulk, mask=[1, 1, 1, 0, 0, 0])
    )  # Diagonal strain optimization only
    opt.run(fmax=c.fmax_fe_bulk)
    cell = FCC_bulk.get_cell()
    a0 = (cell[0][0] + cell[1][1] + cell[2][2]) / 3  # average over all three

    # Generate a new bulk with the optimized lattice constant
    alloy = c.alloy_surface("Fe", a=a0, size=c.alloy_size, orthogonal=True, vacuum=0)
    # restore the normal lattice
    if c.alloy_surface == fcc100:
        alloy.cell[2][2] += np.sqrt(1) * a0 / 2
    elif c.alloy_surface == fcc110:
        alloy.cell[2][2] += np.sqrt(2) * a0 / 4
    elif c.alloy_surface == fcc111:
        alloy.cell[2][2] += np.sqrt(3) * a0 / 3

    alloy.set_pbc([True, True, True])

    # Fill the correct symbols for the alloy into the FCC lattice
    n_atoms = len(alloy)
    elements = list(c.M_composition.keys())
    ratios = list(c.M_composition.values())
    counts = [int(round(n_atoms * r)) for r in ratios[:-1]]
    counts.append(n_atoms - sum(counts))  # force exact total
    symbols = np.repeat(elements, counts)
    shuffle(symbols)

    # Insert the symbols into the structure
    alloy.symbols = symbols

    # Optimize the bulk for strain / forces
    alloy.calc = c.calculator
    opt1 = LBFGS(StrainFilter(alloy, mask=[1, 1, 1, 0, 0, 0]))
    opt1.run(fmax=c.fmax_alloy_bulk)

    # Run NPT simulation for only the alloy
    dyn_npt_alloy = NPTBerendsen(
        alloy,
        timestep=c.timestep_fs * units.fs,
        temperature_K=c.temperature_K,
        taut=c.taut_fs * units.fs,
        pressure_au=c.pressure_bar * units.bar,
        taup=c.taup_fs * units.fs,
        compressibility_au=c.compressibility_per_bar / units.bar,
    )

    # Create logger that prints pressure
    logger_npt_alloy = MDLogger(
        dyn_npt_alloy,
        alloy,
        c.npt_pure_alloy_logfile,
        mode="w",
        header=True,
        stress=True,
        peratom=False,
    )

    MaxwellBoltzmannDistribution(alloy, temperature_K=c.temperature_K)
    trajectory_npt_alloy = Trajectory(c.npt_alloy_trajfile, "w", alloy)
    dyn_npt_alloy.attach(
        trajectory_npt_alloy.write, interval=c.trajectory_write_interval
    )
    dyn_npt_alloy.attach(logger_npt_alloy, interval=c.trajectory_log_interval)
    dyn_npt_alloy.run(steps=c.npt_num_steps_for_alloy)
    write("ALLOY.xyz", alloy)

    return alloy


# ============================================================
# Prepare the molten salt
# ============================================================


def prepare_salt(
    alloy: Atoms | None,
    c: SimulationConfig,
) -> Atoms:
    """Prepare the salt.

    Args:
        alloy (Atoms | None): ASE atoms object of the equilibrated alloy. If alloy is not None, the salt will have a surface that matches the alloy surface. If alloy is None, the salt will be constructed in a cubic cell.
        c (SimulationConfig): Simulation configuration.

    Returns:
        Atoms: ASE atoms object of the non-equilibrated salt.
    """
    # Construct the symbols array by spreading anions and cations evenly, shuffled within their groups
    cations = np.random.permutation(np.repeat(c.salt_cations[0], c.salt_cations[1]))
    anions = np.random.permutation(np.repeat(c.salt_anions[0], c.salt_anions[1]))
    Ntot = len(cations) + len(anions)
    idx = np.linspace(0, Ntot - 1, len(cations), dtype=int)
    mask = np.zeros(Ntot, dtype=bool)
    mask[idx] = True
    symbols = np.empty(Ntot, dtype="<U2")
    symbols[mask] = cations
    symbols[~mask] = anions

    # Get the cell size and volume
    mass = sum(atomic_masses[atomic_numbers[sym]] for sym in symbols)  # amu
    V = mass / (c.initial_density * units.kg / 1e3 * 1e6 / units.m**3)  # Å³
    if alloy is not None:
        box_size_x = alloy.cell[0][0]  # Å
        box_size_y = alloy.cell[1][1]  # Å
        box_size_z = V / (box_size_x * box_size_y)  # Å
        box_size = np.array([box_size_x, box_size_y, box_size_z])  # Å
    else:
        box_length = V ** (1 / 3)  # Å
        box_size = np.array([box_length, box_length, box_length])  # Å

    if c.initial_salt_str == "rocksalt":
        a0 = c.salt_initial_a  # Å
        box_num = np.int32(
            np.ceil(box_size / (2 * a0))
        )  #  Two atoms per rocksalt unit cell
        # Ensure that there are enough positions available in the rocksalt lattice by increasing the height in z-direction
        while len(symbols) > np.prod(box_num):
            box_num[2] += 1
        # Generate an rocksalt lattice with arbitrary symbols
        salt = bulk("XY", "rocksalt", a=a0)
        salt = salt.repeat(box_num)
        # Remove excess positions
        if len(salt) > len(symbols):
            if c.random_removal:
                num_positions_to_remove = len(salt) - len(symbols)
                cat_indices_to_remove = np.random.choice(
                    np.arange(0, len(salt), 2),
                    size=num_positions_to_remove // 2,
                    replace=False,
                )
                an_indices_to_remove = np.random.choice(
                    np.arange(1, len(salt), 2),
                    size=num_positions_to_remove // 2,
                    replace=False,
                )
                indices_to_remove = np.sort(
                    np.concatenate((cat_indices_to_remove, an_indices_to_remove))
                )
                salt = salt[np.setdiff1d(np.arange(len(salt)), indices_to_remove)]
            else:
                salt = salt[: len(symbols)]
        # Populate with the correct chemical symbols
        salt.set_chemical_symbols(symbols)
        salt.set_cell(box_size, scale_atoms=True)

    elif c.initial_salt_str == "random":
        positions = np.zeros((Ntot, 3))
        for i in range(Ntot):
            for j in range(c.max_attempts):
                new_pos = np.random.random(3) * box_size
                if i == 0:  # First atom has no neighbors to check
                    positions[i] = new_pos
                    break
                # Calculate distances to existing atoms
                distances = cdist([new_pos], positions[:i], metric="euclidean")
                if np.all(distances > c.min_distance):
                    positions[i] = new_pos
                    break
            if j == c.max_attempts - 1:
                print(
                    f"Warning: Could not find a suitable position for atom {i+1} after {c.max_attempts} attempts."
                )

        # Create Atoms object
        salt = Atoms(
            symbols=symbols,
            positions=positions,
            cell=box_size,
        )
    return salt


# ============================================================
# Add O2 / H2O impurities by replacing salt ion pairs
# ============================================================


def add_impurities(
    salt: Atoms,
    c: SimulationConfig,
) -> Atoms:
    """Add O2 / H2O impurities by replacing salt ion pairs.

    Args:
        salt (Atoms): ASE atoms object of the salt.
        c (SimulationConfig): Simulation configuration.

    Returns:
        Atoms: ASE atoms object of the salt with impurities.
    """
    if c.impurity == "oxygen":
        n_impurity = c.n_O2
        impurity_symbol = "O2"
    elif c.impurity == "water":
        n_impurity = c.n_H2O
        impurity_symbol = "H2O"
    impurities = []
    for _ in range(n_impurity):
        # Choose a random cation/anion pair from the salt
        atom1 = np.random.choice(salt, 1)[0]
        # Get the closest neighboring atom to the first atom
        dist = np.inf
        for atom in [atom for atom in salt if atom.symbol != atom1.symbol]:
            dist_new = np.linalg.norm(atom1.position - atom.position)
            if dist_new < dist:
                dist = dist_new
                atom2 = atom
        # Place the impurity molecule in the middle of the removed atoms
        impurity_mol = molecule(impurity_symbol)
        impurity_mol.positions = (
            impurity_mol.get_positions()
            - impurity_mol.get_center_of_mass()
            + (atom1.position + atom2.position) / 2
        )
        # Remove the ion pair from the salt (pop higher index first to avoid index shifting issues)
        if atom1.index > atom2.index:
            salt.pop(atom1.index)
            salt.pop(atom2.index)
        else:
            salt.pop(atom2.index)
            salt.pop(atom1.index)
        impurities.append(impurity_mol)
    for imp in impurities:
        salt += imp

    # Set the periodic boundary conditions
    salt.set_pbc([True, True, True])

    write("SALT_IMP.xyz", salt.copy())

    return salt


def run_npt_salt(salt: Atoms, c: SimulationConfig) -> Atoms:
    """Run NPT simulation for only the salt.
    Args:
        salt (Atoms): ASE atoms object of the salt.
        c (SimulationConfig): Simulation configuration.
    Returns:
        Atoms: ASE atoms object of the (volume-wise) equilibrated salt.
    """
    salt.calc = c.calculator
    # Run NPT simulation for only the salt
    dyn_npt_salt = Inhomogeneous_NPTBerendsen(
        salt,
        timestep=c.timestep_fs * units.fs,
        temperature_K=c.temperature_K,
        taut=c.taut_fs * units.fs,
        pressure_au=c.pressure_bar * units.bar,
        taup=c.taup_fs * units.fs,
        compressibility_au=c.compressibility_per_bar / units.bar,
        mask=(0, 0, 1),
    )

    # Create logger that prints pressure
    logger_npt_salt = MDLogger(
        dyn_npt_salt,
        salt,
        c.npt_pure_salt_logfile,
        mode="w",
        header=True,
        stress=True,  # Include pressure info
        peratom=False,
    )

    MaxwellBoltzmannDistribution(salt, temperature_K=c.temperature_K)
    trajectory_npt_salt = Trajectory(c.npt_salt_trajfile, "w", salt)
    dyn_npt_salt.attach(trajectory_npt_salt.write, interval=c.trajectory_write_interval)
    dyn_npt_salt.attach(logger_npt_salt, interval=c.trajectory_log_interval)
    dyn_npt_salt.run(steps=c.npt_num_steps_for_salt)

    # Wrap the salt atom back to the cell
    salt.positions = salt.get_positions(wrap=True, pbc=True)

    write("SALT_IMP_equili.xyz", salt.copy())

    return salt


def add_oxygen_top(salt: Atoms, c: SimulationConfig) -> Atoms:
    """Add additional oxygen on the top layer of simulation cell.
    Args:
        salt (Atoms): ASE atoms object of the salt.
        c (SimulationConfig): Simulation configuration.
    Returns:
        Atoms: ASE atoms object of the salt with additional oxygen atoms.
    """
    if not c.impurity == "oxygen":
        print(
            f"Warning: impurity is {c.impurity} != oxygen, so no oxygen atoms will be added on top"
        )
        return salt
    cell = salt.get_cell()
    cell[2][2] += c.oxygen_top_height
    salt.set_cell(cell)
    cell = salt.get_cell()
    # Add oxygen atoms with distance checking
    for i in range(c.n_O_top):
        attempts = 0
        while attempts < c.max_attempts:
            # Generate a new random position for oxygen
            random_pos = np.array(
                [
                    np.random.rand() * cell[0][0],  # x: random within entire cell
                    np.random.rand() * cell[1][1],  # y: random within entire cell
                    cell[2][2]
                    - c.oxygen_top_height / 2,  # z: fixed at half the height from top
                ]
            )
            # Get all existing oxygen atoms in the structure
            oxygen_indices = [atom.index for atom in salt if atom.symbol == "O"]

            if len(oxygen_indices) > 0:
                # Get positions of all existing oxygen atoms
                oxygen_positions = salt.positions[oxygen_indices]
                # Check distances only to other oxygen atoms
                distances = get_distances(
                    [random_pos], oxygen_positions, cell=cell, pbc=True
                )[1]
                if np.all(distances > c.min_distance):
                    o_add = Atom("O", position=random_pos)
                    salt += o_add
                    break
            else:
                # First oxygen atom, no distance check needed
                o_add = Atom("O", position=random_pos)
                salt += o_add
                break
            attempts += 1

        if attempts == c.max_attempts:
            print(
                f"Warning: Failed to place oxygen atom {i+1} after {c.max_attempts} attempts"
            )

    write("SALT_equili_O2_top.xyz", salt.copy())

    return salt


# ============================================================
# Stick the salt on top of the alloy
# ============================================================
def combine_alloy_salt(
    alloy: Atoms,
    salt: Atoms,
    c: SimulationConfig,
) -> Atoms:
    """Stick the salt on top of the alloy.

    Args:
        alloy (Atoms): ASE atoms object of the alloy.
        salt (Atoms): ASE atoms object of the salt.
        c (SimulationConfig): Simulation configuration.

    Returns:
        Atoms: ASE atoms object of the combined alloy + salt.
    """
    # Ensures that the topmost atom of the alloy is always at the same position as the bottommost atom of the salt
    shift = np.array(
        [0, 0, alloy.cell[2][2] + c.alloy_salt_spacing]
    )  # y remains unchanged
    # Apply to atoms and cell for the salt
    salt.positions += shift
    salt.cell[2] += shift
    # shift the alloy up because of the thermal fluctuation
    alloy.positions += np.array([0, 0, c.alloy_shift])
    # combine the system
    combined = alloy + salt
    combined.cell[2][2] = salt.cell[2][2]

    # Fix the bottom atoms of the bulk to model infinite thickness
    # Works only for PBCs
    combined.set_pbc([True, True, c.pbc_z_direction])
    if c.pbc_z_direction:
        layer_thickness = alloy.cell[2][2] / c.alloy_size[2]
        fixed = [
            atom.index
            for atom in combined
            if atom.position[2] < (c.n_fix_layers_alloy * layer_thickness)
        ]
        combined.set_constraint(FixAtoms(indices=fixed))
    write("alloy_salt_combined.xyz", combined.copy())

    return combined


# ============================================================
# NPT Simulation
# ============================================================


def npt_equilibration(
    system_in: Atoms,
    c: SimulationConfig,
) -> Atoms:
    """Run NPT simulation for only the system_in.

    Args:
        system_in (Atoms): ASE atoms object of the system to be equilibrated.
        c (SimulationConfig): Simulation configuration.

    Returns:
        Atoms: ASE atoms object of the (volume-wise) equilibrated system.
    """
    system = system_in.copy()
    if c.pbc_z_direction:
        system.set_constraint(system_in.constraints)
    system.calc = c.calculator

    dyn_npt = NPTBerendsen(
        system,
        timestep=c.timestep_fs * units.fs,
        temperature_K=c.temperature_K,
        taut=c.taut_fs * units.fs,
        pressure_au=c.pressure_bar * units.bar,
        taup=c.taup_fs * units.fs,
        compressibility_au=c.compressibility_per_bar / units.bar,
    )

    # Create logger that prints pressure
    logger_npt = MDLogger(
        dyn_npt,
        system,
        c.main_npt_logfile,
        mode="w",
        header=True,
        stress=True,  # Include pressure info
        peratom=False,
    )

    def print_pressure_3d_npt():
        temp = system.get_temperature()
        pe = system.get_potential_energy()
        stress = system.get_stress()
        pressures = -stress[:3] / units.GPa  # Convert to pressure in GPa
        print(
            f"Step: {dyn_npt.nsteps:5d} | "
            f"temp: {temp:7.3f} K | "
            f"Pe:   {pe:7.3f} eV | "
            f"P_xx: {pressures[0]:7.3f} GPa | "
            f"P_yy: {pressures[1]:7.3f} GPa | "
            f"P_zz: {pressures[2]:7.3f} GPa"
        )

    trajectory_npt = Trajectory(c.main_npt_trajfile, "w", system)
    dyn_npt.attach(trajectory_npt.write, interval=c.trajectory_write_interval)
    dyn_npt.attach(print_pressure_3d_npt, interval=c.trajectory_print_interval)
    dyn_npt.attach(logger_npt, interval=c.trajectory_log_interval)
    dyn_npt.run(steps=c.npt_num_steps)

    return system


# ============================================================
# NPT Simulation
# ============================================================


def nvt_simulation(
    system: Atoms,
    c: SimulationConfig,
) -> Atoms:
    """Run NVT simulation.

    Args:
        system (Atoms): ASE atoms object of the system.
        c (SimulationConfig): Simulation configuration.

    Returns:
        Atoms: ASE atoms object of the equilibrated system.
    """
    system.calc = c.calculator

    dyn_nvt = Langevin(
        system,
        timestep=c.timestep_fs * units.fs,
        temperature_K=c.temperature_K,
        friction=c.friction_per_fs / units.fs,
    )

    def print_pressure_3d_nvt():
        temp = system.get_temperature()
        pe = system.get_potential_energy()
        stress = system.get_stress()
        pressures = -stress[:3] / units.GPa  # Convert to pressure in GPa
        print(
            f"Step: {dyn_nvt.nsteps:5d} | "
            f"temp: {temp:7.3f} K | "
            f"Pe:   {pe:7.3f} eV | "
            f"P_xx: {pressures[0]:7.3f} GPa | "
            f"P_yy: {pressures[1]:7.3f} GPa | "
            f"P_zz: {pressures[2]:7.3f} GPa"
        )

    # Logger for the NVT simulation
    logger_nvt = MDLogger(
        dyn_nvt,
        system,
        c.main_nvt_logfile,
        mode="w",
        header=True,
        stress=True,  # Include pressure info
        peratom=False,
    )

    trajectory_nvt = Trajectory(c.main_nvt_trajfile, "w", system)
    dyn_nvt.attach(trajectory_nvt.write, interval=c.trajectory_write_interval)
    dyn_nvt.attach(print_pressure_3d_nvt, interval=c.trajectory_print_interval)
    dyn_nvt.attach(logger_nvt, interval=c.trajectory_log_interval)

    dyn_nvt.run(steps=c.nvt_num_steps)

    return system


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # Set the rng seed
    set_seed(42)
    # Load the simulation config
    config = SimulationConfig()
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
