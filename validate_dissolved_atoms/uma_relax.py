from pathlib import Path

import moltensaltcalc as msc
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter

def make_uma_calculator(
    device: str = "cuda",
    model_version: str = "1p1",
    model_size: str = "small",
    task: str = "oc20",
) -> Calculator:
    sim = msc.MoltenSaltSimulator(
        model_name="fairchem",
        model_parameters={
            "model_task": task,
            "model_version": model_version,
            "model_size": model_size,
            "InferenceSettings": "default",
        },
        device=device,
    )
    return sim.calc


def relax_with_uma(
    atoms: Atoms,
    calculator: Calculator,
    output_dir: str | Path,
    fmax: float = 1e-4,
    max_steps: int = 10000,
    box_sidelength_A: float = 20.0,
    relax_cell: bool = False,
) -> Atoms:
    """Relax an ASE Atoms object using UMA."""

    if len(atoms) < 2:
        print("Warning: Less than 2 atoms provided, but UMA can only relax systems with 2 or more atoms. Returning the original atoms object.")
        return atoms

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    atoms = atoms.copy()

    atoms.set_pbc(True)
    atoms.calc = calculator

    if relax_cell:
        # Apply the Frechet cell filter to allow cell relaxation
        atoms = FrechetCellFilter(atoms)
    else:
        # Otherwise place the atoms in a box centered with vacuum
        atoms.set_cell([box_sidelength_A, box_sidelength_A, box_sidelength_A])
        atoms.center()

    trajectory = output_dir / "relax.traj"
    logfile = output_dir / "relax.log"

    optimizer = LBFGS(
        atoms,
        trajectory=str(trajectory),
        logfile=str(logfile),
    )
    optimizer.run(
        fmax=fmax,
        steps=max_steps,
    )

    # If an ASE Filter was returned, unpack the underlying Atoms object
    if hasattr(atoms, "atoms"):
        atoms = atoms.atoms

    return atoms
