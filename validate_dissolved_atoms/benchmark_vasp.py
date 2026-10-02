import argparse
from pathlib import Path

from ase.io import write
from species import SPECIES
from vasp_relax import VaspParams, relax_with_vasp

BASE_DIR = Path("calculations")


def main(
    vasp_command: str,
    ncore: int,
    species_idx: list[int] | None,
    species_names: list[str] | None,
    kernel_truncation: bool,
    overwrite: bool,
):

    for idx, (name, species) in enumerate(SPECIES.items()):

        if species_idx is not None and idx not in species_idx:
            print(f"Skipping species {name} (index {idx})")
            continue

        if species_names is not None and name not in species_names:
            print(f"Skipping species {name} (name not in specified species names)")
            continue

        print()
        print("=" * 70)
        print(f"Species: {name} (idx {idx})")
        print("=" * 70)

        species_dir = BASE_DIR / name / "vasp"

        atoms = species.atoms.copy()

        # Store the original structure
        species_dir.mkdir(parents=True, exist_ok=True)

        write(
            species_dir / "initial.xyz",
            atoms,
        )

        print("Running VASP...")

        vasp_atoms, vasp_energy = relax_with_vasp(
            atoms,
            output_dir=species_dir,
            vasp_params=VaspParams(command=vasp_command, ncore=ncore),
            charge=species.charge,
            magmoms=species.magmoms,
            kernel_truncation=kernel_truncation,
            overwrite=overwrite,
        )

        write(
            species_dir / "relaxed.xyz",
            vasp_atoms,
        )

        print(f"VASP relaxed energy: " f"{vasp_energy:.8f} eV")

    print()
    print("=" * 70)
    print("Benchmark finished")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vasp_command",
        type=str,
        default="mpirun --bind-to hwthread --map-by ppr:12:node:PE=8:spawn vasp_gam",
    )
    parser.add_argument("--ncore", type=int, default=1, help="Value for the NCORE vasp argument. Somehow it breaks for >1.")
    parser.add_argument("--species_idx", nargs="*", type=int, default=None, help="Indices of the species to consider (e.g. '1 2'). If not provided all will be used.")
    parser.add_argument("--species", nargs="*", type=str, default=None, help="Names of the species to be considered (e.g. 'CrF2 CrO3'). If not provided all will be used.")
    parser.add_argument("--kernel_truncation", action="store_true", help="Enable kernel truncation in VASP calculations.")
    parser.add_argument("--overwrite", action="store_true", help="When provided, existing VASP outputs will be overwritten without asking for confirmation.")
    args = parser.parse_args()

    main(
        args.vasp_command,
        args.ncore,
        args.species_idx,
        args.species,
        args.kernel_truncation,
        args.overwrite,
    )
