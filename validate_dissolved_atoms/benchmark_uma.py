import argparse
from pathlib import Path
import numpy as np

from ase.io import write
from species import SPECIES
from uma_relax import make_uma_calculator, relax_with_uma

BASE_DIR = Path("calculations")

UMA_DEVICE = "cuda"
UMA_MODEL = "1p1"
UMA_MODEL_SIZE = "small"
UMA_TASK = "oc20"


def main(species_idx: list[int] | None, species_names: list[str] | None):

    uma_calc = make_uma_calculator(
        device=UMA_DEVICE,
        model_version=UMA_MODEL,
        model_size=UMA_MODEL_SIZE,
        task=UMA_TASK,
    )

    for i, (name, species) in enumerate(SPECIES.items()):

        if species_idx is not None and i not in species_idx:
            continue

        if species_names is not None and name not in species_names:
            continue

        print()
        print("=" * 70)
        print(f"Species: {name}")
        print("=" * 70)

        species_dir = BASE_DIR / name / "uma"
        atoms = species.atoms.copy()

        # Store the original structure
        species_dir.mkdir(parents=True, exist_ok=True)
        write(
            species_dir / f"initial.xyz",
            atoms,
        )

        print("Running UMA...")

        uma_atoms = relax_with_uma(
            atoms,
            calculator=uma_calc,
            output_dir=species_dir,
            fmax=1e-2,
            relax_cell=name in ["Cr_bcc", "Fe_bcc"],
        )

        if uma_atoms.calc is not None:
            print(f"UMA relaxed structure for species {name} with E={uma_atoms.get_potential_energy():.3f} eV")

        # Save the relaxed structure
        write(
            species_dir / "relaxed.extxyz",
            uma_atoms,
        )

    print()
    print("=" * 70)
    print("Benchmark finished")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--species_idx",
        type=int,
        nargs="*",
        help="Indices of species to benchmark. If not provided, all species are benchmarked.",
        default=None,
    )
    parser.add_argument(
        "--species",
        type=str,
        nargs="*",
        help="Names of species to benchmark. If not provided, all species are benchmarked.",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="RNG seed.",
        default=42,
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    main(species_idx=args.species_idx, species_names=args.species)
