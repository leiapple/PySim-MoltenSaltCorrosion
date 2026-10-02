from collections import defaultdict
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read


# ============================================================
# Configuration
# ============================================================

CALCULATIONS_DIR = Path(__file__).parent / "calculations"
OUTPUT_DIR = CALCULATIONS_DIR / "comparison"

# From relaxing the single element with VASP
VASP_ATOM_ENERGIES: dict[str, float] = {
    "Cr": -5.43554838,  # eV
    "Fe": -3.37674862,
    "F": -0.43733750,
    "O":  -1.54647199,
}

# From https://github.com/facebookresearch/fairchem/blob/main/configs/uma/training_release/element_refs/iso_atom_elem_refs.yaml
UMA_ATOM_ENERGIES: dict[str, float] = {
    "Cr": -1.0934019,  # eV
    "Fe": -1.23073475,
    "F": -0.10298174,
    "O": -0.10862579,
}

# A pair of atoms is considered bonded when:
#     distance < COVALENT_FACTOR * (r_cov_i + r_cov_j)

COVALENT_FACTOR = 1.25
COVALENT_RADII = {
    "Cr": 1.39,  # A
    "Fe": 1.32,
    "F": 0.57,
    "O": 0.66,
}

# ============================================================
# Helpers
# ============================================================


def find_relaxed_structure(species_dir: Path) -> Path:
    """
    Find the relaxed structure for one species and one method.

    Expected names:
        vasp -> vasp/relaxed.xyz
        uma  -> uma/relaxed.extxyz
    """

    matches = list(species_dir.rglob("relaxed*"))

    if not matches:
        raise FileNotFoundError(
            f"Could not find relaxed structure below {species_dir}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Found multiple relaxed structures for {species_dir}: {', '.join(str(path) for path in matches)}"
        )

    return matches[0]


def formation_energy(
    atoms: Atoms,
    total_energy: float,
    atomic_energies: dict[str, float],
) -> float:
    """Calculate the molecular formation/atomization energy relative to isolated
    neutral atoms."""

    symbols = atoms.get_chemical_symbols()

    missing = set(symbols) - set(atomic_energies)

    if missing:
        raise ValueError(f"Missing atomic reference energies for: {', '.join(sorted(missing))}")

    reference_energy = sum(atomic_energies[symbol] for symbol in symbols)

    return total_energy - reference_energy


def get_energy_from_structure(atoms: Atoms) -> float:
    """
    Get the energy stored in an ASE structure.

    This works if the relaxed XYZ was written by ASE as an extended XYZ containing the calculator energy.
    """

    try:
        return float(atoms.get_potential_energy())
    except AttributeError:
        pass

    if "energy" in atoms.info:
        return float(atoms.info["energy"])

    if "free_energy" in atoms.info:
        return float(atoms.info["free_energy"])

    raise ValueError("No energy found in the structure's ASE info dictionary.")


def get_bonds(atoms: Atoms) -> list[tuple[int, int]]:
    """
    Determine molecular bonds from covalent radii.

    Returns atom-index pairs (i, j).
    """

    symbols = atoms.get_chemical_symbols()

    missing = set(symbols) - set(COVALENT_RADII)

    if missing:
        raise ValueError(f"Missing covalent radii for: {', '.join(sorted(missing))}")

    bonds = []

    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            r_ij = atoms.get_distance(i, j, mic=False)

            cutoff = COVALENT_FACTOR * (
                COVALENT_RADII[symbols[i]]
                + COVALENT_RADII[symbols[j]]
            )

            if r_ij <= cutoff:
                bonds.append((i, j))

    return bonds


def get_bond_lengths(atoms: Atoms) -> list[dict]:
    """Return all detected bond lengths."""

    symbols = atoms.get_chemical_symbols()
    bonds = get_bonds(atoms)

    result = []

    for i, j in bonds:
        result.append(
            {
                "atom_i": i,
                "atom_j": j,
                "element_i": symbols[i],
                "element_j": symbols[j],
                "length_A": atoms.get_distance(i, j, mic=False),
            }
        )

    return result


def get_bond_angles(atoms: Atoms) -> list[dict]:
    """
    Calculate all angles formed by two bonds sharing a central atom.

    For example:

        F -- Cr -- F

    gives the angle:

        F-Cr-F
    """

    symbols = atoms.get_chemical_symbols()
    bonds = get_bonds(atoms)

    # Build neighbor list.
    neighbors: dict[int, list[int]] = defaultdict(list)

    for i, j in bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)

    result = []

    for center, center_neighbors in neighbors.items():

        if len(center_neighbors) < 2:
            continue

        for a in range(len(center_neighbors)):
            for b in range(a + 1, len(center_neighbors)):

                i = center_neighbors[a]
                j = center_neighbors[b]

                angle = atoms.get_angle(
                    i,
                    center,
                    j,
                    mic=False,
                )

                result.append(
                    {
                        "atom_i": i,
                        "atom_center": center,
                        "atom_j": j,
                        "element_i": symbols[i],
                        "element_center": symbols[center],
                        "element_j": symbols[j],
                        "angle_deg": angle,
                    }
                )

    return result


def bond_key(bond):
    """Match bonds by element pair rather than atom indices. This is important if UMA and VASP reorder atoms."""
    return tuple(
        sorted(
            [bond["element_i"], bond["element_j"]]
        )
    )

def angle_key(angle):
    """Match angles by element triplet rather than atom indices. This is important if UMA and VASP reorder atoms."""
    return (
        angle["element_center"],
        tuple(
            sorted(
                [
                    angle["element_i"],
                    angle["element_j"],
                ]
            ),
        ),
    )


# ============================================================
# Comparison orchestrator
# ============================================================

def compare_species(
    species: str,
    species_dir: Path,
) -> tuple[dict, list[dict], list[dict]]:
    """Compare UMA and VASP for one species."""

    vasp_path = find_relaxed_structure(species_dir / "vasp")
    uma_path = find_relaxed_structure(species_dir / "uma")

    vasp_atoms: Atoms = read(vasp_path)  # type: ignore
    uma_atoms: Atoms = read(uma_path)  # type: ignore

    # Formation energy
    vasp_energy = get_energy_from_structure(vasp_atoms)
    uma_energy = get_energy_from_structure(uma_atoms)

    vasp_formation = formation_energy(
        vasp_atoms,
        vasp_energy,
        VASP_ATOM_ENERGIES,
    )

    uma_formation = formation_energy(
        uma_atoms,
        uma_energy,
        UMA_ATOM_ENERGIES,
    )

    energy_result = {
        "species": species,
        "vasp_energy_eV": vasp_energy,
        "uma_energy_eV": uma_energy,
        "vasp_formation_energy_eV": vasp_formation,
        "uma_formation_energy_eV": uma_formation,
        "formation_energy_difference_eV": (
            uma_formation - vasp_formation
        ),
    }

    # Bond lengths
    vasp_bonds = get_bond_lengths(vasp_atoms)
    uma_bonds = get_bond_lengths(uma_atoms)

    bond_length_results = []

    vasp_by_type = defaultdict(list)
    uma_by_type = defaultdict(list)

    for bond in vasp_bonds:
        vasp_by_type[bond_key(bond)].append(bond)

    for bond in uma_bonds:
        uma_by_type[bond_key(bond)].append(bond)

    for bond_type in sorted(set(vasp_by_type) | set(uma_by_type)):
        vasp_values = sorted(
            bond["length_A"]
            for bond in vasp_by_type[bond_type]
        )

        uma_values = sorted(
            bond["length_A"]
            for bond in uma_by_type[bond_type]
        )

        if len(vasp_values) != len(uma_values):
            raise ValueError(
                f"{species}: different number of {bond_type} bonds "
                f"between VASP ({len(vasp_values)}) and UMA "
                f"({len(uma_values)})."
            )

        for index, (vasp_length, uma_length) in enumerate(
            zip(vasp_values, uma_values)
        ):
            bond_length_results.append(
                {
                    "species": species,
                    "bond_type": f"{bond_type[0]}-{bond_type[1]}",
                    "bond_number": index,
                    "vasp_length_A": vasp_length,
                    "uma_length_A": uma_length,
                    "difference_A": uma_length - vasp_length,
                }
            )
    print("before bond angles")
    # Bond angles
    vasp_angles = get_bond_angles(vasp_atoms)
    uma_angles = get_bond_angles(uma_atoms)

    vasp_by_type = defaultdict(list)
    uma_by_type = defaultdict(list)

    for angle in vasp_angles:
        vasp_by_type[angle_key(angle)].append(angle)

    for angle in uma_angles:
        uma_by_type[angle_key(angle)].append(angle)

    angle_results = []

    for angle_type in sorted(set(vasp_by_type) | set(uma_by_type)):
        vasp_values = sorted(
            angle["angle_deg"]
            for angle in vasp_by_type[angle_type]
        )

        uma_values = sorted(
            angle["angle_deg"]
            for angle in uma_by_type[angle_type]
        )

        if len(vasp_values) != len(uma_values):
            raise ValueError(
                f"{species}: different number of "
                f"{angle_type} angles between VASP "
                f"({len(vasp_values)}) and UMA "
                f"({len(uma_values)})."
            )

        for index, (vasp_angle, uma_angle) in enumerate(zip(vasp_values, uma_values)):
            angle_results.append(
                {
                    "species": species,
                    "angle_type": f"{angle_type[0]}-{angle_type[1]}",
                    "angle_number": index,
                    "vasp_angle_deg": vasp_angle,
                    "uma_angle_deg": uma_angle,
                    "difference_deg": uma_angle - vasp_angle,
                }
            )

    return (
        energy_result,
        bond_length_results,
        angle_results,
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    energy_results = []
    bond_length_results = []
    angle_results = []

    # Each direct subdirectory is assumed to correspond to one species.
    species_dirs = sorted(
        path
        for path in CALCULATIONS_DIR.iterdir()
        if path.is_dir()
        and path.name != OUTPUT_DIR.name
    )

    print(f"Found {len(species_dirs)} species directories.")

    for species_dir in species_dirs:

        species = species_dir.name

        if species in VASP_ATOM_ENERGIES.keys():
            print(f"  Skipping single atoms species {species}...")
            continue

        print(f"\nProcessing {species}...")

        try:
            (
                energy_result,
                lengths,
                angles,
            ) = compare_species(
                species,
                species_dir,
            )

        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        energy_results.append(energy_result)
        bond_length_results.extend(lengths)
        angle_results.extend(angles)

        ediff = energy_result['formation_energy_difference_eV']
        print(
            f"Energy Difference: {ediff:+.6f} eV = {100*abs(ediff/energy_result['vasp_formation_energy_eV']):.3f}%"
        )
        print(
            f"Bond length differences (VASP vs UMA):"
        )
        for bond in lengths:
            print(f"- #{bond['bond_number']}: type: {bond['bond_type']}, difference: {bond['difference_A']:.3f} A = {100*bond['difference_A']/bond['vasp_length_A']:.3f}%")

        print(f"Angle differences (VASP vs UMA):")
        for angle in angles:
            print(f"- #{angle['angle_number']}: type: {angle['angle_type']}, difference: {angle['difference_deg']:.3f}° = {100*angle['difference_deg']/angle['vasp_angle_deg']:.3f}%")
        print()

    # Save the results to npy files
    np.save(
        OUTPUT_DIR / "formation_energies.npy",
        np.array(energy_results, dtype=object),
        allow_pickle=True,
    )

    np.save(
        OUTPUT_DIR / "bond_lengths.npy",
        np.array(bond_length_results, dtype=object),
        allow_pickle=True,
    )

    np.save(
        OUTPUT_DIR / "bond_angles.npy",
        np.array(angle_results, dtype=object),
        allow_pickle=True,
    )

    print("\nSaved:")
    print(f"  {OUTPUT_DIR / 'formation_energies.npy'}")
    print(f"  {OUTPUT_DIR / 'bond_lengths.npy'}")
    print(f"  {OUTPUT_DIR / 'bond_angles.npy'}")


if __name__ == "__main__":
    main()
