from dataclasses import asdict, dataclass, field
from pathlib import Path

from ase import Atoms
from ase.calculators.vasp import Vasp
from ase.io import write
from collections import Counter

# Needed because we want to calculate charged molecules => Need to get the number of neutral electrons to specify NELECT in INCAR
@dataclass(frozen=True)
class VaspSetup:
    suffix: str
    zval: float

POTENTIALS = {
    "Cr": VaspSetup("_pv", 12),
    "Fe": VaspSetup("_pv", 14),
    "F":  VaspSetup("", 7),
    "O":  VaspSetup("", 6),
}

@dataclass
class VaspParams:
    command: str = "vasp_gam"

    # Electronic structure
    xc: str = "RPBE"
    encut: int = 500
    ediff: float = 1e-6
    ediffg: float = -1e-2
    ismear: int = 0
    sigma: float = 0.05
    algo: str = "Normal"

    # Spin
    ispin: int = 2

    # Geometry optimization
    ibrion: int = 2
    nsw: int = 1000
    isif: int = 2

    # k-points
    kpts: list[int] = field(default_factory=lambda: [1, 1, 1])
    gamma: bool = True

    # Numerical accuracy
    prec: str = "Accurate"
    lreal: bool = False

    # Symmetry
    isym: int = 0

    # Initialization
    icharg: int = 2
    nelm: int = 1000

    # Output
    txt: str = "vasp.out"
    lwave: bool = False
    lcharg: bool = False

    # Parallelization / HPC
    ncore: int = 1
    nsim: int = 1
    # lplane: bool = True

class VaspWithKernelTruncation(Vasp):
    """ASE VASP calculator with support for VASP's KERNEL_TRUNCATION block."""

    def write_incar(self, atoms, directory="./", **kwargs):
        # Let ASE write the normal INCAR first.
        super().write_incar(atoms, directory=directory, **kwargs)

        # Then append the VASP 6.5+ kernel-truncation block.
        path = Path(directory) / "INCAR"

        with path.open("a") as f:
            f.write(
                "\nKERNEL_TRUNCATION {\n"
                "    LTRUNCATE       = T\n"
                "    IDIMENSIONALITY = 0\n"
                "    LCOARSEN        = T\n"
                "}\n"
            )



def neutral_nelect(atoms: Atoms) -> int:
    counts = Counter(atoms.get_chemical_symbols())

    missing = set(counts) - set(POTENTIALS)
    if missing:
        raise ValueError(f"No VASP potential/ZVAL defined for element(s): {', '.join(sorted(missing))}")

    return sum(counts[symbol] * POTENTIALS[symbol].zval for symbol in counts)


def get_vasp_setups() -> dict[str, str]:
    return {
        "base": "recommended",
        **{
            symbol: potential.suffix
            for symbol, potential in POTENTIALS.items()
        },
    }


def relax_with_vasp(
    atoms: Atoms,
    output_dir: str | Path,
    charge: int = 0,
    magmoms: list[float] | None = None,
    vasp_params: VaspParams | None = None,
    box_sidelength_A: float = 20.0,
    kernel_truncation: bool = True,
    overwrite: bool = False,
    relax_cell: bool = False,
) -> tuple[Atoms, float]:
    """Relax the given atoms using VASP and return the relaxed structure and its energy. Files are saved to the specified output directory."""

    calc_kwargs = asdict(vasp_params) if vasp_params is not None else {}

    output_dir = Path(output_dir)
    calc_kwargs["directory"] = str(output_dir)
    calc_kwargs["setups"] = get_vasp_setups()
    if output_dir.exists() and not overwrite:
        res = input(
            f"Output directory '{output_dir}' already exists and results could be overwritten. Continue? [y/N]: "
        )
        if res.lower() != "y":
            print("Exiting to avoid overwriting existing results.")
            exit()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the atoms are periodic and centered with vacuum
    atoms.set_pbc(True)
    if relax_cell:
        if kernel_truncation:
            print("WARNING: Cannot use kernel truncation when relaxing the cell. Disabling kernel truncation.")
            kernel_truncation = False
        if calc_kwargs.get("isif", 2) != 3:
            print("WARNING: ISIF is not set to 3 for cell relaxation. Overriding ISIF to 3.")
            calc_kwargs["isif"] = 3
    else:
        atoms.set_cell([box_sidelength_A, box_sidelength_A, box_sidelength_A])
        atoms.center()

    # For neutral systems, VASP automatically determines the number of valence electrons from the selected POTCAR files, but for charged systems we need to compute the number of electrons VASP will see
    if charge != 0:
        nelect = neutral_nelect(atoms) - charge
        calc_kwargs["nelect"] = nelect

    # Set the magnetic moments if provided
    if magmoms is not None:
        if calc_kwargs.get("ispin", 1) == 2:
            if len(magmoms) != len(atoms):
                raise ValueError(
                    f"Length of magmoms ({len(magmoms)}) list must match the number of atoms ({len(atoms)})."
                )
            calc_kwargs["magmom"] = magmoms
        else:
            raise ValueError("ISPIN must be set to 2 to specify magnetic moments.")
    elif calc_kwargs.get("ispin", 1) == 2:
        raise ValueError("ISPIN=2 requires magnetic moments to be provided.")

    write(
        output_dir / "vasp_initial.xyz",
        atoms,
    )

    # Setup the VASP calculator with the specified parameters for relaxation
    if kernel_truncation:
        calc = VaspWithKernelTruncation(**calc_kwargs)
    else:
        calc = Vasp(**calc_kwargs)

    atoms.calc = calc

    # Perform the relaxation in VASP and get the potential energy of the relaxed structure
    energy = atoms.get_potential_energy()

    write(
        output_dir / "vasp_relaxed.xyz",
        atoms,
    )

    return atoms, energy
