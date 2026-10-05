from dataclasses import dataclass

from ase import Atoms
from ase.build import bulk

# ============================================================
# Species definitions for compounds to be relaxed
# ============================================================


@dataclass(frozen=True)
class Species:
    name: str
    atoms: Atoms
    charge: int = 0
    multiplicity: int = 1
    magmoms: list[float] | None = None


# ============================================================
# Initial structures
#
# These are deliberately simple starting geometries.
# They are NOT intended to be chemically accurate structures, just reasonable enough to start relaxation.
# UMA/VASP will subsequently relax them.
#
# ============================================================


def crf2():
    return Atoms(
        "CrF2",
        positions=[
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.80),
            (0.0, 0.0, -1.80),
        ],
    )


def crf3():
    return Atoms(
        "CrF3",
        positions=[
            (0.0, 0.0, 0.0),
            (1.80, 0.0, 0.0),
            (-0.90, 1.56, 0.0),
            (-0.90, -1.56, 0.0),
        ],
    )


def cro2():
    return Atoms(
        "CrO2",
        positions=[
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.65),
            (0.0, 0.0, -1.65),
        ],
    )


def cro3():
    return Atoms(
        "CrO3",
        positions=[
            (0.0, 0.0, 0.0),
            (1.60, 0.0, 0.0),
            (-0.80, 1.386, 0.0),
            (-0.80, -1.386, 0.0),
        ],
    )


def cro2f2():
    return Atoms(
        "CrO2F2",
        positions=[
            (0.0, 0.0, 0.0),
            (1.60, 0.0, 0.0),
            (-1.60, 0.0, 0.0),
            (0.0, 1.80, 0.0),
            (0.0, -1.80, 0.0),
        ],
    )


def cro3f():
    return Atoms(
        "CrO3F",
        positions=[
            (0.0, 0.0, 0.0),
            (1.65, 0.0, 0.0),
            (-0.55, 1.56, 0.0),
            (-0.55, -0.78, 1.35),
            (-0.55, -0.78, -1.35),
        ],
    )


def cro4():
    return Atoms(
        "CrO4",
        positions=[
            (0.0, 0.0, 0.0),
            (1.65, 0.0, 0.0),
            (-0.55, 1.56, 0.0),
            (-0.55, -0.78, 1.35),
            (-0.55, -0.78, -1.35),
        ],
    )


def feo():
    return Atoms(
        "FeO",
        positions=[
            (0.0, 0.0, 0.0),
            (1.65, 0.0, 0.0),
        ],
    )


def feo2():
    return Atoms(
        "FeO2",
        positions=[
            (0.0, 0.0, 0.0),
            (1.65, 0.0, 0.0),
            (-1.65, 0.0, 0.0),
        ],
    )


def feo3():
    return Atoms(
        "FeO3",
        positions=[
            (0.0, 0.0, 0.0),
            (1.65, 0.0, 0.0),
            (-0.825, 1.429, 0.0),
            (-0.825, -1.429, 0.0),
        ],
    )


def feof():
    return Atoms(
        "FeOF",
        positions=[
            (0.0, 0.0, 0.0),
            (1.65, 0.0, 0.0),
            (0.0, 1.80, 0.0),
        ],
    )


def feo2f():
    return Atoms(
        "FeO2F",
        positions=[
            (0.0, 0.0, 0.0),
            (1.65, 0.0, 0.0),
            (-0.825, 1.429, 0.0),
            (0.0, -1.80, 0.0),
        ],
    )


def feof2():
    return Atoms(
        "FeOF2",
        positions=[
            (0.0, 0.0, 0.0),
            (1.65, 0.0, 0.0),
            (0.0, 1.80, 0.0),
            (0.0, -1.80, 0.0),
        ],
    )


def o2():
    return Atoms(
        "O2",
        positions=[
            (0.0, 0.0, 0.0),
            (1.21, 0.0, 0.0),
        ]
    )


def f2():
    return Atoms(
        "F2",
        positions=[
            (0.0, 0.0, 0.0),
            (1.40, 0.0, 0.0),
        ]
    )


def fe_bcc():
    return bulk(
        "Fe",
        crystalstructure="bcc",
        a=2.87  # A
    ).repeat((2, 2, 2))


def cr_bcc():
    return bulk(
        "Cr",
        crystalstructure="bcc",
        a=2.88,   # A
    ).repeat((2, 2, 2))


# ============================================================
# Species database
# ============================================================

SPECIES = {
    # Chromium fluorides
    "CrF2": Species(
        name="CrF2",
        atoms=crf2(),
        multiplicity=5,
        magmoms=[4.0, 0.0, 0.0],  # d4
    ),
    "CrF3": Species(
        name="CrF3",
        atoms=crf3(),
        multiplicity=4,
        magmoms=[3.0, 0.0, 0.0, 0.0],  # d3
    ),
    # Chromium oxides
    "CrO2": Species(
        name="CrO2",
        atoms=cro2(),
        multiplicity=3,
        magmoms=[2.0, 0.0, 0.0],  # d2
    ),
    "CrO3": Species(
        name="CrO3",
        atoms=cro3(),
        magmoms=[0.0, 0.0, 0.0, 0.0],  # d0
    ),
    # Chromium oxyfluorides
    "CrO2F2": Species(
        name="CrO2F2",
        atoms=cro2f2(),
        magmoms=[0.0, 0.0, 0.0, 0.0, 0.0],  # d0
    ),
    # Cr(VI) oxyfluoride ion
    "CrO3F-": Species(
        name="CrO3F-",
        atoms=cro3f(),
        charge=-1,
        magmoms=[0.0, 0.0, 0.0, 0.0, 0.0],  # d0
    ),
    # Chromate
    "CrO4--": Species(
        name="CrO4--",
        atoms=cro4(),
        charge=-2,
        magmoms=[0.0, 0.0, 0.0, 0.0, 0.0],  # d0
    ),
    # Iron oxides
    "FeO": Species(
        name="FeO",
        atoms=feo(),
        multiplicity=7,
        magmoms=[6.0, 0.0],  # d6
    ),
    "FeO2": Species(
        name="FeO2",
        atoms=feo2(),
        multiplicity=5,
        magmoms=[4.0, 0.0, 0.0],  # d4
    ),
    "FeO3": Species(
        name="FeO3",
        atoms=feo3(),
        multiplicity=3,
        magmoms=[2.0, 0.0, 0.0, 0.0],  # d2
    ),
    # Iron oxyfluorides
    "FeOF": Species(
        name="FeOF",
        atoms=feof(),
        multiplicity=6,
        magmoms=[5.0, 0.0, 0.0],  # d5
    ),
    "FeO2F": Species(
        name="FeO2F",
        atoms=feo2f(),
        multiplicity=4,
        magmoms=[3.0, 0.0, 0.0, 0.0],  # d3
    ),
    "FeOF2": Species(
        name="FeOF2",
        atoms=feof2(),
        multiplicity=5,
        magmoms=[4.0, 0.0, 0.0, 0.0],  # d4
    ),
    # Reference energies for formation energy
    "O2": Species(
        name="O2",
        atoms=o2(),
        multiplicity=3,
        magmoms=[2.0, 0.0],  # p4, triplet
    ),
    "F2": Species(
        name="F2",
        atoms=f2(),
        multiplicity=1,
        magmoms=[0.0, 0.0],  # p6
    ),
    "Cr_bcc": Species(
        name="Cr_bcc",
        atoms=cr_bcc(),
        multiplicity=7,
        magmoms=8*[5.0],  # d5
    ),
    "Fe_bcc": Species(
        name="Fe_bcc",
        atoms=fe_bcc(),
        multiplicity=5,
        magmoms=8*[4.0],  # d6
    ),
}


def get_species(name: str) -> Species:
    """Return a copy of a species called by its name"""

    if name not in SPECIES:
        raise KeyError(
            f"Unknown species {name}. " f"Available species: {list(SPECIES)}"
        )

    species = SPECIES[name]

    return Species(
        name=species.name,
        atoms=species.atoms.copy(),
        charge=species.charge,
        multiplicity=species.multiplicity,
        magmoms=species.magmoms,
    )
