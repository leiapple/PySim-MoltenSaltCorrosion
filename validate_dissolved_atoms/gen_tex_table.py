from pathlib import Path
import re

import numpy as np


# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path(__file__).parent / "calculations" / "comparison"


# Define which columns should appear in each table.
#
# The keys MUST match the keys in the dictionaries stored in the .npy files.
#
# Add/remove entries here if your actual dictionary keys differ.

TABLE_CONFIG = {
    "formation_energies.npy": {
        "columns": {
            "species": r"Species",
            "vasp_formation_energy_eV": r"$E_\mathrm{form}^{\mathrm{VASP}}$ (eV)",
            "uma_formation_energy_eV": r"$E_\mathrm{form}^{\mathrm{UMA}}$ (eV)",
            "formation_energy_difference_eV": r"$\Delta E_\mathrm{form}$ (eV)",
        },
        "decimals": {
            "vasp_formation_energy_eV": 3,
            "uma_formation_energy_eV": 3,
            "formation_energy_difference_eV": 3,
        },
        "caption": "Formation energies of relaxed gas-phase structures obtained using VASP and UMA.",
        "label": "tab:dissolved_species_formation_energies",
    },

    "bond_lengths.npy": {
        "columns": {
            "species": r"Species",
            "bond_type": r"Bond",
            "vasp_length_A": r"$d_\mathrm{VASP}$ (\AA)",
            "uma_length_A": r"$d_\mathrm{UMA}$ (\AA)",
            "difference_A": r"$\Delta d$ (\AA)",
        },
        "decimals": {
            "vasp_length_A": 3,
            "uma_length_A": 3,
            "difference_A": 3,
        },
        "caption": "Bond lengths of relaxed gas-phase structures obtained using VASP and UMA.",
        "label": "tab:dissolved_species_lengths",
    },

    "bond_angles.npy": {
        "columns": {
            "species": r"Species",
            "angle_type": r"Angle",
            "vasp_angle_deg": r"$\theta_\mathrm{VASP}$ ($^\circ$)",
            "uma_angle_deg": r"$\theta_\mathrm{UMA}$ ($^\circ$)",
            "difference_deg": r"$\Delta\theta$ ($^\circ$)",
        },
        "decimals": {
            "vasp_angle_deg": 2,
            "uma_angle_deg": 2,
            "difference_deg": 2,
        },
        "caption": "Bond angles of relaxed gas-phase structures obtained using VASP and UMA.",
        "label": "tab:dissolved_species_angles",
    },
}


# ============================================================================
# Chemical formula formatting
# ============================================================================

def format_formula(formula):
    """
    Convert a chemical formula into LaTeX.

    Examples
    --------
    CrF2   -> \\ce{CrF2}
    CrO3   -> \\ce{CrO3}
    CrO4-- -> \\ce{CrO4^{2-}}
    CrO4-  -> \\ce{CrO4^{-}}
    """

    formula = str(formula)

    # Extract charge at the end.
    match = re.search(r"([+-]+)$", formula)

    charge = ""

    if match:
        charge_string = match.group(1)
        formula = formula[:match.start()]

        sign = charge_string[0]
        magnitude = len(charge_string)

        if magnitude == 1:
            charge = rf"^{{{sign}}}"
        else:
            charge = rf"^{{{magnitude}{sign}}}"

    return rf"\ce{{{formula}{charge}}}"


# ============================================================================
# Loading
# ============================================================================

def load_npy(path):
    """Load a .npy file containing dictionaries."""

    data = np.load(path, allow_pickle=True)

    if isinstance(data, np.ndarray):
        data = data.tolist()

    if isinstance(data, dict):
        data = [data]

    return data


# ============================================================================
# Formatting
# ============================================================================

def format_value(key, value, decimals):
    """Format a value for LaTeX."""

    if key == "bond_type":
        return str(value).replace("-", "--", 1)

    if key == "angle_type":
        match = re.fullmatch(
            r"([^-]+)-\(\s*'([^']+)'\s*,\s*'([^']+)'\s*\)",
            str(value),
        )
        if match:
            center, first, second = match.groups()
            return f"{first}--{center}--{second}"

    if isinstance(value, (float, np.floating)):
        n = decimals.get(key, 3)
        return f"{value:.{n}f}"

    if isinstance(value, (int, np.integer)):
        return str(value)

    return str(value)


def generate_latex_table(
    data,
    columns,
    decimals,
    caption=None,
    label=None,
):
    """Generate a LaTeX table."""

    column_keys = list(columns.keys())

    # Text columns are left-aligned; measurement columns are right-aligned.
    text_columns = {"species", "bond_type", "angle_type"}
    alignment = "".join(
        "l" if key in text_columns else "r"
        for key in column_keys
    )

    lines = []

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"    \centering")
    lines.append(rf"    \begin{{tabular}}{{{alignment}}}")
    lines.append(r"         \hline")

    # Header
    headers = [columns[key] for key in column_keys]

    lines.append("         " + " & ".join(headers) + r" \\")
    lines.append(r"         \hline")

    # Rows
    for row in data:

        values = []

        for key in column_keys:

            value = row[key]

            if key == "species":
                value = format_formula(value)
            else:
                value = format_value(key, value, decimals)

            values.append(value)

        lines.append("         " + " & ".join(values) + r" \\")

    lines.append(r"         \hline")
    lines.append(r"    \end{tabular}")

    if caption is not None:
        lines.append(rf"    \caption{{{caption}}}")

    if label is not None:
        lines.append(rf"    \label{{{label}}}")

    lines.append(r"\end{table}")

    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main():

    for filename, config in TABLE_CONFIG.items():

        path = DATA_DIR / filename

        if not path.exists():
            print(f"Skipping {filename}: file not found")
            continue

        print(f"Processing {filename}")

        data = load_npy(path)

        if not data:
            print("  Empty file, skipping")
            continue

        # Check that the requested columns actually exist.
        first_row = data[0]

        missing = [
            key
            for key in config["columns"]
            if key not in first_row
        ]

        if missing:
            print("  ERROR: requested columns not found:")
            for key in missing:
                print(f"    {key}")

            print("\n  Available keys:")
            for key in first_row:
                print(f"    {key}")

            continue

        latex = generate_latex_table(
            data=data,
            columns=config["columns"],
            decimals=config["decimals"],
            caption=config["caption"],
            label=config["label"],
        )

        output_path = path.with_suffix(".tex")

        output_path.write_text(
            latex,
            encoding="utf-8",
        )

        print(f"  -> {output_path}")


if __name__ == "__main__":
    main()
