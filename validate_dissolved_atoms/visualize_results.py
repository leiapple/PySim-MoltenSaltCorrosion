import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    from evaluate_results import find_relaxed_structure
    from IPython.display import display
    from ase.visualize import view
    from ase.io import read
    from pathlib import Path
    from ase import Atoms
    import numpy as np
    import pandas as pd
    import myplots as p
    p.use_style(doc_fontsize=18)
    return Atoms, Path, display, find_relaxed_structure, np, p, pd, read, view


@app.cell
def _(Atoms, Path, display, find_relaxed_structure, read, view):
    def visualize_atoms(species_name: str):
        print(f"Visualizing atoms for species: {species_name}")
        species_dir = Path().cwd() / "calculations" / species_name

        vasp_path = find_relaxed_structure(species_dir / "vasp")
        uma_path = find_relaxed_structure(species_dir / "uma")

        vasp_atoms: Atoms = read(vasp_path)  # type: ignore
        uma_atoms: Atoms = read(uma_path)  # type: ignore

        print("UMA:")
        display(view(uma_atoms, viewer="x3d"))
        print("VASP:")
        display(view(vasp_atoms, viewer="x3d"))
        print("COMBINED:")
        combined = vasp_atoms + uma_atoms
        display(view(combined, viewer="x3d"))

    visualize_atoms("CrO4--")
    return


@app.cell
def _(Path, np, pd):
    # Load the stats from the evaluation
    CALCULATIONS_DIR = Path(__file__).parent / "calculations"
    INPUT_DIR = CALCULATIONS_DIR / "comparison"
    E_form = np.load(INPUT_DIR / "formation_energies.npy", allow_pickle=True)
    bond_lengths = np.load(INPUT_DIR / "bond_lengths.npy", allow_pickle=True)
    bond_angles = np.load(INPUT_DIR / "bond_angles.npy", allow_pickle=True)

    E_form_df = pd.DataFrame(E_form.tolist())
    bond_lengths_df = pd.DataFrame(bond_lengths.tolist())
    bond_angles_df = pd.DataFrame(bond_angles.tolist())
    E_form_df
    return E_form_df, bond_angles_df, bond_lengths_df


@app.cell
def _(E_form_df, bond_angles_df, bond_lengths_df, p, pd):
    fig, ax = p.new(2, 2, figsize=(8, 8))

    p.plot_parity(ax[0], E_form_df["vasp_formation_energy_eV"], E_form_df["uma_formation_energy_eV"], add_rmse=True, rmse_unit=" eV", rmse_loc=(0.05, 0.9), rmse_fontsize=10)
    p.label(ax[0], f"VASP {p.bm('E')}$_{{formation}}$ (eV)", f"UMA {p.bm('E')}$_{{formation}}$ (eV)", "Formation Energy")
    p.plot_parity(ax[1], bond_lengths_df["vasp_length_A"], bond_lengths_df["uma_length_A"], add_rmse=True, rmse_unit = " $\\AA$", rmse_loc=(0.05, 0.9), rmse_fontsize=10)
    p.label(ax[1], "VASP Bond Length ($\\AA$)", "UMA Bond Length ($\\AA$)", "Bond Lengths")
    p.plot_parity(ax[2], bond_angles_df["vasp_angle_deg"], bond_angles_df["uma_angle_deg"], add_rmse=True, rmse_unit = " deg", rmse_loc=(0.05, 0.9), rmse_fontsize=10)
    p.label(ax[2], "VASP Bond Angle (deg)", "UMA Bond Angle (deg)", "Bond Angles")
    p.annotate_subplots(ax, add_before="(", add_after=")")

    # Annotate a chosen set of formation energies
    selected = pd.concat([
        E_form_df.iloc[0:3],
        E_form_df.iloc[10:],
    ])
    for species, E_form_vasp, E_form_uma in selected[
        ["species", "vasp_formation_energy_eV", "uma_formation_energy_eV"]
    ].itertuples(index=False):
    
        # Convert chemical formula to LaTeX/mathtext
        label = (
            species
            .replace("O3", "O$_3$")
            .replace("O4", "O$_4$")
            .replace("O2", "O$_2$")
            .replace("F2", "F$_2$")
            .replace("F3", "F$_3$")
            .replace("--", "$^{2-}$")
            .replace("F-", "F$^{-}$")
        )

        ax[0].annotate(
            label,
            xy=(E_form_vasp, E_form_uma),
            xytext=(E_form_vasp - 2.5, E_form_uma + 0.5 if species != "FeO2" else E_form_uma + 2.5),
            fontsize=9,
            ha="right",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.1",
                facecolor=p.get_colors()[7],
                edgecolor="0.7",
                alpha=0.7,
            ),
            arrowprops=dict(
                arrowstyle="->",
                color="0.3",
                lw=0.8,
                shrinkA=2,
                shrinkB=4,
                connectionstyle="arc3,rad=-0.25",
            ),
            zorder=100
        )
    p.save(fig, "final_plot.pdf")
    p.show()
    return


if __name__ == "__main__":
    app.run()
