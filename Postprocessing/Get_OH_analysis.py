import argparse
import sys

import numpy as np
from ovito.io import import_file
from ovito.modifiers import (
    CoordinationAnalysisModifier,
    DeleteSelectedModifier,
    ExpressionSelectionModifier,
)


def get_id_from_name(type_property, target_name):
    """
    Retrieve the numeric particle type ID from its name/chemical symbol.

    Parameters
    ----------
    type_property : ovito.data.ParticleTypesProperty
        OVITO particle types property containing type definitions
    target_name : str
        Name of the element/type to look up (e.g., "Cu", "Fe", "O")

    Returns
    -------
    int
        Numeric ID associated with the target particle type

    Raises
    ------
    Exception
        If the target_name is not found in the available particle types
    """
    # Iterate through all defined particle types
    for t in type_property.types:
        if t.name == target_name:
            return t.id

    # If we reach here, the element wasn't found
    raise Exception(
        f"Element '{target_name}' not found in particle types. "
        f"Available types: {[t.name for t in type_property.types]}"
    )


def process_system(file_path, system_name, O_id, H_id, cutoff=1.2):
    """
    Process a single system to calculate oxygen species percentages.

    Parameters:
    -----------
    file_path : str
        Path to the input file
    system_name : str
        Name of the system (for output)
    O_id : int
        Particle type ID for oxygen
    H_id : int
        Particle type ID for hydrogen
    cutoff : float
        Cutoff distance for O-H bonds (default: 1.2 Å)

    Returns:
    --------
    dict
        Dictionary containing steps and fractions for O, OH, and H2O
    """
    print(f"\nProcessing {system_name} system from {file_path}...")

    # Import the file
    pipeline = import_file(file_path)

    # Create a copy of the pipeline for processing
    from ovito.pipeline import Pipeline

    pipeline = Pipeline(source=pipeline.source)

    # Step 1: Select all particles except O and H, then delete them
    pipeline.modifiers.append(
        ExpressionSelectionModifier(
            expression=f"ParticleType != {O_id} && ParticleType != {H_id}"
        )
    )
    pipeline.modifiers.append(DeleteSelectedModifier())

    # Step 2: Add coordination analysis to find O-H bonds
    pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=cutoff))

    # Lists to store results
    steps = []
    O_frac = []
    OH_frac = []
    H2O_frac = []

    # Process each frame
    num_frames = pipeline.source.num_frames
    for frame in range(num_frames):
        data = pipeline.compute(frame)

        # Get particle properties
        types = data.particles["Particle Type"][...]
        coordination = data.particles["Coordination"][...]

        # Count oxygen atoms with different coordination numbers
        oxygen_mask = types == O_id
        O_count = np.sum((coordination == 0) & oxygen_mask)
        OH_count = np.sum((coordination == 1) & oxygen_mask)
        water_count = np.sum((coordination == 2) & oxygen_mask)
        total_o_atoms = np.sum(oxygen_mask)

        # Calculate percentages (avoid division by zero)
        if total_o_atoms > 0:
            steps.append(frame)
            O_frac.append(O_count / total_o_atoms)
            OH_frac.append(OH_count / total_o_atoms)
            H2O_frac.append(water_count / total_o_atoms)

        # Print progress every 10 frames
        if (frame + 1) % 10 == 0 or frame == num_frames - 1:
            print(f"  Processed frame {frame+1}/{num_frames}")

    print(f"Completed {system_name} system: {len(steps)} frames processed")
    if len(steps) > 0:
        print(f"  - O (0 neighbors): {np.mean(O_frac)*100:.1f}% average")
        print(f"  - OH (1 neighbor): {np.mean(OH_frac)*100:.1f}% average")
        print(f"  - H2O (2 neighbors): {np.mean(H2O_frac)*100:.1f}% average")

    return {
        "steps": np.array(steps),
        "O_frac": np.array(O_frac),
        "OH_frac": np.array(OH_frac),
        "H2O_frac": np.array(H2O_frac),
    }


def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description="Analyze oxygen species (O, OH, H2O) percentages from MD trajectory files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_oxygen_species.py trajectory.xyz
  python analyze_oxygen_species.py trajectory.dump --cutoff 1.3
  python analyze_oxygen_species.py trajectory.xyz -o results.npz
        """,
    )

    parser.add_argument(
        "file", help="Path to the trajectory file (e.g., .xyz, .dump, .lammpstrj)"
    )
    parser.add_argument(
        "-c",
        "--cutoff",
        type=float,
        default=1.2,
        help="Cutoff distance for O-H bonds in Angstroms (default: 1.2)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="oxygen_species_analysis.npz",
        help="Output filename for NPZ file (default: oxygen_species_analysis.npz)",
    )
    parser.add_argument(
        "--system-name",
        default=None,
        help="Name of the system (if not provided, extracted from filename)",
    )

    args = parser.parse_args()

    # Import the file to get type information
    print(f"Loading file: {args.file}")
    temp_pipeline = import_file(args.file)

    # Get O and H IDs from the first frame
    data = temp_pipeline.compute(0)
    type_property = data.particles["Particle Type"]

    try:
        O_id = get_id_from_name(type_property, "O")
        H_id = get_id_from_name(type_property, "H")
        print(f"Found oxygen (ID: {O_id}) and hydrogen (ID: {H_id})")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your trajectory contains particles named 'O' and 'H'")
        sys.exit(1)

    # Determine system name
    if args.system_name:
        system_name = args.system_name
    else:
        # Extract system name from filename (remove extension and path)
        import os

        system_name = os.path.splitext(os.path.basename(args.file))[0]

    # Process the system
    results = process_system(args.file, system_name, O_id, H_id, args.cutoff)

    # Save results to NPZ file
    print(f"\nSaving data to {args.output}...")
    np.savez(
        args.output,
        system_name=system_name,
        cutoff=args.cutoff,
        O_id=O_id,
        H_id=H_id,
        steps=results["steps"],
        O_frac=results["O_frac"],
        OH_frac=results["OH_frac"],
        H2O_frac=results["H2O_frac"],
    )

    print(f"Data saved successfully!")

    # Print summary statistics
    if len(results["steps"]) > 0:
        print("\n" + "=" * 50)
        print("SUMMARY STATISTICS")
        print("=" * 50)
        print(f"System: {system_name}")
        print(f"Cutoff distance: {args.cutoff} Å")
        print(f"Frames analyzed: {len(results['steps'])}")
        print("\nAverage fractions:")
        print(
            f"  O  (0 neighbors): {np.mean(results['O_frac'])*100:6.2f}% ± {np.std(results['O_frac'])*100:5.2f}%"
        )
        print(
            f"  OH (1 neighbor):  {np.mean(results['OH_frac'])*100:6.2f}% ± {np.std(results['OH_frac'])*100:5.2f}%"
        )
        print(
            f"  H2O (2 neighbors):{np.mean(results['H2O_frac'])*100:6.2f}% ± {np.std(results['H2O_frac'])*100:5.2f}%"
        )
        print("=" * 50)


if __name__ == "__main__":
    main()
