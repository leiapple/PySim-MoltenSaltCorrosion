import argparse
import json
from pathlib import Path

import myplots as p
import numpy as np
from ovito.data import CutoffNeighborFinder
from ovito.io import import_file
from ovito.modifiers import (
    DeleteSelectedModifier,
    ExpandSelectionModifier,
    ExpressionSelectionModifier,
    WrapPeriodicImagesModifier,
)
from tqdm import tqdm

p.use_style()


def get_coordination_data(
    filename,
    outfile,
    max_dist,
    min_z,
    max_z,
    every_nth,
    main_element,
    neighbor_elements,
):
    # Load trajectory
    pipeline = import_file(filename)
    pipeline.modifiers.append(WrapPeriodicImagesModifier())  # Wrap periodic images

    # Select only the Cr atoms
    pipeline.modifiers.append(
        ExpressionSelectionModifier(expression='ParticleType == "Cr"')
    )

    # Expand selection to neighbors within max_dist
    pipeline.modifiers.append(ExpandSelectionModifier(cutoff=max_dist))

    # Apply z-filter (keep only atoms with max_z > z > min_z)
    pipeline.modifiers.append(
        ExpressionSelectionModifier(expression=f"Selection && Position.Z > {min_z}")
    )
    pipeline.modifiers.append(
        ExpressionSelectionModifier(expression=f"Selection && Position.Z < {max_z}")
    )

    # Delete everything NOT selected
    pipeline.modifiers.append(ExpressionSelectionModifier(expression="Selection == 0"))

    pipeline.modifiers.append(DeleteSelectedModifier())

    # Get the mapping of the particle types to their IDs
    type_list = pipeline.compute(0).particles["Particle Type"].types
    type_map = {t.name: t.id for t in type_list}

    # Compute the coordination number of the main_element atoms in the melt
    main_type = type_map[main_element]
    Cr_coordination = []

    for frame in tqdm(np.arange(0, pipeline.source.num_frames)[::every_nth]):
        data = pipeline.compute(frame)
        types = data.particles["Particle Type"].array
        finder = CutoffNeighborFinder(max_dist, data)
        frame_result = []
        for i, ptype in enumerate(types):
            if ptype != main_type:
                continue
            atom_surrounding = {
                f"n_{neighbor_element}": 0 for neighbor_element in neighbor_elements
            }
            atom_surrounding["n_Other"]
            for neigh in finder.find(i):
                ntype = types[neigh.index]
                is_other = True
                for neighbor_element in neighbor_elements:
                    if ntype == type_map[neighbor_element]:
                        atom_surrounding[f"n_{neighbor_element}"] += 1
                        is_other = False
                if is_other:
                    atom_surrounding["n_Other"] += 1
            frame_result.append(atom_surrounding)
        Cr_coordination.append(frame_result)
    # Save the results as json
    with open(f"{outfile}", "w") as f:
        json.dump(Cr_coordination, f)


def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description="Analyze the coordination environment of a selected element, e.g., Cr, dissolved in an NaF melt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Get_coordination_env.py trajectory.xyz -e Fe -n O F
  python Get_coordination_env.py trajectory.dump --cutoff 1.9
  python Get_coordination_env.py trajectory1.traj trajectory2.traj --every-nth 10 --min-z 30 --max-z 50 -o results1.json results2.json
        """,
    )

    parser.add_argument(
        "files",
        nargs="+",
        help="Path(s) to the trajectory file(s) (e.g., .traj, .xyz, .dump, .lammpstrj)",
    )
    parser.add_argument(
        "-e",
        "--element",
        type=str,
        default="Cr",
        help="Atomic symbol of the element for which the coordination environment is computed (default: Cr)",
    )
    parser.add_argument(
        "-n",
        "--neighbors",
        nargs="*",
        type=str,
        default=["O", "F"],
        help="Atomic symbol of the elements to compute separately for the coordination sphere (default: O and F)",
    )
    parser.add_argument(
        "-c",
        "--cutoff",
        type=float,
        default=2.0,
        help="Cutoff distance for including other atoms in the coordination sphere in Angstroms (default: 2.0)",
    )
    parser.add_argument(
        "--min-z",
        type=float,
        default=28,
        help="Minimum z-coordinate for the melt region in Angstroms (default: 28)",
    )
    parser.add_argument(
        "--max-z",
        type=float,
        default=50,
        help="Maximum z-coordinate for the melt region in Angstroms (default: 50)",
    )
    parser.add_argument(
        "--every-nth",
        type=int,
        default=1,
        help="Every nth frame is evaluated (default: 1)",
    )
    parser.add_argument(
        "-o",
        "--output",
        nargs="*",
        type=str,
        default=[],
        help="Output filename(s) the json file(s) (default: generated from input filename)",
    )

    args = parser.parse_args()
    print(args.__dict__)
    for i, filename in enumerate(args.files):
        if args.output and i < len(args.output):
            outfile = args.output[i]
        else:
            outfile = Path(filename).stem + "_coord.json"
        get_coordination_data(
            filename,
            outfile,
            args.cutoff,
            args.min_z,
            args.max_z,
            args.every_nth,
            args.element,
            args.neighbors,
        )


if __name__ == "__main__":
    main()
