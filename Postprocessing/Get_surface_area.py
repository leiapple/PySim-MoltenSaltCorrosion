#!/usr/bin/env python3
"""
OVITO surface area analysis - keeps only specified elements and calculates surface area changes.
"""

import argparse
import os
import sys

import numpy as np
from ovito.io import import_file
from ovito.modifiers import (
    ConstructSurfaceModifier,
    DeleteSelectedModifier,
    SelectTypeModifier,
    WrapPeriodicImagesModifier,
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
    for t in type_property.types:
        if t.name == target_name:
            return t.id
    raise Exception(
        f"Element '{target_name}' not found. "
        f"Available types: {[t.name for t in type_property.types]}"
    )


def get_ids_to_remove(pipeline, elements_to_keep):
    """
    Get IDs of all elements except those to keep.

    Parameters
    ----------
    pipeline : ovito.pipeline.Pipeline
        OVITO pipeline with loaded data
    elements_to_keep : list
        List of element names to keep (e.g., ['Fe', 'Ni', 'Cr'])

    Returns
    -------
    set
        Set of type IDs to remove
    """
    data = pipeline.compute(0)
    type_property = data.particles.particle_types

    all_types = {t.id for t in type_property.types}
    keep_ids = {get_id_from_name(type_property, elem) for elem in elements_to_keep}
    remove_ids = all_types - keep_ids

    print(f"  Keeping: {elements_to_keep} (IDs: {keep_ids})")
    print(f"  Removing: {[t.name for t in type_property.types if t.id in remove_ids]}")

    return remove_ids


def process_trajectory(
    data_path,
    label=None,
    elements_to_keep=["Fe", "Ni", "Cr"],
    grid_resolution=300,
    radius_scaling=0.92,
    isolevel=1.0,
):
    """
    Process trajectory and calculate surface area changes.

    Parameters
    ----------
    data_path : str
        Path to trajectory file
    label : str, optional
        Label for output files
    elements_to_keep : list
        Element names to keep (others removed)
    grid_resolution : int
        Grid resolution for surface construction
    radius_scaling : float
        Radius scaling for Gaussian density method
    isolevel : float
        Isolevel for surface determination

    Returns
    -------
    steps, surface_areas : tuple of numpy.ndarray
        Frame indices and surface area change percentages
    """
    print(f"\nProcessing: {data_path}")

    if label is None:
        label = os.path.splitext(os.path.basename(data_path))[0]

    # Setup pipeline
    pipeline = import_file(data_path)
    pipeline.modifiers.append(WrapPeriodicImagesModifier())

    # Remove unwanted elements
    remove_ids = get_ids_to_remove(pipeline, elements_to_keep)
    if remove_ids:
        pipeline.modifiers.append(SelectTypeModifier(types=remove_ids))
        pipeline.modifiers.append(DeleteSelectedModifier())

    # Construct surface
    pipeline.modifiers.append(
        ConstructSurfaceModifier(
            method=ConstructSurfaceModifier.Method.GaussianDensity,
            transfer_properties=True,
            grid_resolution=grid_resolution,
            radius_scaling=radius_scaling,
            isolevel=isolevel,
            identify_regions=True,
        )
    )

    # Collect data
    from tqdm import tqdm

    steps, surface_areas = [], []

    num_frames = pipeline.source.num_frames

    # Process every 10th frame with progress bar
    for frame_idx in tqdm(range(0, num_frames, 10), desc="Processing frames"):
        data = pipeline.compute(frame_idx)
        area = data.attributes.get("ConstructSurfaceMesh.surface_area")
        if area is not None:
            cell_area = data.cell[0][0] * data.cell[1][1]
            percent_change = 100 * ((area - cell_area) / cell_area - 1)
            steps.append(frame_idx)
            surface_areas.append(percent_change)

    steps = np.array(steps)
    surface_areas = np.array(surface_areas)

    print(
        f"Processed {len(steps)} frames (every 10th frame out of {len(pipeline.frames)} total)"
    )

    steps = np.array(steps)
    surface_areas = np.array(surface_areas)

    # Summary
    print(f"  Frames: {len(steps)}")
    print(f"  Mean change: {np.mean(surface_areas):.2f} ± {np.std(surface_areas):.2f}%")

    # Save data
    filename = f"{label.lower()}_surface_data.npz"
    np.savez(filename, steps=steps, surface_areas=surface_areas, label=label)
    print(f"  Saved: {filename}")

    return steps, surface_areas


def main():
    parser = argparse.ArgumentParser(description="OVITO surface area analysis")

    # Input
    parser.add_argument(
        "-f", "--files", nargs="+", required=True, help="Trajectory file(s) to process"
    )
    parser.add_argument("-l", "--labels", nargs="+", help="Labels for output files")
    parser.add_argument(
        "--keep",
        nargs="+",
        default=["Fe", "Ni", "Cr"],
        help="Elements to keep (default: Fe Ni Cr)",
    )

    # Surface parameters
    parser.add_argument(
        "--grid-res", type=int, default=300, help="Grid resolution (default: 300)"
    )
    parser.add_argument(
        "--radius-scale",
        type=float,
        default=0.92,
        help="Radius scaling (default: 0.92)",
    )
    parser.add_argument(
        "--isolevel", type=float, default=1.0, help="Isolevel (default: 1.0)"
    )

    args = parser.parse_args()

    # Validate labels
    if args.labels and len(args.labels) != len(args.files):
        print("Error: Number of labels must match number of files")
        sys.exit(1)

    # Process files
    for i, file_path in enumerate(args.files):
        label = args.labels[i] if args.labels else None
        process_trajectory(
            file_path, label, args.keep, args.grid_res, args.radius_scale, args.isolevel
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
