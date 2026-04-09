import numpy as np
from ovito.io import import_file

# ============================================================================
# FUNCTION DEFINITIONS
# ============================================================================


def get_max_height_from_elements(filename, elements, frame_index=0):
    """
    Calculate the maximum z-coordinate (height) of specified elements in a given frame.

    This function reads a trajectory file and finds the maximum height of atoms
    belonging to the specified element types. This is useful for automatically
    determining the surface height or topmost atom positions.

    Parameters
    ----------
    filename : str
        Path to the input trajectory file (supports OVITO-compatible formats)
    elements : str or list of str
        Element symbol(s) or name(s) to analyze. Can be a single string or list of strings.
    frame_index : int, optional
        Frame index to analyze (default: 0, the first frame)

    Returns
    -------
    float
        Maximum z-coordinate (height) of the specified elements in the given frame
    dict
        Additional information including max height per element and number of atoms found
    """
    # Convert single element to list for uniform handling
    if isinstance(elements, str):
        elements = [elements]

    print(
        f"\nAuto-detecting z_height from {', '.join(elements)} in frame {frame_index}"
    )

    # Load the trajectory file using OVITO's import system
    pipeline = import_file(filename)
    # Compute the specified frame
    data = pipeline.compute(frame_index)
    # Extract particle positions and types
    positions = data.particles["Position"]
    types = data.particles["Particle Type"]

    # Get particle type information
    type_property = data.particles.particle_types

    # Get the numeric IDs for the target elements
    element_ids = {}
    for elem in elements:
        try:
            element_ids[elem] = get_id_from_name(type_property, elem)
        except Exception as e:
            print(f"Warning: {e}")
            continue

    if not element_ids:
        raise Exception(
            f"None of the specified elements {elements} found in the trajectory"
        )

    print(f"Element IDs found: {element_ids}")

    # Find maximum heights
    max_height = -np.inf
    element_max_heights = {elem: -np.inf for elem in element_ids.keys()}
    element_counts = {elem: 0 for elem in element_ids.keys()}

    for pos, typ in zip(positions, types):
        for elem, elem_id in element_ids.items():
            if typ == elem_id:
                z = pos[2]
                element_counts[elem] += 1
                if z > element_max_heights[elem]:
                    element_max_heights[elem] = z
                if z > max_height:
                    max_height = z
                break

    # Check if any atoms were found
    total_atoms = sum(element_counts.values())
    if total_atoms == 0:
        raise Exception(f"No atoms found for elements {elements}")

    # Print summary
    print(f"\nMaximum Height Analysis Results:")
    print(f"  Total atoms found: {total_atoms}")
    for elem in elements:
        if elem in element_counts:
            print(
                f"  {elem}: {element_counts[elem]} atoms, max height = {element_max_heights[elem]:.6f}"
            )
        else:
            print(f"  {elem}: Not found in trajectory")
    print(f"\n  Overall maximum height: {max_height:.6f}")

    return max_height, {
        "max_height": max_height,
        "element_max_heights": element_max_heights,
        "element_counts": element_counts,
        "frame_index": frame_index,
        "elements_used": elements,
    }


def get_dissolved_atomic_density(
    filename,
    output_name,
    elem_type,
    z_height,
    z_height_min=None,
    z_height_max=None,
    time_factor=1.0,
):
    """
    Calculate the areal density of dissolved atoms from a molecular dynamics trajectory.
    Supports single or multiple element types. All data is saved in a single file.

    This function reads a trajectory file, counts atoms above a given z-height threshold,
    and computes the areal density (atoms per unit area) for each frame. It can handle:
    - Single element analysis
    - Multiple elements (all saved in the same output file)
    - Z-range filtering (atoms within a specified height range)

    Parameters
    ----------
    filename : str
        Path to the input trajectory file (supports OVITO-compatible formats)
    output_name : str
        Base name for the output .npz file (e.g., "density_data")
    elem_type : str or list of str
        Element symbol(s) or name(s) to analyze. Can be a single string or list of strings.
    z_height : float
        Threshold z-coordinate (in simulation units). Atoms with z > z_height are
        considered "dissolved" (above the surface).
    z_height_min : float, optional
        Minimum z-coordinate for counting (inclusive). If None, no lower bound.
    z_height_max : float, optional
        Maximum z-coordinate for counting (inclusive). If None, no upper bound.
    time_factor : float, optional
        Scaling factor for time frames (default: 1.0). Use to convert frame indices to
        physical time units (e.g., if timestep = 0.002 ps, use time_factor=0.002)

    Returns
    -------
    dict
        Dictionary containing the time and density data for each element.
        Also saves all data to a single .npz file.
    """

    # Convert single element to list for uniform handling
    if isinstance(elem_type, str):
        elem_type = [elem_type]

    print(f"Processing: {filename}")
    print(f"Analyzing elements: {', '.join(elem_type)}")

    # Setup height filtering description
    height_condition = [f"z > {z_height}"]
    if z_height_min is not None:
        height_condition.append(f"z >= {z_height_min}")
    if z_height_max is not None:
        height_condition.append(f"z <= {z_height_max}")
    print(f"Height condition: {' and '.join(height_condition)}")

    # Load the trajectory file using OVITO's import system
    pipeline = import_file(filename)

    # Compute the first frame to access particle type information
    data = pipeline.compute(0)

    # Extract the particle types property (contains type names and IDs)
    type_property = data.particles.particle_types

    # Get the numeric IDs for the target elements
    element_ids = {}
    for elem in elem_type:
        element_ids[elem] = get_id_from_name(type_property, elem)

    print(f"Element IDs: {element_ids}")

    # Initialize data storage - store everything in a single dictionary
    data_dict = {
        "time": [],  # Common time array for all elements
        "density": {elem: [] for elem in elem_type},  # Density arrays for each element
        "metadata": {
            "time_factor": time_factor,
            "z_height": z_height,
            "z_height_min": z_height_min if z_height_min is not None else -1,
            "z_height_max": z_height_max if z_height_max is not None else -1,
            "elements": elem_type,
            "filename": filename,
        },
    }

    # Iterate through all frames in the trajectory
    for i, frame in enumerate(pipeline.frames):
        # Extract particle positions and types for the current frame
        positions = frame.particles["Position"]
        types = frame.particles["Particle Type"]

        # Calculate the cross-sectional area of the simulation box (xy-plane)
        # Assumes orthogonal box with cell vectors [a, 0, 0], [0, b, 0], [0, 0, c]
        area = frame.cell[0][0] * frame.cell[1][1]

        # Initialize counts for this frame
        counts = {elem: 0 for elem in elem_type}

        # Count atoms for each element
        for pos, typ in zip(positions, types):
            # Check height conditions
            height_ok = True
            if not (pos[2] > z_height):
                height_ok = False

            if z_height_min is not None and pos[2] < z_height_min:
                height_ok = False
            if z_height_max is not None and pos[2] > z_height_max:
                height_ok = False

            if height_ok:
                # Check which element this atom belongs to
                for elem, elem_id in element_ids.items():
                    if typ == elem_id:
                        counts[elem] += 1
                        break

        # Store time and densities
        data_dict["time"].append(i * time_factor)
        for elem in elem_type:
            data_dict["density"][elem].append(counts[elem] / area)

    # Convert density lists to numpy arrays
    for elem in elem_type:
        data_dict["density"][elem] = np.array(data_dict["density"][elem])
    data_dict["time"] = np.array(data_dict["time"])

    # Save all data to a single .npz file
    # Prepare the data dictionary for saving
    save_dict = {
        "time": data_dict["time"],
        "time_factor": data_dict["metadata"]["time_factor"],
        "z_height": data_dict["metadata"]["z_height"],
        "z_height_min": data_dict["metadata"]["z_height_min"],
        "z_height_max": data_dict["metadata"]["z_height_max"],
        "elements": np.array(
            data_dict["metadata"]["elements"]
        ),  # Convert to numpy array for saving
    }

    # Add density arrays for each element
    for elem in elem_type:
        save_dict[f"density_{elem}"] = data_dict["density"][elem]

    # Save to file
    np.savez(output_name, **save_dict)

    print(f"\nData saved to {output_name}.npz")
    print(f"Total frames processed: {len(data_dict['time'])}")
    print(f"Elements saved: {', '.join(elem_type)}")

    return data_dict


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


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main execution block: runs the analysis with the configured parameters.
    """
    # ============================================================================
    # CONFIGURATION PARAMETERS
    # ============================================================================
import argparse

if __name__ == "__main__":
    """
    Main execution block: runs the analysis with the configured parameters.
    """
    # ============================================================================
    # COMMAND LINE ARGUMENT PARSING
    # ============================================================================

    parser = argparse.ArgumentParser(
        description="Analyze trajectory data for element dissolution behavior",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single element analysis
  python script.py --filename /path/to/traj.nvt --output NaCl_O2_60 --elem Fe

  # Multiple elements analysis
  python script.py --filename /path/to/traj.nvt --output NaCl_O2_60 --elem Fe Ni Cr

  # With custom z-height threshold
  python script.py --filename /path/to/traj.nvt --output NaCl_O2_60 --elem Fe Ni Cr --z-height 5.0

  # Auto-detect surface (no z-height specified)
  python script.py --filename /path/to/traj.nvt --output NaCl_O2_60 --elem Fe Ni Cr --z-height None
        """,
    )

    # Required arguments
    parser.add_argument(
        "--filename",
        "-f",
        type=str,
        required=True,
        help="Input trajectory file path (required)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output file name for saved data (without extension, .npz will be added) (required)",
    )

    # Element type(s) to analyze
    parser.add_argument(
        "--elem",
        "-e",
        type=str,
        nargs="+",  # Accepts one or more arguments
        required=True,
        help="Element type(s) to analyze (e.g., --elem Fe or --elem Fe Ni Cr)",
    )

    # Height threshold (optional)
    parser.add_argument(
        "--z-height",
        "-z",
        type=float,
        default=None,
        help="Height threshold (z-coordinate) in Angstroms or simulation units. "
        "Atoms with z > z_height are considered 'dissolved' (e.g., above a surface). "
        "Specify 'None' (default) for automated determination of the surface.",
    )

    # Optional additional arguments you might want to add
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Increase output verbosity"
    )

    parser.add_argument(
        "--timestep",
        "-t",
        type=float,
        default=None,
        help="Timestep for analysis (if not specified, will be auto-detected)",
    )

    # Parse arguments
    args = parser.parse_args()

    # ============================================================================
    # CONFIGURATION PARAMETERS
    # ============================================================================

    # Input trajectory file path
    filename = args.filename

    # Output file name for saved data (without extension, .npz will be added)
    output_name = args.output

    # Element type(s) to analyze - can be:
    #   - Single element: "Cr"
    #   - Multiple elements: ["Fe", "Ni", "Cr"]
    # args.elem is already a list if multiple arguments are provided
    elem_type = args.elem if len(args.elem) > 1 else args.elem[0]

    # Height threshold (z-coordinate) in Angstroms or simulation units
    # Atoms with z > z_height are considered "dissolved" (e.g., above a surface)
    z_height = (
        args.z_height
    )  # specify None to apply automated determination of the surface

    # Optional: print configuration if verbose
    if args.verbose:
        print("Configuration:")
        print(f"  Filename: {filename}")
        print(f"  Output name: {output_name}")
        print(f"  Element type(s): {elem_type}")
        print(
            f"  Z-height threshold: {z_height if z_height is not None else 'auto-detect'}"
        )
        print(
            f"  Timestep: {args.timestep if args.timestep is not None else 'auto-detect'}"
        )
        print()

    # Optional: Set minimum and maximum height threshold for dissolved atoms
    # If None, only z_height is used
    z_height_min = None
    z_height_max = None
    # specify trajectory file name

    if z_height == None:
        z_height_buffer = (
            1.8  # Angstrom, the height need for element to fully dissolve in the salt
        )
        max_height, detection_info = get_max_height_from_elements(
            filename, elem_type, frame_index=0
        )
        z_height = max_height + z_height_buffer
        print(
            f"\nAuto-detected z_height set to: {z_height:.6f} (max height + buffer {z_height_buffer})"
        )

    result = get_dissolved_atomic_density(
        filename=filename,
        output_name=output_name,
        elem_type=elem_type,  # List of elements
        z_height=z_height,
        time_factor=0.1,  # time factor to make sure the time unit is ps
    )

    print("\nAnalysis complete!")
    print("\nTo load the data:")
    print("  import numpy as np")
    print("  data = np.load('output.npz', allow_pickle=True)")
    print("  time = data['time']")
    print("  density_Cr = data['density_Cr']")
    print("  elements = data['elements']")
    print("  z_height = data['z_height']")
