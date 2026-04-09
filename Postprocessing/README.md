# OVITO Trajectory Analysis Utilities

This folder contains three Python scripts for post-processing atomistic trajectory files with [OVITO](https://www.ovito.org/). The tools are designed for corrosion and surface analyses from MD trajectories, with outputs saved as compressed NumPy (`.npz`) files for downstream plotting and analysis.

## Included scripts

- `Get_atomic_density.py`
  Computes the areal density of dissolved atoms above a user-defined or automatically detected surface height.
- `Get_OH_analysis.py`
  Classifies oxygen-containing species as `O`, `OH`, or `H2O` from O-H coordination analysis.
- `Get_surface_area.py`
  Computes the relative surface area change of a selected set of atoms using OVITO's Gaussian-density surface construction.

---

## Requirements

### Python packages

Install the required packages in your environment:

```bash
pip install numpy ovito tqdm
```

### Input data

All scripts read **OVITO-compatible trajectory files**, for example:

- `.xyz`
- `.dump`
- `.lammpstrj`
- other formats supported by `ovito.io.import_file`

---

## Repository layouts

A minimal layout could look like this:

```text
repo/
├── README.md
├── Get_atomic_density.py
├── Get_OH_analysis.py
├── Get_surface_area.py
└── data/
    └── trajectory files
```

---

## 1. `Get_atomic_density.py`

### Purpose

This script calculates the **areal density of selected elements above a height threshold** as a function of time. It is useful for tracking dissolution or lift-out of atoms from a surface.

### What it does

- reads a trajectory with OVITO
- identifies particle type IDs from element names
- counts atoms with `z > z_height`
- supports one or multiple target elements
- optionally auto-detects the top surface height from the first frame and adds a buffer
- saves all element densities into one `.npz` file

### Command-line usage

```bash
python Get_atomic_density.py --filename TRAJ_FILE --output OUTPUT_NAME --elem ELEMENT [ELEMENT ...]
```

### Arguments

- `--filename`, `-f` : input trajectory file
- `--output`, `-o` : output file name without extension
- `--elem`, `-e` : one or more element names, for example `Cr` or `Fe Ni Cr`
- `--z-height`, `-z` : manual height threshold
- `--verbose`, `-v` : print runtime configuration
- `--timestep`, `-t` : placeholder argument currently parsed but not used in the calculation

### Examples

Single-element analysis:

```bash
python Get_atomic_density.py \
  --filename data/traj.nvt \
  --output density_Cr \
  --elem Cr \
  --z-height 5.0
```

Multi-element analysis:

```bash
python Get_atomic_density.py \
  --filename data/traj.nvt \
  --output density_all \
  --elem Fe Ni Cr \
  --z-height 5.0
```

Auto-detect the surface height:

```bash
python Get_atomic_density.py \
  --filename data/traj.nvt \
  --output density_auto \
  --elem Fe Ni Cr
```

### Output

The script saves:

```text
OUTPUT_NAME.npz
```

The `.npz` file contains:

- `time`
- `time_factor`
- `z_height`
- `z_height_min`
- `z_height_max`
- `elements`
- `density_<ELEMENT>` for each requested element

Example loading:

```python
import numpy as np

data = np.load("density_all.npz", allow_pickle=True)
time = data["time"]
elements = data["elements"]
density_cr = data["density_Cr"]
```

### Notes

- The time axis is written as `frame_index * 0.1`, since the script currently calls the main routine with `time_factor=0.1`.
- Auto-detection uses the maximum `z` position of the selected element(s) in frame 0 and adds a buffer of `1.8 Å`.
- Although the help text mentions `--z-height None`, the argument is defined as `float`. In practice, auto-detection is triggered by **omitting** `--z-height`.

---

## 2. `Get_OH_analysis.py`

### Purpose

This script analyzes oxygen speciation based on the number of hydrogen neighbors within a cutoff distance. It classifies oxygen atoms into:

- `O` : 0 H neighbors
- `OH` : 1 H neighbor
- `H2O` : 2 H neighbors

### What it does

- reads an OVITO-compatible trajectory
- detects particle IDs for `O` and `H`
- removes all atoms except oxygen and hydrogen
- performs coordination analysis with a user-defined cutoff
- computes frame-by-frame fractions of `O`, `OH`, and `H2O`
- saves the results to an `.npz` file

### Command-line usage

```bash
python Get_OH_analysis.py TRAJ_FILE [--cutoff CUTOFF] [--output OUTPUT] [--system-name NAME]
```

### Arguments

- `file` : input trajectory file
- `-c`, `--cutoff` : O-H bond cutoff distance in Å, default `1.2`
- `-o`, `--output` : output `.npz` file name, default `oxygen_species_analysis.npz`
- `--system-name` : optional label stored in the output

### Examples

Default analysis:

```bash
python Get_OH_analysis.py data/trajectory.xyz
```

Custom cutoff:

```bash
python Get_OH_analysis.py data/trajectory.dump --cutoff 1.3
```

Custom output name:

```bash
python Get_OH_analysis.py data/trajectory.xyz -o species_analysis.npz
```

### Output

The saved `.npz` file contains:

- `system_name`
- `cutoff`
- `O_id`
- `H_id`
- `steps`
- `O_frac`
- `OH_frac`
- `H2O_frac`

Example loading:

```python
import numpy as np

data = np.load("oxygen_species_analysis.npz")
steps = data["steps"]
o_frac = data["O_frac"]
oh_frac = data["OH_frac"]
h2o_frac = data["H2O_frac"]
```

### Notes

- The classification is based on the OVITO coordination number of each oxygen atom.
- The script exits with an error if the trajectory does not contain particle types named exactly `O` and `H`.

---

## 3. `Get_surface_area.py`

### Purpose

This script calculates the **surface area change** of selected elements over time using OVITO's `ConstructSurfaceModifier` with the Gaussian-density method.

### What it does

- loads one or more trajectories
- keeps only the requested element types
- wraps periodic images
- constructs a surface mesh using Gaussian density
- samples every 10th frame
- computes relative surface area change
- saves one `.npz` file per trajectory

### Command-line usage

```bash
python Get_surface_area.py --files TRAJ1 [TRAJ2 ...] [--labels LABEL1 LABEL2 ...] [--keep Fe Ni Cr]
```

### Arguments

- `-f`, `--files` : one or more input trajectory files
- `-l`, `--labels` : optional labels for output files
- `--keep` : element names to keep, default `Fe Ni Cr`
- `--grid-res` : grid resolution for surface construction, default `300`
- `--radius-scale` : Gaussian radius scaling, default `0.92`
- `--isolevel` : surface isolevel, default `1.0`

### Examples

Single trajectory:

```bash
python Get_surface_area.py --files data/traj.nvt
```

Custom kept elements:

```bash
python Get_surface_area.py --files data/traj.nvt --keep Fe Cr
```

Multiple trajectories with labels:

```bash
python Get_surface_area.py \
  --files data/traj1.nvt data/traj2.nvt \
  --labels case1 case2
```

### Output

For each trajectory, the script saves:

```text
<label>_surface_data.npz
```

The `.npz` file contains:

- `steps`
- `surface_areas`
- `label`

Example loading:

```python
import numpy as np

data = np.load("case1_surface_data.npz")
steps = data["steps"]
surface_areas = data["surface_areas"]
label = data["label"]
```

### Notes

- Only every 10th frame is processed.
- The reported value is the percent change derived from the constructed surface area and the simulation cell cross-sectional area in the `xy` plane.
- `tqdm` is used for the progress bar.

---

## Example plotting snippet

```python
import numpy as np
import matplotlib.pyplot as plt

# Example: plot dissolved Cr density
adata = np.load("density_all.npz", allow_pickle=True)
plt.plot(adata["time"], adata["density_Cr"])
plt.xlabel("Time")
plt.ylabel("Areal density of dissolved Cr")
plt.tight_layout()
plt.show()
```

---

## Citation / acknowledgment

TODO.
