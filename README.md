# Antarctic GlaDS emulation

Tim Hill, 2025 (tim_hill_2@sfu.ca) | https://github.com/timghill/antarctic-glads

## Description

The project structure is:

 * `utils/`: shared code for setting up experiments and analyzing outputs
 * `issm/`: individual directories for GlaDS-ISSM model runs
 * `analysis/`: Emulator fitting, evaluation, and all analysis

Each directory has a README file to describe the contents.

## Installation

The analysis source code has been tested against python 3.11.10. Package requirements are listed in `requirements.txt`, and it is recommended to use a virtual environment to manage versions. For example

```
virtualenv --python 3.11 pyenv/
source pyenv/bin/activate
pip install -r requirements.txt
```

To install the code for this project on your python path, install in editable (`-e`) mode with pip from the project root directory:

```
pip install -e .
```

Your python environment and installation can be verified by running `test_install.sh`. This script should run with no errors and should update several figures in `experiments/synthetic/analysis/figures/`.

## Usage

### Setup for a new basin

#### Domain outline
Making a mesh and construct inputs for a new basin is mostly automated, with some manual intervention to define the ice front:

 1. Move into `data/ANT_Basins` and copy one of the 'read_basins_manual_*.py` scripts, naming according to the new basin name.
 2. Change the region variable, e.g., `region = 'B-C'` and change the filename at the end of the script.
 3. Comment out any `outline = np.delete` and `outline.np.insert` lines and run the code. A plot should come up with the grounding line and ice front vertices enumerated.
 4. Zoom into your region and find any large ice shelves. Use `np.delete` to remove grounding line vertices where there is an ice shelf and use `np.insert` to insert the appropriate range of ice front vertices.
 5. Check your outline and make sure it has been saved to the right file.

#### GlaDS and ISSM mesh
Now setup the GlaDS and ISSM runs. From `issm/`, copy one of the experiment directories, excluding subdirectories. For example:
```
mkdir Jpp-K
cp F-G/* Jpp-K/
```
In `data/geom/generate_outline.py`, modify the filename for the basin outline to load the correct outline. Run this script:
```
python generate_outline.py
```
Then make the mesh. You shouldn't have to change any variables in the meshing script:
```
python make_mesh.py
```
Check the figures produced to make sure the mesh and domain geometry (elevation, thickness) look as expected.

#### GlaDS inputs
To make the basal sliding velocity and melt rate fields, move into `data/lanl-mali` and run
```
python interp_mali.py
```
This script pulls from the `data/geom` directory so you shouldn't have to modify anything. Check the figures. If the mesh and input scripts have run with no errors, you should now be able to run GlaDS and ISSM.

#### Run GlaDS

You can submit the whole ensemble using
```
submit.run N
```
replacing `N` with the number of jobs to use for the ensemble of 100 runs. You can use N=100 to run the ensemble quickly.

Or, you can run an individual case (e.g., for a test) by running
```
python -m utils.glads.run_job.py ../train_config.py jobid
```
replacing `jobid` with the number (e.g., `1`)

#### Run ISSM
You can run the ISSM inversion for friction coefficient by submitting
```
sbatch runme.sh
```
which will run the inversion and make a few basic plots.




