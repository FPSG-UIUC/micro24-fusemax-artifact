# MICRO24 FuseMax Artifact

This repository provides the evaluation setups for the MICRO24 artifact
evaluation for the paper *FuseMax: Leveraging Extended Einsums to Optimize
Attention Accelerator Design*. We provide both a Jupyter lab and script-based
installation options for the artifact evaluation.

## System Requirements

### Software

- Ubuntu 20.04 with `sudo` access
- Python 3.8

## Installation

These instructions use a Python virtual environment. The use of other Python
environments (e.g., `conda`) may change the required paths.

### Step 0: Clone the repository

Submodules must also be recursively cloned.

```bash
git clone --recurse-submodules git@github.com:FPSG-UIUC/micro24-fusemax-artifact.git
cd micro24-fusemax-artifact
```

### Step 1: Create the virtual environment

```bash
python -m venv env
source env/bin/activate
```

### Step 2: Install dependencies

Install prerequisites:

```bash
pip install -r requirements.txt
sudo apt-get install libboost-all-dev=1.71.0.0ubuntu2
```

Install Timeloop:
```bash
cd accelergy-timeloop-infrastructure
make install_accelergy
pip install ./src/timeloopfe
make install_timeloop
```

More information about Timeloop/Accelergy can be found
[here](https://github.com/Accelergy-Project/accelergy-timeloop-infrastructure/),
including installation instructions
[here](https://timeloop.csail.mit.edu/v4/installation).

Install Accelergy library plug-in and copy the custom Accelergy tables:
```bash
cd src/accelergy-library-plug-in
pip install .
cd ../../..
cp -r custom_pc_2021 env/share/accelergy/estimation_plug_ins/accelergy-library-plugin/library
```

More information about the Accelergy library plug-in can be found [here](https://github.com/Accelergy-Project/accelergy-library-plug-in).

### Step 3: Check Installation

Check that all imports work as desired.

```bash
cd src
python utils/check.py
cd ..
```

Note: Because paths are relative, the check *must* be run inside the `src` directory.

The output should be
```bash
Imports OK
input file: /path/to/micro24-fusemax-artifact/data/generated/check/timeloop/parsed-processed-input.yaml
execute:/path/to/micro24-fusemax-artifact/env/bin/accelergy /path/to/micro24-fusemax-artifact/data/generated/check/timeloop/parsed-processed-input.yaml --oprefix timeloop-model. -o ./ > timeloop-model.accelergy.log 2>&1
Utilization = 1.00 | pJ/Compute =    0.267
Timeloop OK
Accelergy Area OK
Accelergy Energy OK
```

## Run Experiments

### Option 1: Use Jupyter Lab

Start Jupyter Lab:
```bash
jupyter lab
```

Open the Jupyter Lab in the browser and run `notebooks/figs.ipynb`. The same
checks as **Step 3** above can be run from `notebooks/check.ipynb`. All figures
will display in the notebook.

### Option 2: Run from command line

```bash
cd src
python run.py
cd ..
```

All figures can be found in `data/generated/figs/`.
