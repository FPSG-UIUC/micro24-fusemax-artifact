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

Install Accelergy plug-ins:
```bash
cd src/accelergy-library-plug-in
pip install .
```

Copy the custom Accelergy tables:
```
cd ../../..
cp -r custom_pc_2021 env/share/accelergy/estimation_plug_ins/accelergy-library-plugin/library
```

### Step 3: Check Installation

Check that all imports work as desired.

```bash
cd src
python utils/check.py
cd ..
```

The output should be
```bash
Imports OK
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

Open the Jupyter Lab in the browser and run `notebooks/figs.ipynb`. All figures
will display in the notebook.

### Option 2: Run from command line

```bash
cd src
python run.py
cd ..
```

All figures can be found in `data/generated/figs/`.
