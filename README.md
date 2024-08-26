# MICRO24 FuseMax Artifact

[![DOI](https://zenodo.org/badge/839495574.svg)](https://zenodo.org/doi/10.5281/zenodo.13377042)

This repository provides the evaluation setups for the MICRO24 artifact
evaluation for the paper *FuseMax: Leveraging Extended Einsums to Optimize
Attention Accelerator Design*. We provide a docker environment and Jupyter
notebook for the artifact evaluation.

## System Requirements

### Hardware

- x86-64 CPU
- 5GB disk space

### Software

- [Docker](https://www.docker.com/products/docker-desktop/)
- Web browser

## Installation

#### Step 0: Clone the repository

Submodules must also be recursively cloned.

```bash
git clone --recurse-submodules git@github.com:FPSG-UIUC/micro24-fusemax-artifact.git
cd micro24-fusemax-artifact
```

### [Recommended] Option 1: Docker

#### Step 1: Prepare your `docker-compose.yaml`

Copy the `docker-compose.yaml.template` file to a new `docker-compose.yaml`


```bash
cp docker-compose.yaml.template docker-compose.yaml
```

Edit the `docker-compose.yaml` with the appropriate `USER_UID` and `USER_GID`.

#### Step 2: Pull the Docker image

We provide two options for obtaining the docker image. Please choose one of the
options listed below.

##### Option 1: Use `docker-compose`

```bash
docker-compose pull
```

##### Option 2: Build the image from source

```bash
cd ./setup/common/accelergy-timeloop-infrastructure
make build-amd64
cd ../../docker/timeloop-accelergy-pytorch
make build-amd64
cd ../../..
```

#### Step 3: Start the container

Start the container, including the Jupyter lab.

```bash
docker-compose up
```


### Option 2: Native Installation

Expected installation time: 20 minutes

Additionally requires:
- Ubuntu 20.04 with `sudo` access
- Python 3.8

Note: These instructions use a Python virtual environment. The use of other
Python environments (e.g., `conda`) may change the required paths.

#### Step 1: Create the virtual environment

If not already available, install `venv`:
```bash
sudo apt-get install python3.8-venv
```

Create the environment:

```bash
python3 -m venv env
source env/bin/activate
```

#### Step 2: Install dependencies

Install prerequisites:

```bash
pip install -r setup/native/requirements.txt
sudo apt-get install libboost-all-dev=1.71.0.0ubuntu2
```

Install Timeloop:
```bash
cd setup/common/accelergy-timeloop-infrastructure
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
cd ../../../../..
cp -r setup/common/custom_pc_2021 env/share/accelergy/estimation_plug_ins/accelergy-library-plugin/library
```

More information about the Accelergy library plug-in can be found
[here](https://github.com/Accelergy-Project/accelergy-library-plug-in).

#### Step 3: Check Installation

Check that all imports work as desired.

```bash
cd workspace/src
python3 scripts/check.py
cd ../..
```

Note: Because paths are relative, the check *must* be run inside the `workspace/src` directory.

The output should be
```bash
Imports OK
input file: /path/to/micro24-fusemax-artifact/workspace/outputs/generated/check/timeloop/parsed-processed-input.yaml
execute:/path/to/micro24-fusemax-artifact/env/bin/accelergy /path/to/micro24-fusemax-artifact/workspace/outputs/generated/check/timeloop/parsed-processed-input.yaml --oprefix timeloop-model. -o ./ > timeloop-model.accelergy.log 2>&1
Utilization = 1.00 | pJ/Compute =    0.267
Timeloop OK
Accelergy Area OK
Accelergy Energy OK
```

#### Step 4: Launch the Juptyer Lab

Start Jupyter Lab:
```bash
jupyter lab
```


## Run Experiments

Expected run time: 9 hours

### [Recommended] Option 1: Use Jupyter Lab

Start the Jupyter lab as described above. Open it in the browser by navigating
to the displayed 127.0.0.1 URL.

Run `workspace/notebooks/figs.ipynb`.  All figures will display in the
notebook. Expected outputs can be found in Figures 6-12 of the paper or in
`workspace/outputs/pregenerated/figs/`.

The installation checks can also be run via `workspace/notebooks/check.ipynb`.

### Option 2: Run from command line

```bash
cd workspace/src
python3 scripts/run.py
cd ../..
```

Note: Because paths are relative, this script *must* be run inside the `src` directory.

Generated figures can be found in `workspace/outputs/generated/default/figs/`.
Expected outputs can be found in Figures 6-12 of the paper or in
`workspace/outputs/pregenerated/figs/`.

## Directory Structure

The following is a guide to the director structure for this repository, with
descriptions accompanying each leaf folder.

```
micro24-fusemax-artifact
├── docker-compose.yaml.template
├── README.md
├── setup
│   ├── common
│   │   ├── accelergy-timeloop-infrastructure
│   │   │   └── <Timeloop / Accelergy Source>
│   │   └── custom_pc_2021
│   │       └── <Custom Accelergy Tables>
│   ├── docker
│   │   └── timeloop-accelergy-pytorch
│   │       └── <Docker Source>
│   └── native
│       └── requirements.txt
└── workspace
    ├── inputs
    │   ├── timeloop-accelergy-exercises
    │   └── yamls
    │       └── <Timeloop / Accelergy Input YAMLs>
    ├── notebooks
    │   ├── check.ipynb
    │   └── figs.ipynb
    ├── outputs
    │   ├── pregenerated
    │   │   ├── figs
    │   │   │   └── <Figures 6-12>
    │   │   └── results
    │   │       └── <Raw CSVs used to generate Figures>
    │   └── pregenerated-check
    │       ├── accelergy_area
    │       │   └── <Expected outputs for Accelergy area check>
    │       ├── accelergy_energy
    │       │   └── <Expected outputs for Accelergy energy check>
    │       └── timeloop
    │           └── <Expected outputs for Timeloop check>
    └── src
        ├── accel
        │   └── <Timeloop / Accelergy models of the various accelerator configurations>
        ├── scripts
        │   └── <Scripts to check installation / run experiments>
        └── utils
            └── <Scripts for reading Timeloop outputs, drawing graphs, etc.>
```
