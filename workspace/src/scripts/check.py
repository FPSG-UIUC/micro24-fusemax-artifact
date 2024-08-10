import filecmp
from pathlib import Path
import subprocess

import timeloopfe.v4 as tl

def imports():
    import matplotlib
    import pandas
    import seaborn
    import ipywidgets

    print("Imports OK")

def timeloop():
    spec = tl.Specification.from_yaml_files(
    "../yamls/proposal/arch-2d.yaml",
    "../yamls/proposal/problems/qk.yaml",
    "../yamls/proposal/mappings/qk.yaml",
    "../timeloop-accelergy-exercises/workspace/example_designs/example_designs/_components/*")

    tl.call_model(spec, "../data/generated/check/timeloop")

    if filecmp.cmp("../data/pregenerated/check/timeloop/timeloop-model.stats.txt",
            "../data/generated/check/timeloop/timeloop-model.stats.txt"):
        print("Timeloop OK")
    else:
        print("Timeloop ERROR")

def accelergy():
    spec = tl.Specification.from_yaml_files(
        "../yamls/area_energy/architecture/accel_fusemax/accel-top-1d-path.yaml",
        # We don't actually use it but tl requires it to be defined
        "../yamls/area_energy/architecture/pseudo/problem-pseudo.yaml",
        "../yamls/area_energy/architecture/pseudo/mapper-pseudo.yaml",
        # Shared stuff
        "../yamls/area_energy/architecture/components/*.yaml",
        "../yamls/area_energy/architecture/variables.yaml",
    )

    tl.call_accelergy_verbose(
        spec,
        output_dir="../data/generated/check/accelergy_area",
        log_to = "../data/generated/check/accelergy_area/accelergy_verbose.log",
    )

    if filecmp.cmp("../data/pregenerated/check/accelergy_area/ART.yaml",
            "../data/generated/check/accelergy_area/ART.yaml"):
        print("Accelergy Area OK")
    else:
        print("Accelergy Area ERROR")

    output_dir = "../data/generated/check/accelergy_energy"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open("../data/generated/check/accelergy_energy/accelergy_verbose.log", "w") as f:
        subprocess.run(["accelergy",
                        "../yamls/area_energy/architecture/pseudo/action_counts.yaml",
                        "../data/generated/check/accelergy_area/parsed-processed-input.yaml",
                        "-o",
                        "../data/generated/check/accelergy_energy"],
                        stderr=f)

    if filecmp.cmp("../data/pregenerated/check/accelergy_energy/energy_estimation.yaml",
            "../data/generated/check/accelergy_energy/energy_estimation.yaml"):
        print("Accelergy Energy OK")
    else:
        print("Accelergy Energy ERROR")

def main():
    imports()
    timeloop()
    accelergy()

if __name__ == "__main__":
    main()
