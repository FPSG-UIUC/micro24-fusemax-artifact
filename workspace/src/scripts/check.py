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
    "../inputs/yamls/proposal/arch-2d.yaml",
    "../inputs/yamls/proposal/problems/qk.yaml",
    "../inputs/yamls/proposal/mappings/qk.yaml",
    "../inputs/timeloop-accelergy-exercises/workspace/example_designs/example_designs/_components/*")

    tl.call_model(spec, "../outputs/generated-check/timeloop")

    if filecmp.cmp("../outputs/pregenerated-check/timeloop/timeloop-model.stats.txt",
            "../outputs/generated-check/timeloop/timeloop-model.stats.txt"):
        print("Timeloop OK")
    else:
        print("Timeloop ERROR")

def accelergy():
    spec = tl.Specification.from_yaml_files(
        "../inputs/yamls/area_energy/architecture/accel_fusemax/accel-top-1d-path.yaml",
        # We don't actually use it but tl requires it to be defined
        "../inputs/yamls/area_energy/architecture/pseudo/problem-pseudo.yaml",
        "../inputs/yamls/area_energy/architecture/pseudo/mapper-pseudo.yaml",
        # Shared stuff
        "../inputs/yamls/area_energy/architecture/components/*.yaml",
        "../inputs/yamls/area_energy/architecture/variables.yaml",
    )

    tl.call_accelergy_verbose(
        spec,
        output_dir="../outputs/generated-check/accelergy_area",
        log_to = "../outputs/generated-check/accelergy_area/accelergy_verbose.log",
    )

    if filecmp.cmp("../outputs/pregenerated-check/accelergy_area/ART.yaml",
            "../outputs/generated-check/accelergy_area/ART.yaml"):
        print("Accelergy Area OK")
    else:
        print("Accelergy Area ERROR")

    output_dir = "../outputs/generated-check/accelergy_energy"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open("../outputs/generated-check/accelergy_energy/accelergy_verbose.log", "w") as f:
        subprocess.run(["accelergy",
                        "../inputs/yamls/area_energy/architecture/pseudo/action_counts.yaml",
                        "../outputs/generated-check/accelergy_area/parsed-processed-input.yaml",
                        "-o",
                        "../outputs/generated-check/accelergy_energy"],
                        stderr=f)

    if filecmp.cmp("../outputs/pregenerated-check/accelergy_energy/energy_estimation.yaml",
            "../outputs/generated-check/accelergy_energy/energy_estimation.yaml"):
        print("Accelergy Energy OK")
    else:
        print("Accelergy Energy ERROR")

def outputs(csv, experiment_dir="../outputs/generated/default"):
    experiment_dir = Path(experiment_dir)
    results_dir = experiment_dir / "results"

    if filecmp.cmp(results_dir / f"{csv}.csv",
                   f"../outputs/pregenerated/results/{csv}.csv"):
        print("Matches pregenerated outputs")
    else:
        print("Does not match pregenerated outputs")

def main():
    imports()
    timeloop()
    accelergy()

if __name__ == "__main__":
    main()
