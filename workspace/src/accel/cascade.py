from pathlib import Path
import subprocess

from ruamel.yaml import YAML
import timeloopfe.v4 as tl

from src.utils.stats import Stats
from src.utils.transformer import Transformer


class Cascade:
    def __init__(self, model, seq_len):
        self.model = model
        self.seq_len = seq_len
        self.transformer = Transformer.from_csv("cloud", model, seq_len)

        self.tensors = {}
        self.computed = {}

        self.data_locs = {}
        self.einsums_2d = {}

    def add_data_locs(self, einsum, mappings):
        self.data_locs[einsum] = {}
        for mapping in mappings:
            if mapping["type"] not in {"dataspace", "datatype"}:
                continue

            self.data_locs[einsum][mapping["target"]] = mapping["keep"].copy()

    def add_data_locs_mapper(self, output_dir):
        yaml = YAML(typ="safe")
        for einsum in self.compute_cost:
            einsum_dir = output_dir + "/" + einsum.lower()

            with open(einsum_dir + "/timeloop-mapper.map.yaml", "r") as f:
                mappings = yaml.load(f)["mapping"]

            self.add_data_locs(einsum, mappings)

    def build_accelergy_stats_general(self, output_dir, arch, source="model"):
        stats = {}
        for einsum in self.compute_cost.keys():
            estats = {}
            einsum_dir = output_dir + "/" + einsum.lower()

            for mem in self.data_locs[einsum]:
                estats[mem] = self.build_accelergy_buf_counts(einsum, einsum_dir, mem, source=source)

            for fu, cost in self.compute_cost[einsum].items():
                estats[fu] = self.build_accelergy_comp_counts(einsum_dir, cost=cost, source=source)

            stats[einsum] = estats

        final = {"DRAM": {"read": 0, "write": 0, "update": 0, "leak": 0},
            "L3": {"read": 0, "write": 0, "update": 0, "leak": 0},
            "reg_file_2d": {"read": 0, "write": 0, "update": 0, "leak": 0},
            "reg_file_1d": {"read": 0, "write": 0, "update": 0, "leak": 0}}
        if arch == "proposal":
            final["func_2d"] = {"mac": {"compute": 0, "leak": 0},
                    "max": {"compute": 0, "leak": 0},
                    "add": {"compute": 0, "leak": 0}}
            final["func_1d"] = {"mac": {"compute": 0, "leak": 0},
                    "max": {"compute": 0, "leak": 0},
                    "add": {"compute": 0, "leak": 0},
                    "divide": {"compute": 0, "leak": 0}}

        else:
            final["func_2d"] = {"mac": {"compute": 0, "leak": 0},
                    "add": {"compute": 0, "leak": 0}}
            final["func_1d"] = {"mac": {"compute": 0, "leak": 0},
                    "max": {"compute": 0, "leak": 0},
                    "exponentiatial": {"compute": 0, "leak": 0},
                    "add": {"compute": 0, "leak": 0},
                    "divide": {"compute": 0, "leak": 0}}

        for einsum, estats in stats.items():
            self.combine_stats(final["DRAM"], estats["DRAM"])
            self.combine_stats(final["L3"], estats["L3"])

            if einsum in self.einsums_2d:
                if "reg_file" in estats.keys():
                    self.combine_stats(final["reg_file_2d"], estats["reg_file"])

                # Hold-over from the Eyeriss template
                else:
                    self.combine_stats(final["reg_file_2d"], estats["reg0"])
                    self.combine_stats(final["reg_file_2d"], estats["reg1"])
                    self.combine_stats(final["reg_file_2d"], estats["reg2"])

                if arch == "proposal":
                    self.combine_stats(final["reg_file_1d"], estats["reg_file_1d"])

                for fu, fstats in final["func_2d"].items():
                    if fu in estats.keys():
                        self.combine_stats(fstats, estats[fu])

            else:
                self.combine_stats(final["reg_file_1d"], estats["reg_file"])

                for fu, fstats in final["func_1d"].items():
                    if fu in estats.keys():
                        self.combine_stats(fstats, estats[fu])

        return final


    def build_accelergy_yaml(self, output_dir, stats):
        yaml = YAML(typ="safe")

        # Write the input to the output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        counts_2d = {"action_counts": {"version": "0.4", "local": []}}
        local = counts_2d["action_counts"]["local"]

        local.append(self.__count("DRAM", 1, stats["DRAM"]))
        local.append(self.__count("global_buffer", 1, stats["L3"]))

        # Using only one of each of the components, since they are identical
        # and it makes Accelergy run faster
        local.append(self.__count("reg_file", 1, stats["reg_file_2d"]))

        for fu_name, fu in stats["func_2d"].items():
            local.append(self.__count(fu_name, 1, fu))

        with open(output_dir + "/action_counts_2d.yaml", "w") as f:
            yaml.dump(counts_2d, f)

        counts_1d = {"action_counts": {"version": "0.4", "local": []}}
        local = counts_1d["action_counts"]["local"]

        local.append(self.__count("reg_file", 1, stats["reg_file_1d"]))
        for fu_name, fu in stats["func_1d"].items():
            local.append(self.__count(fu_name, 1, fu))

        with open(output_dir + "/action_counts_1d.yaml", "w") as f:
            yaml.dump(counts_1d, f)

    def build_accelergy_buf_counts(self, einsum, output_dir, level, source="model"):
        stats = Stats(output_dir + "/timeloop-" + source + ".stats.txt")

        # leak and update are unused
        counts = {"read": 0, "write": 0, "leak": 0, "update": 0}

        label = "=== " + level + " ==="
        for tensor in self.data_locs[einsum][level]:
            read = stats.read_data([label,
                                   tensor + ":", "Scalar reads"])
            fill = stats.read_data([label,
                                   tensor + ":", "Scalar fills"])
            write = stats.read_data([label,
                                   tensor + ":", "Scalar updates"])

            # TODO: This is because of the error with first read ellision
            if read != write or level != "DRAM":
                counts["read"] += read
            counts["write"] += write + fill

        if level == "DRAM":
            width = 4
        elif level == "L3":
            width = 256
        else:
            width = 1

        counts["read"] = counts["read"] / width
        counts["write"] = counts["write"] / width

        return counts

    def build_accelergy_comp_counts(self, output_dir, source="model", cost=1):
        stats = Stats(output_dir + "/timeloop-" + source + ".stats.txt")

        compute = stats.read_data(["=== mac ===", "Computes (total)"])

        # Leak is unused
        return {"compute": compute * cost, "leak": 0}

    def __count(self, name, num, stats):
        counts = []
        for key, val in stats.items():
            counts.append({})
            counts[-1]["name"] = key
            counts[-1]["counts"] = val

        full_name = "system_top_level." + name + "[" + str(num) + "]"
        return {"action_counts": counts, "name": full_name}


    def build_input(self, einsum, output_dir):
        raise NotImplementedError

    def build_input_prelude(self, einsum, yaml_dir, mapping_dir, attn_problem=True):
        # Load the YAMLs
        yaml = YAML(typ="safe")

        inputs = {}

        with open("../inputs/yamls/" + yaml_dir + "/problems/" + einsum.lower() + ".yaml", "r") as f:
            inputs.update(yaml.load(f))

        with open("../inputs/yamls/" + yaml_dir + "/" + mapping_dir + "/" + einsum.lower() + ".yaml", "r") as f:
            inputs.update(yaml.load(f))

        # Update the problem only if using attention
        if attn_problem:
            self.transformer.update_problem(inputs)

        self.tensors[einsum] = []
        for space in inputs["problem"]["shape"]["data_spaces"]:
            self.tensors[einsum].append(space["name"])

        return inputs

    def build_input_postlude(self, output_dir, inputs):
        yaml = YAML(typ="safe")

        # Write the input to the output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        with open(output_dir + "/problem+mapping.yaml", "w") as f:
            yaml.dump(inputs, f)

    def check_dram(self, einsum, tensor):
        raise NotImplementedError

    def check_llb(self, einsum, tensor):
        raise NotImplementedError

    def collect_area(self, output_dir):
        yaml = YAML(typ="safe")

        area = 0
        array_2d = 0
        for array in ["2d", "1d"]:
            with open(output_dir + "/" + array + "/ART.yaml", "r") as f:
                art = yaml.load(f)["ART"]["tables"]

            for table in art:
                name = table["name"].split(".")[1].split("[")[0]
                num = int(table["name"].split("..")[1][:-1])

                # Area for adder already accounted for as a part of the mac
                if name != "add":
                    area += num * table["area"]

                    if array == "2d" and num > 1:
                        array_2d += num * table["area"]

        return array_2d, area


    # Note: Assumes that build_input and run_model have already been run
    def collect_mem_traffic(self, einsum, output_dir, llb, prop_spilled=0, opt_spill=True, source="model"):
        stats = Stats(output_dir + "/timeloop-" + source + ".stats.txt")

        buf_rd = 0
        buf_wr = 0
        mem_rd = 0
        mem_wr = 0
        for tensor in self.tensors[einsum]:
            is_output = tensor == einsum
            tensor_prefix = tensor + ":"

            # LLB is a buffer that could spill to DRAM if overbooked
            if self.check_llb(einsum, tensor):
                llb_label = "=== " + llb + " ==="

                read_buf = stats.read_data([llb_label,
                                       tensor_prefix, "Scalar reads"])
                write_buf = stats.read_data([llb_label,
                                       tensor_prefix, "Scalar updates"])

                # TODO: This is just to avoid if first read elision, since
                # that is currently broken in Timeloop

                if read_buf != write_buf:
                    buf_rd += read_buf
                buf_wr += write_buf

            if self.check_dram(einsum, tensor):
                # If data is not buffered in a spillable buffer, then use the
                # Timeloop data movement
                if not self.check_llb(einsum, tensor):
                    prop_separately_moved = 1

                # If we are optimizing the spilling, i.e., not loading data that
                # will be immediately spilled
                elif opt_spill:
                    prop_separately_moved = 1 - prop_spilled

                # Only data that is *actually* buffered needs to be separately
                # read from DRAM
                else:
                    prop_separately_moved = 1

                read_mem = stats.read_data(["=== DRAM ===",
                                        tensor_prefix,
                                        "Scalar reads"]) * prop_separately_moved
                write_mem = stats.read_data(["=== DRAM ===",
                                        tensor_prefix,
                                        "Scalar updates"]) * prop_separately_moved

                # TODO: This is just to avoid if first read elision, since
                # that is currently broken in Timeloop
                if read_mem != write_mem:
                    mem_rd += read_mem
                mem_wr += write_mem

        # 2B values
        return ((mem_rd + buf_rd * prop_spilled) * 2, (mem_wr + buf_wr * prop_spilled) * 2)

    # Note: Assumes that build_input and run_model have already been run
    def collect_latency(self, einsum, output_dir, mem_traffic, source="model", mem_bw=400 * 2**30):
        stats = Stats(output_dir + "/timeloop-" + source + ".stats.txt")

        # Bandwidth is 400 GB/s for the cloud configuration
        mem_latency = mem_traffic / mem_bw

        # Frequency is 940 GHz
        cycles = stats.read_data(["=== mac ===", "Cycles"])
        comp_latency = cycles / (940 * 10**6)

        return mem_latency, comp_latency

    def combine_stats(self, stats1, stats2):
        assert stats1.keys() == stats2.keys()
        for key, val in stats2.items():
            if isinstance(val, dict):
                self.combine_stats(stats1[key], val)
            else:
                stats1[key] += val

    def eval_area(self, output_dir, arch, spec_callback=None):
        self.run_accelergy_area(output_dir, arch, spec_callback=spec_callback)
        return self.collect_area(output_dir)

    def eval_energy(self, output_dir, arch, spec_callback=None):
        stats = self.build_accelergy_stats(output_dir)
        self.build_accelergy_yaml(output_dir, stats)
        self.run_accelergy_energy(output_dir, arch, spec_callback=spec_callback)
        energy = self.read_energy(output_dir)

        return energy


    def read_energy(self, output_dir):
        yaml = YAML(typ="safe")

        energy = 0
        with open(output_dir + "/2d/energy_estimation.yaml", "r") as f:
            energy += yaml.load(f)["energy_estimation"]["Total"]

        with open(output_dir + "/1d/energy_estimation.yaml", "r") as f:
            energy += yaml.load(f)["energy_estimation"]["Total"]

        return energy


    def read_comp_energy(self, output_dir):
        yaml = YAML(typ="safe")

        names = []
        energies = []
        with open(output_dir + "/2d/energy_estimation.yaml", "r") as f:
            n, e = self.__get_comp_energy(yaml.load(f)["energy_estimation"]["components"])
            names += [name + "_2d" for name in n]
            energies += e

        with open(output_dir + "/1d/energy_estimation.yaml", "r") as f:
            n, e = self.__get_comp_energy(yaml.load(f)["energy_estimation"]["components"])
            names += [name + "_1d" for name in n]
            energies += e

        return names, energies

    def __get_comp_energy(self, components):
        names = []
        energies = []
        for comp in components:
            names.append(comp["name"].split(".")[1].split("[")[0])
            energies.append(comp["energy"])

        return names, energies


    # Note: Assumes that build_accelergy has already been run
    def run_accelergy_area(self, output_dir, arch, spec_callback=None):
        spec_2d = tl.Specification.from_yaml_files(
            "../inputs/yamls/area_energy/architecture/accel_" + arch + "/accel-top-2d-path.yaml",
            # We don't actually use it but tl requires it to be defined
            "../inputs/yamls/area_energy/architecture/pseudo/problem-pseudo.yaml",
            "../inputs/yamls/area_energy/architecture/pseudo/mapper-pseudo.yaml",
            # Shared stuff
            "../inputs/yamls/area_energy/architecture/components/*.yaml",
            "../inputs/yamls/area_energy/architecture/variables.yaml",
        )

        if spec_callback is not None:
            spec_callback(spec_2d, "2d")

        tl.call_accelergy_verbose(
            spec_2d,
            output_dir=output_dir + "/2d",
            log_to = output_dir + "/2d/accelergy_verbose.log",
        )

        spec_1d = tl.Specification.from_yaml_files(
            "../inputs/yamls/area_energy/architecture/accel_" + arch + "/accel-top-1d-path.yaml",
            # We don't actually use it but tl requires it to be defined
            "../inputs/yamls/area_energy/architecture/pseudo/problem-pseudo.yaml",
            "../inputs/yamls/area_energy/architecture/pseudo/mapper-pseudo.yaml",
            # Shared stuff
            "../inputs/yamls/area_energy/architecture/components/*.yaml",
            "../inputs/yamls/area_energy/architecture/variables.yaml",
        )

        if spec_callback is not None:
            spec_callback(spec_1d, "1d")

        tl.call_accelergy_verbose(
            spec_1d,
            output_dir=output_dir + "/1d",
            log_to = output_dir + "/1d/accelergy_verbose.log",
        )

    def run_accelergy_energy(self, output_dir, arch, spec_callback=None):
        self.run_accelergy_area(output_dir, arch, spec_callback=spec_callback)

        with open(output_dir + "/2d/accelergy_verbose.yaml", "w") as f:
            subprocess.run([
                "accelergy",
                output_dir + "/action_counts_2d.yaml",
                output_dir + "/2d/parsed-processed-input.yaml",
                "-o",
                output_dir + "/2d"],
                stderr=f)

        with open(output_dir + "/1d/accelergy_verbose.yaml", "w") as f:
            subprocess.run([
                "accelergy",
                output_dir + "/action_counts_1d.yaml",
                output_dir + "/1d/parsed-processed-input.yaml",
                "-o",
                output_dir + "/1d"],
            stderr=f)

    # Note: Assumes that build_input has already been run
    def run_mapper(self, einsum, output_dir, arch_yaml, spec_callback=None):
        spec = tl.Specification.from_yaml_files(
            output_dir +
            "/problem+mapping.yaml",
            arch_yaml,
            "../inputs/yamls/baselines/mapper.yaml",
            "../inputs/timeloop-accelergy-exercises/workspace/example_designs/example_designs/_components/*",
        )

        if spec_callback is not None:
            spec_callback(spec)

        tl.call_mapper(spec, output_dir)

    # Note: Assumes that build_input has already been run
    def run_model(self, einsum, output_dir, arch_yaml, spec_callback=None):
        spec = tl.Specification.from_yaml_files(
            output_dir +
            "/problem+mapping.yaml",
            arch_yaml,
            "../inputs/timeloop-accelergy-exercises/workspace/example_designs/example_designs/_components/*",
        )

        if spec_callback is not None:
            spec_callback(spec, einsum)

        tl.call_model(spec, output_dir)

    def update_factor(self, factors, rank, shape):
        i = factors.index(rank + "=1")
        shape = max(1, min(shape, getattr(self.transformer, rank)))
        factors[i] = rank + "=" + str(shape)
