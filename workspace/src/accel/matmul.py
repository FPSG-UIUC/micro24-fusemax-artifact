import os
import shutil

from src.accel.cascade import Cascade

class MatMul(Cascade):
    def __init__(self, platform, model, seq_len):
        super().__init__(model, seq_len)

        l3_width = 256 * 2
        if platform == "proposal":
            self.L3 = 16 * 2**20 // l3_width
            self.reg_file = 16

        else:
            self.L3 = 32 * 2**20 // l3_width
            self.reg_file = 4

        self.compute_cost = {
            "Q": {"mac": 1},
            "K": {"mac": 1},
            "V": {"mac": 1},
            "Z": {"mac": 1},
            "FFN1": {"mac": 1},
            "FFN2": {"mac": 1},
        }

        self.einsums_2d = set(self.compute_cost.keys())

    def build_accelergy_stats(self, output_dir, run_mapper=True):
        if "traffic" not in self.computed:
            self.eval(output_dir, run_mapper=run_mapper)

        # Prepare K and V generation
        if os.path.exists(output_dir / "k"):
            shutil.rmtree(output_dir / "k")
        shutil.copytree(output_dir / "q", output_dir / "k")

        if os.path.exists(output_dir / "v"):
            shutil.rmtree(output_dir / "v")
        shutil.copytree(output_dir / "q", output_dir / "v")

        self.add_data_locs_mapper(output_dir)

        stats = self.build_accelergy_stats_general(output_dir, "proposal", source="mapper")

        return stats

    def build_input(self, einsum, output_dir):
        inputs = self.build_input_prelude("MM", "baselines", "unfused-constraints", attn_problem=False)
        self.tensors[einsum] = self.tensors["MM"]

        self.__update_problem(einsum, inputs)

        self.build_input_postlude(output_dir, inputs)

    def __update_problem(self, einsum, inputs):
        M = self.transformer.B * self.transformer.P

        if einsum == "FFN1":
            K = self.transformer.D
            N = self.transformer.S
        elif einsum == "FFN2":
            K = self.transformer.S
            N = self.transformer.D
        elif einsum == "Z":
            K = self.transformer.E * self.transformer.H
            N = self.transformer.D
        # Q, K, V generation
        else:
            K = self.transformer.D
            N = self.transformer.E * self.transformer.H

        inputs["problem"]["instance"]["K"] = K
        inputs["problem"]["instance"]["M"] = M
        inputs["problem"]["instance"]["N"] = N

    def check_dram(self, einsum, tensor):
        return True

    def check_llb(self, einsum, tensor):
        return False

    def eval_einsum(self, einsum, output_dir, run_mapper=True):
        self.build_input(einsum, output_dir)

        if run_mapper:
            self.run_mapper(einsum, output_dir, "../inputs/yamls/proposal/arch-2d.yaml", spec_callback=self.__timeloop_callback)

        traffic = sum(self.collect_mem_traffic(einsum, output_dir, "L3", source="mapper"))
        mem_lat, comp_lat = self.collect_latency(einsum, output_dir, traffic, source="mapper")

        return traffic, max(mem_lat, comp_lat)

    def eval_energy(self, output_dir):
        return super().eval_energy(output_dir, "fusemax", spec_callback=self.__accelergy_callback)

    def eval(self, output_dir, run_mapper):
        results = {}
        for einsum in ["Q", "Z", "FFN1", "FFN2"]:
            results[einsum] = self.eval_einsum(einsum, output_dir / einsum.lower(), run_mapper=run_mapper)

        results["K"] = results["Q"]
        results["V"] = results["Q"]

        traffic = sum(result[0] for result in results.values())
        latency = sum(result[1] for result in results.values())

        return traffic, latency


    def __timeloop_callback(self, spec):
        spec["architecture"]["nodes"].find("L3")["attributes"]["depth"] = self.L3
        spec["architecture"]["nodes"].find("reg_file")["attributes"]["depth"] = self.reg_file

    def __accelergy_callback(self, spec, array):
        if array == "2d":
            spec["architecture"]["nodes"].find("global_buffer")["attributes"]["depth"] = self.L3
            spec["architecture"]["nodes"].find("reg_file")["attributes"]["depth"] = self.reg_file
