from ruamel.yaml import YAML

from src.accel.cascade import Cascade
from src.accel.stable_softmax import StableSoftmax

class Unfused(Cascade):
    def __init__(self, model, seq_len):
        self.L3_width = 256 * 2
        self.L3_depth = 32 * 2**20 // self.L3_width
        self.reg_file = 4

        super().__init__(model, seq_len)

        self.compute_cost = {
            "QK": {"mac": 1},
            "AV": {"mac": 1},
        }

        self.einsums_2d = {"QK", "AV"}

    def build_accelergy_stats(self, output_dir, run_mapper=True):
        if "traffic" not in self.computed:
            self.eval(output_dir, run_mapper=run_mapper)

        self.add_data_locs_mapper(output_dir)
        stats = self.build_accelergy_stats_general(output_dir, "flat", source="mapper")

        ss_stats = self.computed["softmax_accel"].build_accelergy_stats(output_dir / "ss")

        self.combine_stats(stats, ss_stats)
        return stats



    def build_input(self, einsum, output_dir):
        inputs = self.build_input_prelude(einsum, "baselines", "unfused-constraints")

        self.__update_mappings(inputs)

        self.build_input_postlude(output_dir, inputs)

    def __update_mappings(self, inputs):
        mapping = inputs["constraints"]["targets"][0]
        mapping["factors"][1] = "H=" + str(self.transformer.H)

    def check_llb(self, einsum, tensor):
        return False

    def check_dram(self, einsum, tensor):
        return True

    def eval_components(self, output_dir, run_mapper=True):
        args = ("QK", output_dir / "qk")
        self.build_input(*args)
        if run_mapper:
            self.run_mapper(*args, "../inputs/yamls/proposal/arch-2d.yaml", spec_callback=self.__timeloop_callback)

        qk_traffic = sum(self.collect_mem_traffic(*args, "L3", source="mapper"))
        qk_mem_lat, qk_comp_lat = self.collect_latency(*args, qk_traffic, source="mapper")

        args = ("AV", output_dir / "av")
        self.build_input(*args)
        if run_mapper:
            self.run_mapper(*args, "../inputs/yamls/proposal/arch-2d.yaml", spec_callback=self.__timeloop_callback)

        av_traffic = sum(self.collect_mem_traffic(*args, "L3", source="mapper"))
        av_mem_lat, av_comp_lat = self.collect_latency(*args, av_traffic, source="mapper")

        ss = StableSoftmax(self.model, self.seq_len, False, avail_buf=self.L3_depth * self.L3_width)
        self.computed["softmax_accel"] = ss
        ss_traffic, ss_mem_lat, ss_comp_lat = ss.eval_components(
            output_dir / "ss")

        traffic = qk_traffic + av_traffic + ss_traffic
        self.computed["traffic"] = traffic

        self.computed["qk_mem_lat"] = qk_mem_lat
        self.computed["qk_comp_lat"] = qk_comp_lat
        self.computed["av_mem_lat"] = av_mem_lat
        self.computed["av_comp_lat"] = av_comp_lat
        self.computed["ss_mem_lat"] = ss_mem_lat
        self.computed["ss_comp_lat"] = ss_comp_lat

        return traffic, qk_mem_lat, qk_comp_lat, av_mem_lat, av_comp_lat, ss_mem_lat, ss_comp_lat

    def eval_energy(self, output_dir, arch):
        return super().eval_energy(output_dir, arch, spec_callback=self.__accelergy_callback)

    def eval(self, output_dir, run_mapper=True):
        if "traffic" not in self.computed:
            self.eval_components(output_dir, run_mapper)

        # Here, we will again use the TeAAL heuristics
        qk_lat = max(self.computed["qk_mem_lat"], self.computed["qk_comp_lat"])
        av_lat = max(self.computed["av_mem_lat"], self.computed["av_comp_lat"])
        ss_lat = max(self.computed["ss_mem_lat"], self.computed["ss_comp_lat"])

        latency = qk_lat + av_lat + ss_lat
        self.computed["latency"] = latency

        return self.computed["traffic"], latency

    def eval_utilization(self, output_dir, run_mapper=True):
        if "latency" not in self.computed:
            self.eval(output_dir, run_mapper)

        # 2D array is always fully utilized when active
        util_2d = (self.computed["qk_comp_lat"] + self.computed["av_comp_lat"]) / self.computed["latency"]

        # 1D array is always fully utilized when active
        util_1d = self.computed["ss_comp_lat"] / self.computed["latency"]

        return util_2d, util_1d

    def __timeloop_callback(self, spec):
        spec["architecture"]["nodes"].find("L3")["attributes"]["depth"] = self.L3_depth
        spec["architecture"]["nodes"].find("reg_file")["attributes"]["depth"] = self.reg_file

    def __accelergy_callback(self, spec, array):
        if array == "2d":
            spec["architecture"]["nodes"].find("global_buffer")["attributes"]["depth"] = self.L3_depth
            spec["architecture"]["nodes"].find("reg_file")["attributes"]["depth"] = self.reg_file
