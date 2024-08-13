from src.accel.cascade import Cascade
from src.accel.stable_softmax import StableSoftmax
from src.utils.csv_utils import CSVUtils
from src.utils.stats import Stats


class Flat(Cascade):
    def __init__(self, platform, model, seq_len, csv_fn):
        self.platform = platform
        if self.platform == "cloud":
            self.PE_dim = 256
            self.L3 = 32 * 2**20
            self.mem_bw = 400 * 2**30

        elif self.platform == "edge":
            self.PE_dim = 32
            self.L3 = 512 * 2**10
            self.mem_bw = 50 * 2**30

        else:
            raise NotImplementedError

        super().__init__(model, seq_len)

        self.csv = CSVUtils(csv_fn)

        self.true_input = {"Q", "K", "V"}
        self.true_output = {"AV"}

        # Filled during build_input
        self.locs = {}

        self.compute_cost = {
            "QK": {"mac": 1},
            "AV": {"mac": 1},
        }

        self.einsums_2d = ["QK", "AV"]

    def build_input(self, einsum, output_dir):
        inputs = self.build_input_prelude(einsum, "baselines", "flat-mappings")

        # Update the mapping
        if einsum == "QK":
            self.__update_qk(inputs)

        elif einsum == "AV":
            self.__update_av(inputs)

        else:
            raise ValueError("Unknown einsum: " + einsum)

        self.add_data_locs(einsum, inputs["mapping"])

        self.build_input_postlude(output_dir, inputs)

    def build_accelergy_stats(self, output_dir):
        if "traffic" not in self.computed:
            self.eval(output_dir, validation=False)

        stats = self.build_accelergy_stats_general(output_dir, "flat")

        # But the DRAM traffic may be wrong (if spilling)
        # 64 bit width -> 8B per line
        stats["DRAM"]["read"] = self.computed["qk_av_rd"] / 8
        stats["DRAM"]["write"] = self.computed["qk_av_wr"] / 8

        ss_stats = self.computed["softmax_accel"].build_accelergy_stats(output_dir / "ss")

        self.combine_stats(stats, ss_stats)
        return stats


    def check_dram(self, einsum, tensor):
        return tensor in self.true_input or tensor in self.true_output or \
            self.locs[tensor] == "offchip"

    def check_llb(self, einsum, tensor):
        return self.locs[tensor] == "onchip"

    def eval_2d_util(self, output_dir, validation=False):
        if "latency" not in self.computed:
            self.eval(output_dir, validation=validation)

        if "util_qk" not in self.computed:
            qk_stats = Stats(output_dir / "qk" / "timeloop-model.stats.txt")
            util_qk = qk_stats.read_data(["=== mac ===", "Utilized instances"])

            instances = int(qk_stats.read_data(["=== mac ===", "Instances"]).split(" ")[0])

            av_stats = Stats(output_dir / "av" / "timeloop-model.stats.txt")
            util_av = av_stats.read_data(["=== mac ===", "Utilized instances"])

            self.computed["util_qk"] = self.computed["qk_comp_lat"] * util_qk / (self.computed["latency"] * instances)
            self.computed["util_av"] = self.computed["av_comp_lat"] * util_av / (self.computed["latency"] * instances)

        return self.einsums_2d, [self.computed["util_qk"], self.computed["util_av"]]

    def eval_components(self, output_dir, validation=False):
        prop_spilled = self.__get_val("proportion_spilled")

        args = ("QK", output_dir / "qk")
        self.build_input(*args)
        self.run_model(*args, "../inputs/yamls/baselines/arch-2d.yaml", self.__timeloop_callback)

        # If we are validating, do not use optimal spilling
        qk_rd, qk_wr = self.collect_mem_traffic(*args, "L3", prop_spilled, opt_spill=not validation)
        qk_traffic = qk_rd + qk_wr

        qk_mem_lat, qk_comp_lat = self.collect_latency(*args, qk_traffic, mem_bw=self.mem_bw)

        args = ("AV", output_dir / "av")
        self.build_input(*args)
        self.run_model(*args, "../inputs/yamls/baselines/arch-2d.yaml", self.__timeloop_callback)

        # If we are validating, do not use optimal spilling
        av_rd, av_wr = self.collect_mem_traffic(*args, "L3", prop_spilled, opt_spill=not validation)
        av_traffic = av_rd + av_wr

        av_mem_lat, av_comp_lat = self.collect_latency(*args, av_traffic, mem_bw=self.mem_bw)

        buf_sz = self.compute_avail_buf(output_dir)
        is_QK_A_onchip = self.locs["QK"] == "onchip"
        if is_QK_A_onchip:
            ss_spilled = prop_spilled
        else:
            ss_spilled = None

        qk_av_traffic = qk_traffic + av_traffic
        qk_av_mem_lat = qk_mem_lat + av_mem_lat
        comp_2d_lat = qk_comp_lat + av_comp_lat
        valid_energy = qk_av_traffic * 32

        self.computed["qk_comp_lat"] = qk_comp_lat
        self.computed["av_comp_lat"] = av_comp_lat

        self.computed["qk_av_rd"] = qk_rd + av_rd
        self.computed["qk_av_wr"] = qk_wr + av_wr

        self.computed["qk_av_traffic"] = qk_av_traffic
        self.computed["qk_av_mem_lat"] = qk_av_mem_lat
        self.computed["comp_2d_lat"] = comp_2d_lat
        self.computed["valid_energy"] = valid_energy

        if validation:
            return qk_av_traffic, qk_av_mem_lat, comp_2d_lat, valid_energy

        ss = StableSoftmax(self.model, self.seq_len, is_QK_A_onchip, PE_dim=self.PE_dim, avail_buf=buf_sz, mem_bw=self.mem_bw, prop_spilled=ss_spilled)
        self.computed["softmax_accel"] = ss
        ss_traffic, ss_mem_lat, ss_comp_lat = ss.eval_components(
            output_dir / "ss")

        traffic = qk_av_traffic + ss_traffic
        mem_lat = qk_mem_lat + ss_mem_lat + av_mem_lat
        comp_1d_lat = ss_comp_lat

        self.computed["traffic"] = traffic
        self.computed["mem_lat"] = mem_lat
        self.computed["comp_1d_lat"] = comp_1d_lat

        return qk_av_traffic, traffic, qk_av_mem_lat, mem_lat, comp_2d_lat, comp_1d_lat

    def eval(self, output_dir, validation=False):
        if "traffic" not in self.computed:
            self.eval_components(output_dir, validation)

        # Use the TeAAL heuristics for combining results from different
        # components
        latency = max(self.computed["mem_lat"], self.computed["comp_2d_lat"], self.computed["comp_1d_lat"])

        self.computed["latency"] = latency

        return self.computed["traffic"], latency

    def eval_utilization(self, output_dir, validation=False):
        if "latency" not in self.computed:
            self.eval(output_dir, validation)

        if "util_qk" not in self.computed:
            self.eval_2d_util(output_dir, validation)

        # 2D array is not always fully utilized when active
        util_2d = self.computed["util_qk"] + self.computed["util_av"]

        # 1D array is always fully utilized when active
        util_1d = self.computed["comp_1d_lat"] / self.computed["latency"]

        return util_2d, util_1d

    def compute_avail_buf(self, output_dir):
        if self.locs["QK"] == "onchip":
            return None

        stats = Stats(output_dir / "qk" / "timeloop-model.stats.txt")
        occupied = 0
        for tensor in self.tensors["QK"]:
            if self.locs[tensor] == "onchip":
                occupied +=  stats.read_data(["=== L3 ===", tensor + ":", "Utilized capacity"])

        stats = Stats(output_dir / "av" / "timeloop-model.stats.txt")
        for tensor in self.tensors["AV"]:
            if self.locs[tensor] == "onchip":
                occupied +=  stats.read_data(["=== L3 ===", tensor + ":", "Utilized capacity"])

        return self.L3 - occupied * 2


    def __update_qk(self, inputs):
        stationarity = self.__get_val("QK_stationarity")

        if stationarity == "input":
            stat_perm = "MEPHB"
        elif stationarity == "weight":
            stat_perm = "PEMHB"
        else:
            stat_perm = "EMPHB"

        self.__update(inputs, stat_perm)

        # Update DRAM dataspace
        mappings = inputs["mapping"].__iter__()
        is_QK_onchip = self.__get_val("QK_loc") == "onchip"

        # DRAM temporal
        next(mappings)

        # Bypass on-chip QK
        map_ = next(mappings)
        if is_QK_onchip:
            map_["keep"] = ["Q", "K"]
            map_["bypass"] = ["QK"]
        else:
            map_["keep"] = ["Q", "K", "QK"]
            map_["bypass"] = []

    def __update_av(self, inputs):
        stationarity = self.__get_val("AV_stationarity")

        if stationarity == "input":
            stat_perm = "FMPHB"
        elif stationarity == "weight":
            stat_perm = "PFMHB"
        else:
            stat_perm = "MFPHB"

        self.__update(inputs, stat_perm)

        # Update DRAM dataspace
        mappings = inputs["mapping"].__iter__()
        is_A_onchip = self.__get_val("A_loc") == "onchip"

        # DRAM temporal
        next(mappings)

        # Bypass on-chip A
        map_ = next(mappings)
        if is_A_onchip:
            map_["keep"] = ["V", "AV"]
            map_["bypass"] = ["A"]
        else:
            map_["keep"] = ["A", "V", "AV"]
            map_["bypass"] = []

    def __update(self, inputs, stat_perm):
        # Note: Does not update the DRAM dataspace
        fusion_granularity = self.__get_val("fusion_granularity")

        if fusion_granularity == "row":
            P1 = self.__get_val("P1")
            P2 = self.transformer.P // P1
        else:
            P1 = self.transformer.P
            P2 = 1

        factors = inputs["mapping"][-2]["factors"].copy()

        # Update the loop nest
        mappings = inputs["mapping"].__iter__()

        # At the DRAM level, add the B and H ranks
        map_ = next(mappings)
        dram_factors = self.__add_bh(factors.copy())
        self.update_factor(dram_factors, "P", P2)
        map_["factors"] = dram_factors

        # DRAM dataspace
        next(mappings)

        # L3 temporal is everything besides the 256x256 inner tile
        map_ = next(mappings)
        l3_factors = factors.copy()
        self.__update_l3(l3_factors, stat_perm, "M", self.transformer.M, self.PE_dim)
        self.__update_l3(l3_factors, stat_perm, "P", P1, self.PE_dim)

        if "E=1" in factors:
            self.__update_l3(l3_factors, stat_perm, "E", self.transformer.E, self.PE_dim)
        if "F=1" in factors:
            self.__update_l3(l3_factors, stat_perm, "F", self.transformer.F, self.PE_dim)

        map_["factors"] = l3_factors

        map_["permutation"] = list(stat_perm)

        # Buffer tensors (or not)
        keep, bypass = self.__bypass_l3(inputs["problem"]["shape"]["name"])

        # Set the L3 bypass
        map_ = next(mappings)

        map_["keep"] = keep
        map_["bypass"] = bypass

        # Set the temporal rank of the PE_col
        map_ = next(mappings)
        pec_factors = factors.copy()
        self.update_factor(pec_factors, stat_perm[1], self.PE_dim)
        map_["factors"] = pec_factors

        # Set the temporal rank of the PE
        map_ = next(mappings)
        pec_factors = factors.copy()
        self.update_factor(pec_factors, stat_perm[2], self.PE_dim)
        map_["factors"] = pec_factors

    def __add_bh(self, factors):
        self.update_factor(factors, "B", self.transformer.B)
        self.update_factor(factors, "H", self.transformer.H)
        return factors

    def __get_val(self, key):
        return self.csv.query((self.platform, self.model, self.seq_len), [key])[key]

    def __bypass_l3(self, einsum):
        keep = []
        bypass = []

        for tensor in self.tensors[einsum]:
            loc = self.__get_val(tensor + "_loc")
            self.locs[tensor] = loc
            if loc == "onchip":
                keep.append(tensor)
            else:
                bypass.append(tensor)

        return keep, bypass

    def __timeloop_callback(self, spec, einsum):
        spec["architecture"]["nodes"].find("PE_col")["spatial"]["meshX"] = self.PE_dim
        spec["architecture"]["nodes"].find("PE")["spatial"]["meshY"] = self.PE_dim

        # For Timeloop, because of the spilling, we just need the buffer size to be large
        # The real buffer size is accounted for during the area/energy modeling

    def __update_l3(self, factors, perm, rank, shape, PE_dim):
        if rank == perm[0]:
            self.update_factor(factors, rank, shape)
        else:
            self.update_factor(factors, rank, shape // PE_dim)
