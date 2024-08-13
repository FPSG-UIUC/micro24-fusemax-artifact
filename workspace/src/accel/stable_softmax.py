from src.accel.cascade import Cascade

# This class evaluates the numerically stable softmax
# Used for unfused baseline and FLAT
#
# The Einsums are as follows:
# M[b, h, p] = QK[b, h, m, p] : reduced with max
# SN[b, h, m, p] = exp(QK[b, h, m, p] - M[b, h, p])
# SD[b, h, p] = SN[b, h, m, p]
# A[b, h, m, p] = SN[b, h, m, p] / SD[b, h, p]
#
# Data spaces:
# qk_m fiber in L3 (overbooked)
# m_mut in L3
# sn_m in L3 (overbooked)
# sd_mut in L3
# a_m in L3 (overbooked)


class StableSoftmax(Cascade):
    def __init__(self, model, seq_len, is_QK_A_onchip, PE_dim=256, avail_buf=None, mem_bw=400 * 2**30, prop_spilled=None):
        super().__init__(model, seq_len)

        self.PE_dim = PE_dim
        self.mem_bw = mem_bw
        self.is_QK_A_onchip = is_QK_A_onchip

        if prop_spilled is None:
            # The L3 is 32MB
            self.prop_spilled = 1 - min(1, avail_buf / (self.transformer.M * 2))
        else:
            self.prop_spilled = prop_spilled

        self.compute_cost = {
            "M": {"max": 1},
            "SN": {"add": 1, "exponentiatial": 1},
            "SD": {"add": 1},
            "A": {"divide": 1}
        }

    def build_accelergy_stats(self, output_dir):
        if "traffic" not in self.computed:
            self.eval(output_dir)

        stats = self.build_accelergy_stats_general(output_dir, "flat")

        # But the DRAM traffic may be wrong (if spilling)
        # 64 bit width -> 8B per line
        stats["DRAM"]["read"] = self.computed["rd_traffic"] / 8
        stats["DRAM"]["write"] = self.computed["wr_traffic"] / 8

        return stats


    def build_input(self, einsum, output_dir):
        inputs = self.build_input_prelude(
            einsum, "baselines", "softmax-mappings")

        # Update the mapping
        self.__update_factors(inputs)
        if einsum == "M":
            self.__update_m(inputs)

        elif einsum == "A":
            self.__update_a(inputs)

        self.add_data_locs(einsum, inputs["mapping"])

        self.build_input_postlude(output_dir, inputs)

    def check_dram(self, einsum, tensor):
        if (einsum == "M" and tensor == "QK") or (
                einsum == "A" and tensor == "A"):
            return not self.is_QK_A_onchip

        return False

    def check_llb(self, einsum, tensor):
        if einsum == "SD" and tensor == "SN":
            return False

        if tensor in {"M", "SD"}:
            return False

        if tensor == "A":
            return self.is_QK_A_onchip

        return True

    def eval_components(self, output_dir):
        args = ("M", output_dir / "m")
        self.build_input(*args)
        self.run_model(*args, "../inputs/yamls/baselines/arch-1d.yaml", self.__timeloop_callback)
        m_rd, m_wr = self.collect_mem_traffic(*args, "L3", self.prop_spilled)
        m_mem_lat, m_comp_lat = self.collect_latency(*args, m_rd + m_wr, mem_bw=self.mem_bw)

        args = ("SN", output_dir / "sn")
        self.build_input(*args)
        self.run_model(*args, "../inputs/yamls/baselines/arch-1d.yaml", self.__timeloop_callback)
        sn_rd, sn_wr = self.collect_mem_traffic(*args, "L3", self.prop_spilled)
        sn_mem_lat, sn_comp_lat = self.collect_latency(*args, sn_rd + sn_wr, mem_bw=self.mem_bw)

        args = ("SD", output_dir / "sd")
        self.build_input(*args)
        self.run_model(*args, "../inputs/yamls/baselines/arch-1d.yaml", self.__timeloop_callback)
        sd_rd, sd_wr = self.collect_mem_traffic(*args, "L3", self.prop_spilled)
        sd_mem_lat, sd_comp_lat = self.collect_latency(*args, sd_rd + sd_wr, mem_bw=self.mem_bw)

        args = ("A", output_dir / "a")
        self.build_input(*args)
        self.run_model(*args, "../inputs/yamls/baselines/arch-1d.yaml", self.__timeloop_callback)
        a_rd, a_wr = self.collect_mem_traffic(*args, "L3", self.prop_spilled)
        a_mem_lat, a_comp_lat = self.collect_latency(*args, a_rd + a_wr, mem_bw=self.mem_bw)

        rd_traffic = m_rd + sn_rd + sd_rd + a_rd
        wr_traffic = m_wr + sn_wr + sd_wr + a_wr
        self.computed["rd_traffic"] = rd_traffic
        self.computed["wr_traffic"] = wr_traffic

        traffic = rd_traffic + wr_traffic
        self.computed["traffic"] = traffic

        mem_latency = m_mem_lat + sn_mem_lat + sd_mem_lat + a_mem_lat

        # Multi-cycle operations (exp, div) can be pipelined
        comp_latency = m_comp_lat + sn_comp_lat + sd_comp_lat + a_comp_lat

        return traffic, mem_latency, comp_latency

    def eval(self, output_dir):
        traffic, mem_latency, comp_latency = self.eval_components(output_dir)
        latency = max(mem_latency, comp_latency)

        return traffic, latency

    def __timeloop_callback(self, spec, einsum):
        spec["architecture"]["nodes"].find("PE")["spatial"]["meshX"] = self.PE_dim

        # For Timeloop, because of the spilling, we just need the buffer size to be large
        # The real buffer size is accounted for during the area/energy modeling

    def __update_m(self, inputs):
        if not self.is_QK_A_onchip:
            return

        mappings = inputs["mapping"].__iter__()

        # DRAM factors
        next(mappings)

        # DRAM dataspace
        map_ = next(mappings)
        map_["keep"] = []
        map_["bypass"] = ["QK", "M"]

    def __update_a(self, inputs):
        if not self.is_QK_A_onchip:
            return

        mappings = inputs["mapping"].__iter__()

        # DRAM factors
        next(mappings)

        # DRAM dataspace
        map_ = next(mappings)
        map_["keep"] = []
        map_["bypass"] = ["SN", "SD", "A"]

        # L3 factors
        next(mappings)

        # L3 dataspace
        map_ = next(mappings)
        map_["keep"] = ["SN", "SD", "A"]
        map_["bypass"] = []

    def __update_factors(self, inputs):
        mappings = inputs["mapping"].__iter__()

        # DRAM factors
        map_ = next(mappings)
        map_["factors"] = [
            "B=" + str(self.transformer.B),
            "H=" + str(self.transformer.H),
            "M=1",
            "P=" + str(self.transformer.P),
        ]

        # DRAM dataspace
        map_ = next(mappings)

        # L3 factors
        map_ = next(mappings)
        M1 = max(1, self.transformer.M // self.PE_dim)
        map_["factors"] = ["B=1", "H=1", "M=" + str(M1), "P=1"]

        # L3 dataspace
        map_ = next(mappings)

        # PE spatial
        map_ = next(mappings)
        M0 = self.PE_dim
        map_["factors"] = ["B=1", "H=1", "M=" + str(M0), "P=1"]
