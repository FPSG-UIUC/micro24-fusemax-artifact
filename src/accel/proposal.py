from functools import reduce

from src.accel.cascade import Cascade


class Proposal(Cascade):
    def __init__(self, model, seq_len, PE_dim=256, l3_sz=16 * 2**20):
        super().__init__(model, seq_len)

        self.PE_dim = PE_dim

        # Compute P1
        bk_bv_space = self.PE_dim * self.transformer.E * \
            2 + self.PE_dim * self.transformer.F * 2
        avail_l3 = l3_sz - bk_bv_space
        # Q, RM, RD, RNV
        used_space_per_p = (self.transformer.E + 1 + 1 + self.transformer.F) * 2

        proposed_P1 = avail_l3 // (used_space_per_p * self.PE_dim)
        self.P1 = min(proposed_P1, self.transformer.P // self.PE_dim)

        # Do not use improper factorization
        for factor in self.factors(self.transformer.P // self.PE_dim):
            if factor <= self.P1:
                self.P1 = factor
                break

        # TODO: Fix
        assert self.P1 > 0

        # Partition M in the Einsum
        self.transformer.partition("M", "M", "N", self.PE_dim)

        self.einsums_2d = ["QK", "LM", "SLN", "SLD", "SLNV"]

        self.compute_cost = {
            "QK": {"mac": 1},
            "LM": {"max": 1},
            "RM": {"max": 1},
            "PRM": {"mac": 6, "add": 1},
            "SPD": {"mac": 1},
            "SPNV": {"mac": 1},
            "SLN": {"mac": 6, "add": 1},
            "SLD": {"add": 1},
            "SLNV": {"mac": 1},
            "RD": {"add": 1},
            "RNV": {"add": 1},
            "AV": {"divide": 1}
        }

    def factors(self, n):
        # Source:
        # https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python
        factors = list(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
        factors.sort(reverse=True)
        return factors

    def build_input(self, einsum, output_dir, mapping_dir="mappings"):
        inputs = self.build_input_prelude(einsum, "proposal", mapping_dir)

        # Update the mapping
        self.__update_mappings(einsum, inputs)

        self.add_data_locs(einsum, inputs["mapping"])

        self.build_input_postlude(output_dir, inputs)

    def build_accelergy_stats(self, output_dir):

        if "traffic" not in self.computed:
            self.eval(output_dir)

        final = self.build_accelergy_stats_general(output_dir, "proposal")

        return final


    def check_dram(self, einsum, tensor):
        if tensor in {"Q", "BK", "BV", "AV"}:
            return True

        return False

    # Data is never spilled to memory
    def check_llb(self, einsum, tensor):
        return False

    def eval_per_einsum(self, output_dir, spec_callback=None, arch_prefix=""):
        results = {}
        for einsum, cost_dict in self.compute_cost.items():
            args = (einsum, output_dir + "/" + einsum.lower())
            self.build_input(*args)

            if einsum in self.einsums_2d:
                arch = "yamls/proposal/" + arch_prefix + "arch-2d.yaml"
            else:
                arch = "yamls/proposal/" + arch_prefix + "arch-1d.yaml"

            self.run_model(*args, arch, spec_callback)

            traffic = sum(self.collect_mem_traffic(*args, None, False))
            mem_lat, comp_lat = self.collect_latency(*args, traffic)

            cost = sum(cost_dict.values())
            results[einsum] = (traffic, mem_lat, comp_lat * cost)

        self.computed["results"] = results
        return results

    def eval_2d_util(self, output_dir, spec_callback=None):
        if "results" not in self.computed:
            self.eval(output_dir, spec_callback=spec_callback)

        utils = []
        for einsum in self.einsums_2d:
            utils.append(self.computed["results"][einsum][2] / self.computed["latency"])

        return self.einsums_2d, utils

    def eval_components(self, output_dir, spec_callback=None):
        if "results" not in self.computed:
            self.eval_per_einsum(output_dir, spec_callback=spec_callback)

        results = self.computed["results"]

        traffic = sum(result[0] for result in results.values())
        mem_lat = sum(result[1] for result in results.values())
        comp_2d_lat = sum(result[2] for einsum, result in results.items() if einsum in self.einsums_2d)
        comp_1d_lat = sum(result[2] for einsum, result in results.items() if einsum not in self.einsums_2d)

        self.computed["comp_2d_lat"] = comp_2d_lat
        self.computed["comp_1d_lat"] = comp_1d_lat

        return traffic, mem_lat, comp_2d_lat, comp_1d_lat

    def eval(self, output_dir, spec_callback=None):
        if "results" not in self.computed:
            self.eval_per_einsum(output_dir, spec_callback=spec_callback)

        results = self.computed["results"]

        # Use the TeAAL heuristics to combine the results
        traffic = sum(result[0] for result in results.values())

        # Mem-bound latency
        mem_latency = sum(result[1] for result in results.values())

        # Compute bound, see waterfall diagram
        qk_lm = results["QK"][2] + results["LM"][2]
        rm = results["RM"][2]
        sln_sld = results["SLN"][2] + results["SLD"][2]
        prm_spd = results["PRM"][2] + results["SPD"][2]
        slnv = results["SLNV"][2]
        spnv = results["SPNV"][2]
        rd_rnv_av = results["RD"][2] + results["RNV"][2] + results["AV"][2]

        comp_latency = max(qk_lm, rd_rnv_av) + rm + max(sln_sld, prm_spd) + max(slnv, spnv)
        latency = max(comp_latency, mem_latency)

        self.computed["latency"] = latency

        return traffic, latency

    def eval_utilization(self, output_dir):
        if "latency" not in self.computed:
            self.eval(output_dir)

        if "comp_2d_lat" not in self.computed:
            self.eval_components(output_dir)

        # 2D array is always fully utilized when active
        util_2d = (self.computed["comp_2d_lat"]) / self.computed["latency"]

        # 1D array is always fully utilized when active
        util_1d = self.computed["comp_1d_lat"] / self.computed["latency"]

        return util_2d, util_1d

    def __update_mappings(self, einsum, inputs):
        factors = self.__clear_factors(inputs["mapping"][-2]["factors"])
        mappings = inputs["mapping"].__iter__()

        # DRAM factors
        map_ = next(mappings)
        dram_factors = factors.copy()
        self.update_factor(dram_factors, "B", self.transformer.B)
        self.update_factor(dram_factors, "H", self.transformer.H)
        if "M=1" in factors:
            self.update_factor(dram_factors, "M", self.transformer.M)
        P1_size = self.P1 * self.PE_dim
        if P1_size < self.transformer.P:
            self.update_factor(
                dram_factors,
                "P",
                self.transformer.P //
                P1_size)
        map_["factors"] = dram_factors

        # DRAM dataspace
        map_ = next(mappings)

        # L3 factors
        map_ = next(mappings)
        l3_factors = factors.copy()
        if "E=1" in factors:
            self.update_factor(l3_factors, "E", self.transformer.E)
        elif "F=1" in factors:
            self.update_factor(l3_factors, "F", self.transformer.F)
        self.update_factor(l3_factors, "P", self.P1)

        if "N=1" in factors:
            N_l3 = self.__get_factor(map_["factors"], "N")
            self.update_factor(l3_factors, "N", N_l3)

        map_["factors"] = l3_factors

        # L3 dataspace
        next(mappings)

        if einsum in self.einsums_2d:
            # reg_file_1d
            next(mappings)
            next(mappings)

        # (2D) PE_col or (1D) PE
        map_ = next(mappings)
        pe_col_factors = factors.copy()
        self.update_factor(pe_col_factors, "P", self.PE_dim)
        map_["factors"] = pe_col_factors

        if einsum in self.einsums_2d:
            # PE
            map_ = next(mappings)
            pe_factors = factors.copy()
            self.update_factor(pe_factors, "N", self.PE_dim)
            map_["factors"] = pe_factors

    def __clear_factors(self, factors):
        new = []
        for factor in factors:
            new.append(factor.split("=")[0] + "=1")

        return new

    def __get_factor(self, factors, rank):
        for factor in factors:
            split = factor.split("=")
            if split[0] == rank:
                return int(split[1])
