from src.accel.proposal import Proposal

class FlatPEProposal(Proposal):
    def __init__(self, model, seq_len, PE_dim=256, l3_sz=16 * 2**20):
        super().__init__(model, seq_len, PE_dim=PE_dim, l3_sz=l3_sz)

        # All other Einsums must be computed on the 1D array
        self.einsums_2d = ["QK", "SLNV"]

        self.compute_cost = {
            "QK": {"mac": 1},
            "LM": {"max": 1},
            "RM": {"max": 1},
            "PRM": {"exponential": 1, "add": 1},
            "SPD": {"mac": 1},
            "SPNV": {"mac": 1},
            "SLN": {"exponential": 1, "add": 1},
            "SLD": {"add": 1},
            "SLNV": {"mac": 1},
            "RD": {"add": 1},
            "RNV": {"add": 1},
            "AV": {"divide": 1}
        }

        # Compute P1
        bk_bv_space = self.PE_dim * self.transformer.E * \
            2 + self.PE_dim * self.transformer.F * 2
        avail_l3 = l3_sz - bk_bv_space
        # Q, QK, RM, SLN, RD, RNV
        used_space_per_p = (self.transformer.E + PE_dim + 1 + PE_dim + 1 + self.transformer.F) * 2

        proposed_P1 = avail_l3 // (used_space_per_p * self.PE_dim)
        self.P1 = min(proposed_P1, self.transformer.P // self.PE_dim)

        # Do not use improper factorization
        for factor in self.factors(self.transformer.P // self.PE_dim):
            if factor <= self.P1:
                self.P1 = factor
                break

        # TODO: Fix
        assert self.P1 > 0


    def build_input(self, einsum, output_dir):
        super().build_input(einsum, output_dir, "flat_pe_mappings")

    def build_accelergy_stats(self, output_dir):

        if "traffic" not in self.computed:
            self.eval(output_dir)

        final = self.build_accelergy_stats_general(output_dir, "flat")

        return final


    def eval_per_einsum(self, output_dir, spec_callback=None):
        return super().eval_per_einsum(output_dir, spec_callback=spec_callback, arch_prefix="flat_")

    def eval(self, output_dir, spec_callback=None):
        if "results" not in self.computed:
            self.eval_components(output_dir, spec_callback=spec_callback)

        results = self.computed["results"]

        # Use the TeAAL heuristics to combine the results
        traffic = sum(result[0] for result in results.values())

        # Mem-bound latency
        mem_latency = sum(result[1] for result in results.values())

        comp_latency = max(self.computed["comp_2d_lat"], self.computed["comp_1d_lat"])

        latency = max(mem_latency, comp_latency)
        self.computed["latency"] = latency

        return traffic, latency
