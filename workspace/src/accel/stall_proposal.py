from src.accel.proposal import Proposal

class StallProposal(Proposal):
    def __init__(self, model, seq_len, PE_dim=256, l3_sz=16 * 2**20):
        super().__init__(model, seq_len, PE_dim=PE_dim, l3_sz=l3_sz)

    def eval(self, output_dir, spec_callback=None):
        if "results" not in self.computed:
            self.eval_per_einsum(output_dir, spec_callback=spec_callback)

        results = self.computed["results"]

        # Use the TeAAL heuristics to combine the results
        traffic = sum(result[0] for result in results.values())

        # Mem-bound latency
        mem_latency = sum(result[1] for result in results.values())

        # Compute latency without interleaving
        qk_lm = results["QK"][2] + results["LM"][2]

        # Stall to wait for RM
        stall = results["LM"][2] * self.PE_dim
        rm = results["RM"][2]

        sln_sld = results["SLN"][2] + results["SLD"][2]
        prm_spd = results["PRM"][2] + results["SPD"][2]

        slnv = results["SLNV"][2]
        spnv_rnv = results["SPNV"][2] + results["RNV"][2]

        rd_av = results["RD"][2] + results["AV"][2]

        comp_latency = max(qk_lm, rd_av) + stall + rm + max(sln_sld, prm_spd) + max(slnv, spnv_rnv)

        latency = max(comp_latency, mem_latency)

        self.computed["latency"] = latency

        return traffic, latency

