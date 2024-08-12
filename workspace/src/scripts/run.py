import os
import sys
sys.path.insert(0, "..")

from pathlib import Path
from tqdm.notebook import tqdm
from itertools import product

from src.accel.flat import Flat
from src.accel.flat_pe_proposal import FlatPEProposal
from src.accel.matmul import MatMul
from src.accel.proposal import Proposal
from src.accel.stall_proposal import StallProposal
from src.accel.unfused import Unfused
import src.utils.graph as graph
from src.utils.pareto import *


def attn(accel):
    output_dir = "../outputs/generated"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(output_dir + "/attn-" + accel + ".csv", "w") as f:
        f.write("model,seq_len,traffic,latency,energy,util_2d,util_1d,")

    models = ["BERT", "TrXL", "T5", "XLM"]
    seq_lens = ["1K", "4K", "16K", "64K", "256K", "1M"]

    started = False
    combinations = list(product(models, seq_lens))

    with tqdm(total=len(combinations),
              desc="Evaluating models",
              unit="combination",
              dynamic_ncols=True) as pbar:

        for model, seq_len in combinations:
            # Use tqdm.write instead of print to avoid interfering with the progress bar
            tqdm.write(f"Evaluating {model} on {seq_len} tokens")

            timeloop_dir = "../outputs/generated/attn/" + accel + "/" + model + "/" + seq_len
            if accel == "unfused":
                unfused = Unfused(model, seq_len)
                eval_stats = unfused.eval(timeloop_dir, run_mapper=True)
                # run_mapper is always True
                energy = unfused.eval_energy(timeloop_dir, "flat")
                util_stats = unfused.eval_utilization(timeloop_dir, run_mapper=True)

                # We do not need the per-Einsum 2D utilization
                names, utils_2d = [], []

            elif accel == "flat":
                timeloop_flat = Flat("cloud", model, seq_len, "../outputs/pregenerated/flat_validation.csv")
                eval_stats = timeloop_flat.eval(timeloop_dir, False)
                energy = timeloop_flat.eval_energy(timeloop_dir, "flat")
                util_stats = timeloop_flat.eval_utilization(timeloop_dir, False)
                names, utils_2d = timeloop_flat.eval_2d_util(timeloop_dir, False)

            elif accel == "cascade":
                proposal = FlatPEProposal(model, seq_len)
                eval_stats = proposal.eval(timeloop_dir)
                energy = proposal.eval_energy(timeloop_dir, "flat")
                util_stats = proposal.eval_utilization(timeloop_dir)
                names, utils_2d = proposal.eval_2d_util(timeloop_dir)

            elif accel == "arch":
                proposal = StallProposal(model, seq_len)
                eval_stats = proposal.eval(timeloop_dir)
                energy = proposal.eval_energy(timeloop_dir, "fusemax")
                util_stats = proposal.eval_utilization(timeloop_dir)
                names, utils_2d = proposal.eval_2d_util(timeloop_dir)

            elif accel == "binding":
                proposal = Proposal(model, seq_len)
                eval_stats = proposal.eval(timeloop_dir)
                energy = proposal.eval_energy(timeloop_dir, "fusemax")
                util_stats = proposal.eval_utilization(timeloop_dir)
                names, utils_2d = proposal.eval_2d_util(timeloop_dir)

            else:
                raise ValueError("Unknown accelerator " + accel)

            if not started:
                started = True

                with open(output_dir + "/attn-" + accel + ".csv", "a") as f:
                    f.write(",".join(names) + "\n")

            data = [model, seq_len, *eval_stats, energy, *util_stats, *utils_2d]

            with open(output_dir + "/attn-" + accel + ".csv", "a") as f:
                f.write(",".join([str(val) for val in data]) + "\n")

            # Update the progress bar
            pbar.update(1)


def end2end(platform):
    output_dir = "../outputs/generated"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(output_dir + "/end2end-" + platform + ".csv", "w") as f:
        f.write("model,seq_len,traffic,latency,energy\n")

    models = ["BERT", "TrXL", "T5", "XLM"]
    seq_lens = ["1K", "4K", "16K", "64K", "256K", "1M"]

    combinations = list(product(models, seq_lens))

    with tqdm(total=len(combinations),
              desc="Evaluating models",
              unit="combination",
              dynamic_ncols=True) as pbar:

        for model, seq_len in combinations:
            # Use tqdm.write instead of print to avoid interfering with the progress bar
            tqdm.write(f"Evaluating {model} on {seq_len} tokens")    

            timeloop_dir = "../outputs/generated/end2end/" + platform + "/" + model + "/" + seq_len
            matmul = MatMul(platform, model, seq_len)
            eval_stats = matmul.eval(timeloop_dir, run_mapper=True)
            energy = matmul.eval_energy(timeloop_dir)

            data = [model, seq_len, *eval_stats, energy]

            with open(output_dir + "/end2end-" + platform + ".csv", "a") as f:
                f.write(",".join([str(val) for val in data]) + "\n")

            # Update the progress bar
            pbar.update(1)

def pareto():
    output_dir = "../outputs/generated"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(output_dir + "/pareto.csv", "w") as f:
        f.write("accel,model,PE_dim,traffic,mem_lat,comp_2d_lat,comp_1d_lat,latency,array_2d_area,area\n")

    models = ["BERT", "TrXL", "T5", "XLM"]
    Es = [64, 64, 64, 128]
    models_Es =  zip(models, Es)
    dims = [2**(i + 4) for i in range(6)]

    combinations = list(product(models_Es, dims))

    with tqdm(total=len(combinations),
              desc="Evaluating models",
              unit="combination",
              dynamic_ncols=True) as pbar:

        for (model, E), PE_dim in combinations:
            multiplier = 1
            # Use tqdm.write instead of print to avoid interfering with the progress bar
            tqdm.write(f"Evaluating {model} on 256K tokens, with PE array {str(PE_dim)}x{str(PE_dim)}")
            while True:
                _, l3_sz = get_l3_sz(PE_dim, multiplier, E)

                proposal = Proposal(model, "256K", PE_dim=PE_dim, l3_sz=l3_sz)
                tl_cb = lambda spec, einsum: timeloop_arch_cb(spec, einsum, PE_dim, multiplier, E)
                ac_cb = lambda spec, array: accelergy_arch_cb(spec, array, PE_dim, multiplier, E)
                timeloop_dir = output_dir + "/pareto/" + model + "/" + str(PE_dim) + "/" + str(l3_sz // 2**10) + "K"

                result = proposal.eval_components(timeloop_dir, spec_callback=tl_cb)
                _, latency = proposal.eval(timeloop_dir, spec_callback=tl_cb)
                array_2d, area = proposal.eval_area(timeloop_dir, "fusemax", spec_callback=ac_cb)

                if result[1] > result[3]:
                    multiplier *= 2
                else:
                    break

            data = ["proposal", model, PE_dim, *result, latency, array_2d, area]
            with open(output_dir + "/pareto.csv", "a") as f:
                f.write(",".join([str(val) for val in data]) + "\n")

            if l3_sz > 32 * 2**20:
                break

            # Update the progress bar
            pbar.update(1)

def main():
    attn("unfused")
    attn("flat")
    attn("cascade")
    attn("arch")
    attn("binding")

    end2end("flat")
    end2end("proposal")

    pareto()

    graph.draw_bar_graph(graph.load_data("util_1d"), "Utilization 1D", "fig6a")
    graph.draw_bar_graph(graph.load_data("util_2d"), "Utilization 2D", "fig6b")
    graph.draw_breakdown()
    graph.draw_bar_graph(graph.load_data("latency", data_cb=lambda a, u: u / a), "Speedup", "fig8")
    graph.draw_bar_graph(graph.load_data("energy", data_cb=lambda a, u: a / u), "Energy Use", "fig9")
    graph.draw_bar_graph(graph.load_data("latency", kernel="end2end", data_cb=lambda a, u: u / a), "Speedup", "fig10")
    graph.draw_bar_graph(graph.load_data("energy", kernel="end2end", data_cb=lambda a, u: a / u), "Energy Use", "fig11")
    graph.draw_pareto()

if __name__ == "__main__":
    main()
