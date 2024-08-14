import os
import sys
sys.path.insert(0, os.getcwd())

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns

from src.utils.csv_utils import CSVUtils

def load_data(col, kernel="attn", raw_cb=None, data_cb=None, skip=set(), experiment_dir="../outputs/generated/default"):

    results_dir = Path(experiment_dir) / "results"

    accels = {"unfused": "Unfused", "flat": "FLAT", "cascade": "+Cascade", "arch": "+Architecture", "binding": "+Binding"}
    archs = {"unfused": "flat", "flat": "flat", "cascade": "flat", "arch": "proposal", "binding": "proposal"}

    data = {"model": [], "seq_len": [], "Accelerator": [], "data": []}

    unfused = []
    for accel in accels:
        if accel in skip:
            continue

        reader = CSVUtils(results_dir / f"attn-{accel}.csv")

        attn_csv = reader.get_all()
        i = attn_csv[0].index(col)

        if kernel == "end2end":
            reader = CSVUtils(results_dir / f"end2end-{archs[accel]}.csv")
            end2end_csv = reader.get_all()
            k = end2end_csv[0].index(col)


        models = []
        seq_lens = []
        for j, line in enumerate(attn_csv[1:]):
            if line[0] in skip:
                if accel == "unfused":
                    unfused.append(None)
                continue

            if line[1] in skip:
                if accel == "unfused":
                    unfused.append(None)
                continue

            val = line[i]
            if kernel == "end2end":
                val += end2end_csv[j + 1][k]

            if raw_cb is not None:
                val = raw_cb(val, accel)

            if accel == "unfused":
                unfused.append(val)

            if data_cb is not None:
                val = data_cb(val, unfused[j])

            data["model"].append(line[0])
            data["seq_len"].append(line[1])
            data["Accelerator"].append(accels[accel])
            data["data"].append(val)

            if line[0] not in models:
                models.append(line[0])

            if line[1] not in seq_lens:
                seq_lens.append(line[1])

    # Add the dummy FuseMax
    if "proposal" not in skip:
        data["model"].append(models[-1])
        data["seq_len"].append(seq_lens[-1])
        data["Accelerator"].append("FuseMax")
        data["data"].append(0)

    return pd.DataFrame.from_dict(data)

def load_breakdown(experiment_dir):
    experiment_dir = Path(experiment_dir)
    results_dir = experiment_dir / "results"


    accels = {"flat": "FL", "cascade": "+C", "arch": "+A", "binding": "+B"}
    einsums = {"flat": ["QK", "AV"], "cascade": ["QK", "SLNV"], "arch": ["QK", "LM", "SLN", "SLD", "SLNV"], "binding": ["QK", "LM", "SLN", "SLD", "SLNV"]}

    data = {"seq_len": [], "Accelerator": [], "QK": [], "LM": [], "SLN": [], "SLD": [], "SLNV/AV": []}
    for accel in accels:
        reader = CSVUtils(results_dir / f"attn-{accel}.csv")

        attn_csv = reader.get_all()

        i = attn_csv[0].index(einsums[accel][0])

        for line in attn_csv[1:]:
            if line[0] != "BERT":
                continue

            data["seq_len"].append(line[1])
            data["Accelerator"].append(accels[accel])
            for j, einsum in enumerate(einsums[accel]):
                es = "SLNV/AV" if einsum in {"SLNV", "AV"} else einsum
                data[es].append(line[i + j])

                if len(einsums[accel]) == 2 and j == 0:
                    for es in ["LM", "SLN", "SLD"]:
                        data[es].append(0)

    return pd.DataFrame.from_dict(data)


def draw_bar_graph(data, ylabel, fn, ymax=None, experiment_dir="../outputs/generated/default"):
    figs_dir = Path(experiment_dir) / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Increase the font size
    fontsize = 20
    plt.rcParams.update({'font.size': fontsize})

    # Move dummy FuseMax to the end
    accels = list(data["Accelerator"].unique())
    if "FuseMax" in accels:
        accels.remove("FuseMax")
        accels.append("FuseMax")

    # Graph
    sns.set_style('whitegrid')
    g = sns.catplot(x="seq_len", hue="Accelerator", col="model", y="data",
                    data=data, kind="bar", height=4, aspect=1.3, palette='Set1',
                    hue_order=accels)

    # Set Y-label
    g.axes.flat[0].set_ylabel(ylabel)

    # Reorder legend
    g.legend.remove()
    handles, labels = g.axes.flat[-1].get_legend_handles_labels()
    if "FuseMax" in accels:
        handles[-1].set_visible(False)
        order = [0, 1, 5, 2, 3, 4]
    else:
        order = range(len(accels))

    # Format the legend
    legend = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
               loc="center left", bbox_to_anchor=(1, 0.5), fontsize=fontsize)
    legend.get_frame().set_facecolor('none')
    legend.get_frame().set_edgecolor('none')

    # Combine graphs
    for ax in g.axes.flat[1:]:
        sns.despine(ax=ax, left=True)

    for ax in g.axes.flat:
        # Set X-label
        ax.set_xlabel(ax.get_title().split("=")[1])
        ax.set_title('')
        ax.margins(x=0.05) # slightly more margin as a separation

        # Set Y-max
        if ymax is not None:
            ax.set_ylim(top=ymax)

    plt.subplots_adjust(wspace=0, bottom=0.18, left=0.06)

    plt.savefig(figs_dir / f"{fn}.pdf", format="pdf", bbox_inches="tight")

def draw_breakdown(experiment_dir="../outputs/generated/default"):
    figs_dir = Path(experiment_dir) / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    df = load_breakdown(experiment_dir)

    # Increase the font size
    fontsize = 20
    plt.rcParams.update({'font.size': fontsize})

    sns.set_style('whitegrid')

    fig, axes = plt.subplots(nrows = 1, ncols = 6, figsize = (25,4))
    for i, seq_len in enumerate(df["seq_len"].unique()):
        # for einsum in ["QK", "LM", "SLN", "SLD", "SLNV/AV"]:
        df_seq = df.loc[df["seq_len"] == seq_len]
        df_seq = df_seq.set_index("Accelerator")
        df_seq = df_seq.drop(["seq_len"], axis=1)
        df_seq.plot(kind="bar", stacked=True, ax=axes[i], width=0.75)

    # Set Y-label
    axes.flat[0].set_ylabel("Proportion Active")

    # Combine graphs
    sns.despine(fig=fig, left=True)

    # Remove y-ticks
    for ax in axes.flat[1:]:
        ax.tick_params(labelleft=False, left=False)

    for ax, seq_len in zip(axes.flat, df["seq_len"].unique()):
        # Set X-label
        ax.set_xlabel(seq_len)
        ax.set_title('')

        # Subplot separation
        plot_margin = 0.25
        x0, x1, y0, y1 = ax.axis()
        ax.axis((x0 - plot_margin,
                  x1 + plot_margin,
                  y0,
                  y1))

        # Set Y-max
        ax.set_ylim(top=1.0)

        # Remove X-gridlines
        ax.grid(visible=False, axis="x")

    # Remove legends
    for ax in axes.flat[:-1]:
        ax.get_legend().remove()

    # Format the legend
    legend = axes[-1].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=fontsize)
    legend.get_frame().set_facecolor('none')
    legend.get_frame().set_edgecolor('none')

    plt.subplots_adjust(wspace=0, bottom=0.18, left=0.06)

    plt.savefig(figs_dir / "fig7.pdf", format="pdf", bbox_inches="tight")

def draw_pareto(experiment_dir="../outputs/generated/default"):
    experiment_dir = Path(experiment_dir)

    results_dir = Path(experiment_dir) / "results"

    figs_dir = Path(experiment_dir) / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_dir / "pareto.csv")

    df = df.rename({"model": "Model"}, axis=1)

    # Convert from um^2 to cm^2
    df['area'] = df['area'].apply(lambda x: x * 10**-8)

    # Increase the font size
    fontsize = 15
    plt.rcParams.update({'font.size': fontsize})

    sns.set_style('whitegrid')
    plt.figure()
    sns.lineplot(data=df, x="area", y="latency", hue="Model", style="Model", markers=True)

    # Use log axes
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set axis titles
    ax.set_xlabel("Area (cm${}^2$)")
    ax.set_ylabel("Attention Latency (s)")

    plt.savefig(figs_dir / "fig12.pdf", format="pdf", bbox_inches="tight")
