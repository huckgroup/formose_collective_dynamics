import os

from pathlib import Path
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

row_map = {"NaOH": 0, "formaldehyde": 1, "glycolaldehyde": 2, "DHA": 3}  # "CaCl2": 0,
sns.set_theme(
    style="ticks",
    context="paper",
    font_scale=0.8,
    rc={
        "axes.linewidth": 0.5,
        # "axes.facecolor": "#eceff4",
        "axes.edgecolor": "#2e3440",
        "axes.labelcolor": "#2e3440",
        "text.color": "#2e3440",
        "xtick.color": "#2e3440",
        "ytick.color": "#2e3440",
        "font.family": "Helvetica",
    },
)


def si_collective_dynamics_plots(
    df_embedding: pd.DataFrame,
    data: pd.DataFrame,
):
    isolation_colorp = [
        "#BF616A",
        "#56b4e9",
        "#0072b2",
        "#009e73",
    ]
    combined_colorp = [
        "#BF616A",
        "#e69f00",
        "#d55e00",
        "#cc79a7",
    ]
    class_mapper = dict(
        zip(
            df_embedding["label"]
            + "-"
            + df_embedding["compound"]
            + " "
            + df_embedding["type"],
            df_embedding["clusters"],
        )
    )
    data["clusters"] = data["trace"].map(class_mapper)

    for main_compound, section_group in data.groupby("compound"):
        fig, axs = plt.subplots(
            len(data["clusters"].unique()),
            4,
            layout="constrained",
            width_ratios=[8, 1.5, 1.5, 1.5],
            sharex="col",
            sharey="col",
            figsize=(17.8 / 2.54, 20 / 2.54),
        )

        for i, (class_nr, group) in enumerate(data.groupby("clusters")):
            i = class_nr - 1
            if len(group.trace.unique()) < 200:
                lw = 0.2
                alpha = 0.5
            else:
                lw = 0.1
                alpha = 0.5
            for (compound, section), compound_group in group.groupby(
                ["compound", "section"]
            ):
                local_ax = axs[i, 0]
                sns.lineplot(
                    compound_group,
                    x="Time (min)",
                    y="scaled_intensity",
                    ax=local_ax,
                    color="lightgrey",
                    units="label",
                    legend=False,
                    estimator=None,
                    alpha=alpha,
                    lw=lw,
                    zorder=5,
                )
                if compound == main_compound:
                    if section == "Isolation":
                        color = isolation_colorp[row_map[compound]]
                    else:
                        color = combined_colorp[row_map[compound]]
                    sns.lineplot(
                        compound_group,
                        x="Time (min)",
                        y="scaled_intensity",
                        ax=local_ax,
                        color=color,
                        units="label",
                        legend=False,
                        estimator=None,
                        alpha=alpha,
                        lw=lw,
                        zorder=6,
                    )
        compound_group = data[data.section == "Combined"]
        for i in range(len(df_embedding.clusters.unique())):
            for col in [1, 2, 3]:
                ax = axs[i, col]
                ax.scatter(
                    df_embedding["u1"],
                    df_embedding["u2"],
                    color="lightgrey",
                    alpha=0.6,
                    s=0.05,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                group = df_embedding[df_embedding.clusters == i + 1]
                if col == 1:
                    group = group[group["type"] == "Isolation"]
                    group = group[group["compound"] != "NaOH"]
                    color = isolation_colorp[row_map[main_compound]]
                    if i == 0:
                        ax.set_title("75 s input")
                if col == 2:
                    group = group[group["type"] == "Isolation"]
                    group = group[group["compound"] == "NaOH"]
                    color = isolation_colorp[row_map[main_compound]]
                    if i == 0:
                        ax.set_title("120 s input")
                if col == 3:
                    group = group[group["type"] == "Combined"]
                    color = combined_colorp[row_map[main_compound]]
                    if i == 0:
                        ax.set_title("Combined")

                ax.scatter(
                    group["u1"],
                    group["u2"],
                    s=1,
                    color="#4c566a",
                    marker="o",
                )
                ax.scatter(
                    group[group.compound == main_compound]["u1"],
                    group[group.compound == main_compound]["u2"],
                    s=1,
                    color=color,
                    marker="o",
                )

        [ax.set_ylabel("") for ax in axs[:, 0]]
        [ax.set_ylabel(f"Group {i+1}") for i, ax in enumerate(axs[:, 1])]
        [ax.set_ylim(-1.2, 1.2) for i, ax in enumerate(axs[:, 0])]
        axs[0, 0].set_xlim(10, 60)
        axs[-1, 0].set_xlabel("Time from input start (min)")
        fig.supylabel("Scaled intensity", fontsize="medium")
        fig.savefig(
            Path(os.environ["formose_collective_dynamics"])
            / f"figures/FAD17_18_si_{main_compound}_grouped_traces.png",
            dpi=300,
        )
        fig.savefig(
            Path(os.environ["formose_collective_dynamics"])
            / f"figures/FAD17_18_si_{main_compound}_grouped_traces.svg",
        )
        # plt.show()

        plt.close()

    return


def grouped_traces_organics_combined(df_embedding: pd.DataFrame, data: pd.DataFrame):
    class_mapper = dict(
        zip(
            df_embedding["label"]
            + "-"
            + df_embedding["compound"]
            + " "
            + df_embedding["type"],
            df_embedding["clusters"],
        )
    )
    data["clusters"] = data["trace"].map(class_mapper)

    data_w_naoh = data.copy(deep=True)
    data = data[data.compound != "NaOH"]
    isolation_colorp = [
        "#BF616A",
        "#56b4e9",
        "#0072b2",
        "#009e73",
    ]
    combined_colorp = [
        "#BF616A",
        "#e69f00",
        "#d55e00",
        "#cc79a7",
    ]

    fig, axs = plt.subplots(
        len(data["clusters"].unique()) + 1,
        3,
        layout="constrained",
        height_ratios=[0.1, *[1] * len(data["clusters"].unique())],
        sharex=False,
        sharey="row",
        figsize=(18 / 2.54, 12 / 2.54),
    )

    for i, (class_nr, group) in enumerate(data.groupby("clusters")):
        for j in range(3):
            ax = axs[i + 1, j].twinx()
            sns.lineplot(
                data=data_w_naoh[data_w_naoh["clusters"] == class_nr],
                x="Time (min)",
                y="scaled_intensity",
                units="trace",
                estimator=None,
                ax=ax,
                color="darkgrey",
                linewidth=0.03,
                # alpha=0.2,
            )
            ax.set_ylabel("")
            ax.set_xlim(10, 60)
            if not j == 2:
                ax.set_yticklabels([])

        for j, (compound, compound_group) in enumerate(group.groupby("compound")):
            if len(compound_group.trace.unique()) < 10:
                lw = 0.3
                alpha = 0.5
            else:
                lw = 0.10
                alpha = 0.5
            local_ax = axs[i + 1, row_map[compound] - 1].twinx()
            isolation_compounds = compound_group[
                compound_group["section"] == "Isolation"
            ]
            combined_compounds = compound_group[compound_group["section"] == "Combined"]
            if not isolation_compounds.empty:

                sns.lineplot(
                    isolation_compounds,
                    x="Time (min)",
                    y="scaled_intensity",
                    ax=local_ax,
                    color=isolation_colorp[row_map[compound]],
                    units="trace",
                    legend=False,
                    estimator=None,
                    lw=lw,
                )
            if not combined_compounds.empty:
                sns.lineplot(
                    combined_compounds,
                    x="Time (min)",
                    y="scaled_intensity",
                    ax=local_ax,
                    color=combined_colorp[row_map[compound]],
                    units="trace",
                    legend=False,
                    estimator=None,
                    lw=lw,
                )

            local_ax.set_ylabel("")
            local_ax.set_xlim(10, 60)
            if not row_map[compound] == 3:
                local_ax.set_yticklabels([])
            if i == 0:
                legend_ax = axs[0, row_map[compound] - 1]
                legend_ax.set_xticks([])
                legend_ax.set_yticks([])
                legend_ax.set_ylabel("")
                sns.despine(ax=legend_ax, bottom=True, left=True)
                c1 = isolation_colorp[row_map[compound]]
                c2 = combined_colorp[row_map[compound]]
                l1 = {"glycolaldehyde": "Gly", "formaldehyde": "FA", "DHA": "DHA"}[
                    compound
                ]
                l2 = {
                    "glycolaldehyde": "Gly + NaOH",
                    "formaldehyde": "FA + NaOH",
                    "DHA": "DHA + NaOH",
                }[compound]

                legend_ax.legend(
                    handles=[
                        *[
                            Line2D([0], [0], color=c, label=l)
                            for c, l in zip([c1, c2], [l1, l2])
                        ]
                    ],
                    loc="lower center",
                    bbox_to_anchor=(0.5, -2.5),
                    frameon=False,
                    ncols=2,
                )
                legend_ax.set_title(compound)

    for i, (class_nr, _) in enumerate(data.groupby("clusters")):
        loc_ax = axs[1 + i, 0]
        loc_ax.set_yticks([])
        loc_ax.set_ylabel(f"Gr. {class_nr}")

    compound_group = data[data.section == "Combined"]
    [ax.set_xlabel("") for ax in axs[:, :].flatten()]
    [ax.set_xticklabels("") for ax in axs[:-1, :].flatten()]
    [ax.set_ylim(-1.2, 1.2) for i, ax in enumerate(axs[1:, 0])]
    fig.supxlabel("Time from input start (min)", fontsize="medium", y=0.04)

    fig.text(
        s="Scaled intensity",
        x=0.99,
        y=0.5,
        rotation=270,
        horizontalalignment="center",
        verticalalignment="center",
        in_layout=True,
    )
    plt.tight_layout(h_pad=-0.2, w_pad=-0.15)
    fig.savefig(
        Path(os.environ["formose_collective_dynamics"])
        / f"figures/FAD17_18_grouped_traces_organics.png",
        dpi=600,
        transparent=True,
    )
    fig.savefig(
        Path(os.environ["formose_collective_dynamics"])
        / f"figures/FAD17_18_grouped_traces_organics.svg",
    )
    plt.close()

    # plt.show()

    return


def naoh_traces_only(
    data: pd.DataFrame,
    df_embedding: pd.DataFrame,
    save_path: Path,
):
    data_naoh = data[data.compound == "NaOH"]
    fig, axs = plt.subplots(
        1,
        figsize=(5 / 2.54, 4 / 2.54),
        layout="constrained",
        sharex="col",
    )
    sns.lineplot(
        data=data_naoh,
        x="Time (min)",
        y="scaled_intensity",
        units="label",
        estimator=None,
        ax=axs,
        color="#BF616A",
        linewidth=0.08,
        alpha=0.6,
    )
    axs.set_xlim(10, 60)
    axs.set_ylabel("Scaled Intensity")
    fig.savefig(save_path, dpi=600)
    plt.close()


def naoh_trace_panels_with_concentration(
    data: pd.DataFrame,
    df_embedding: pd.DataFrame,
    c_changes: pd.DataFrame,
    save_path: Path,
):
    class_mapper = dict(
        zip(
            df_embedding["label"]
            + "-"
            + df_embedding["compound"]
            + " "
            + df_embedding["type"],
            df_embedding["clusters"],
        )
    )
    data["cluster"] = data["trace"].map(class_mapper)
    data_naoh = data[data.compound == "NaOH"]
    fig, axs = plt.subplots(
        9,
        figsize=(8 / 2.54, 10 / 2.54),
        height_ratios=[0.5, *[1] * 8],
        layout="constrained",
        sharex="col",
    )
    c_changes["time"] -= 3420
    c_changes["time"] /= 60
    for i, cluster_n in enumerate(sorted(data.cluster.unique())):
        group = data_naoh[data_naoh.cluster == cluster_n]
        ax_line = axs[i + 1].twinx()
        group_all = data[data.cluster == cluster_n]
        if len(group.label.unique()) < 10:
            lw = 0.3
        else:
            lw = 0.15

        sns.lineplot(
            data=group_all,
            x="Time (min)",
            y="scaled_intensity",
            units="trace",
            estimator=None,
            ax=ax_line,
            color="lightgrey",
            linewidth=0.05,
        )

        if not group.empty:
            sns.lineplot(
                data=group,
                x="Time (min)",
                y="scaled_intensity",
                units="label",
                estimator=None,
                ax=ax_line,
                color="#BF616A",
                linewidth=lw,
                alpha=0.8,
            )
        ax_line.set_ylabel("")

    axs[0].set_xlim(10, 60)
    [ax.set_yticks([]) for ax in axs]
    [ax.set_ylabel(f"Gr. {i+1}") for i, ax in enumerate(axs[1:])]
    axs[0].set_ylabel("Conc.")
    naoh_ax = axs[0].twinx()
    naoh_ax.plot(c_changes["time"], c_changes["NaOH"], c="#BF616A")
    naoh_ax.set_ylabel("[NaOH]\n(mM)", rotation=270, verticalalignment="bottom")
    axs[-1].set_xlabel("Time (min)")
    fig.text(
        s="Scaled intensity",
        x=0.94,
        y=0.5,
        rotation=270,
        horizontalalignment="center",
        verticalalignment="center",
        in_layout=True,
    )

    plt.tight_layout(h_pad=-0.2)
    fig.savefig(save_path)
    plt.close()

    # plt.show()
    return


def main():
    project_dir = Path(os.environ["formose_collective_dynamics"])
    exp_code = "FAD17_18"
    concentration_changes = pd.read_csv(
        project_dir
        / "data"
        / exp_code
        / f"flowprofiles/{exp_code}_concentration_changes.csv"
    )
    # c_changes = pd.read_csv(exp_dir / f"flowprofiles/{exp_code}_flow_profiles.csv")
    concentration_changes.rename(
        lambda x: x.split(" ")[0], axis="columns", inplace=True
    )
    data = pd.read_parquet(
        project_dir / f"data/{exp_code}/{exp_code}_processed_data.parquet"
    )
    data.rename({"index": "time_idx"}, axis=1, inplace=True)

    data = data[
        ~(
            ((data.compound == "NaOH") | (data.section == "Combined"))
            & (data["snr"] <= 1.5)
        )
    ]
    data = data[~data.label.isin(["130.16_0.565", "186.22_0.681", "242.28_0.792"])]
    data = data[~((data.section == "Isolation") & (data["snr"] <= 1.5))]
    data["scaled_intensity"] = data["smooth_intensity"] - data.groupby(
        "trace"
    ).smooth_intensity.transform("min")
    data["scaled_intensity"] /= data.groupby("trace").scaled_intensity.transform("max")
    data["scaled_intensity"] *= 2
    data["scaled_intensity"] -= 1
    data["Time (min)"] = data["Time"] / 60

    # CaCl2 data is not considered in the analysis as there is no combined and isolation input
    data = data[data.compound != "CaCl2"]

    df_embedding = pd.read_csv(project_dir / "analysis_files/umap_coordinates.csv")
    si_collective_dynamics_plots(df_embedding, data)
    grouped_traces_organics_combined(df_embedding, data)
    naoh_traces_only(
        data,
        df_embedding,
        save_path=project_dir / "figures/naoh_traces_only.svg",
    )

    naoh_trace_panels_with_concentration(
        data,
        df_embedding,
        concentration_changes,
        save_path=project_dir / "figures/naoh_clusters_w_concentration.svg",
    )
    return


if __name__ == "__main__":
    main()
