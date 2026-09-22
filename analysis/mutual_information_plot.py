import seaborn as sns
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt

compound_groups = {
    0: [
        "229.07_0.692",
        "199.06_0.664",
        "117.02_0.555",
        "155.03_0.564",
    ],
    1: [
        "204.05_0.665",
        "203.05_0.668",
        "260.07_0.658",
        "275.07_0.74",
        "233.06_0.699",
    ],
    2: [
        "369.07_0.678",
        "384.08_0.687",
        "354.07_0.672",
    ],
}
sns.set_theme(
    style="ticks",
    context="paper",
    font_scale=0.9,
    rc={
        "axes.linewidth": 0.5,
        "axes.edgecolor": "#2e3440",
        "axes.labelcolor": "#2e3440",
        "text.color": "#2e3440",
        "xtick.color": "#2e3440",
        "ytick.color": "#2e3440",
        "font.family": "Helvetica",
        "svg.fonttype": "none",
    },
)

nord_palette = [
    "#bf616a",
    "#ebcb8b",
    "#a3be8c",
    "#b48ead",
    "#5e81ac",
    "#8fbcbb",
    "#d08770",
    "#88c0d0",
    "#81a1c1",
    "#2e3440",
    "#3b4252",
    "#434c5e",
    "#4c566a",
    "#d8dee9",
    "#e5e9f0",
    "#eceff4",
]


def si_maximum_mi_condition(mi_df: pd.DataFrame, project_dir: Path):
    mi_df["section_name"] = mi_df.input_compound + mi_df.section.map(
        {0: " Isolation", 2: " Combined"}
    )
    max_vals = mi_df.groupby(
        ["input_compound", "section_name", "trace_label"], as_index=False
    ).mutual_info.max()
    max_dict = dict()
    for i, (compound, group) in enumerate(max_vals.groupby("input_compound")):
        group_wf = group.pivot(
            index="trace_label", columns="section_name", values="mutual_info"
        )
        max_dict[compound] = group_wf
    fig, axs = plt.subplots(
        1,
        2,
        sharex=True,
        sharey=True,
        figsize=(12 / 2.54, 6 / 2.54),
        layout="constrained",
    )
    for i in range(2):
        j = [1, 0][i]
        axs[i].scatter(
            max_dict["NaOH"].iloc[:, j],
            max_dict["CaCl2"].iloc[:, j],
            alpha=0.5,
            color="silver",
            s=5,
        )
        axs[i].plot([0, 2], [0, 2], "--", color="#9E9E9E")
        axs[i].set_title(["Combined", "Isolated"][j])
        for group_number, label_list in compound_groups.items():
            for label in label_list:
                axs[i].scatter(
                    max_dict["NaOH"].loc[max_dict["NaOH"].index == label].iloc[:, j],
                    max_dict["CaCl2"].loc[max_dict["CaCl2"].index == label].iloc[:, j],
                    color=nord_palette[group_number],
                    s=20,
                )
        axs[i].set_xlim([0, 1.85])
        axs[i].set_ylim([0, 1.85])
    fig.supylabel("max(MI CaCl$_2$)")
    fig.supxlabel("max(MI NaOH)")
    fig.savefig(project_dir / "figures/FAD011_mutual_information_scatter.png", dpi=300)
    fig.savefig(project_dir / "figures/FAD011_mutual_information_scatter.svg")
    plt.show()


def mutual_information_sets(mi_df: pd.DataFrame, project_dir: Path):
    fig, axs = plt.subplots(
        4,
        len(compound_groups),
        figsize=(12.1 / 2.54, 10 / 2.54),
        sharey=True,
        sharex=True,
        layout="constrained",
    )
    mi_df.section = mi_df.section.map({0: "Isolation", 2: "Combined"})
    mi_df = mi_df[mi_df.dt.between(-5, 2)]
    for i, ((section, compound), group) in enumerate(
        mi_df.groupby(["section", "input_compound"], sort=False)
    ):
        for j, group_traces in compound_groups.items():
            row = {"NaOH": 0, "CaCl2": 2}[compound] + {"Isolation": 0, "Combined": 1}[
                section
            ]
            ax = axs[row, j]
            sns.lineplot(
                group,
                ax=ax,
                x="dt",
                y="mutual_info",
                units="trace_label",
                estimator=None,
                color="grey",
                lw=0.03,
                alpha=1,
                legend=False,
            )

            ax.vlines(
                x=0,
                ymin=0,
                ymax=1.9,
                linestyles="dashed",
                # alpha=0.6,
                color="#4C566A",
                lw=0.8,
            )
            ax.set_xlim(-5, 2)
            ax.set_ylim(0, 1.85)
            for label in group_traces:
                df = group.loc[group["trace_label"] == label]
                ax.plot(df.dt, df["mutual_info"], color=nord_palette[j], lw=1)
            if j == 0:
                ax.set_ylabel(f"{section}\nInput\n{compound}\nCoupling")
            ax.set_xlabel("$\delta$t (min)")
            if i == 0:
                ax.set_title(f"Set {j+1}")
    fig.supylabel("I(input;trace) (nats)", fontsize="medium")
    plt.tight_layout(h_pad=0.6, w_pad=-0.2)
    fig.savefig(project_dir / "figures/mutual_information_sets.png", dpi=300)
    fig.savefig(project_dir / "figures/mutual_information_sets.svg")

    plt.show()
    return


def main():
    project_dir = Path(os.environ["formose_collective_dynamics"])
    mi_df = pd.read_csv(project_dir / "analysis_files/mutual_info_FAD011.csv")
    mi_df = mi_df[
        ~mi_df.trace_label.isin(["130.16_0.565", "186.22_0.681", "242.28_0.792"])
    ]

    mi_df["dt"] /= 60
    mi_df.loc[mi_df["section"] == 1, "section"] = 0

    si_maximum_mi_condition(mi_df, project_dir)
    mutual_information_sets(mi_df, project_dir)

    return


if __name__ == "__main__":
    main()
