import os
import plotly.graph_objects as go
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

row_map = {"NaOH": 0, "formaldehyde": 1, "glycolaldehyde": 2, "DHA": 3}


def umap_sankey_insets_line(
    df_embedding: pd.DataFrame, data: pd.DataFrame, project_dir: Path
):
    # The Sankey diagram as displayed in figure 3 is manually constructed.

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
    data["cluster"] = data["trace"].map(class_mapper)

    for (cluster_n, compound, section_type), group in df_embedding.groupby(
        ["clusters", "compound", "type"]
    ):
        fig, ax = plt.subplots(
            1, 1, figsize=(2.6 / 2.54, 1.9 / 2.54), layout="constrained"
        )
        line_data = data.query(
            "(cluster == @cluster_n) & (compound == @compound) & (section == @section_type)"
        )
        if section_type == "Isolation":
            c = isolation_colorp[row_map[compound]]
        else:
            c = combined_colorp[row_map[compound]]

        sns.lineplot(
            line_data,
            x="Time (min)",
            y="scaled_intensity",
            units="trace",
            legend=False,
            ax=ax,
            estimator=None,
            color=c,
            lw=0.1,
            alpha=0.6,
        )
        ax.facecolor = "white"
        ax.set_xlim([25, 45])
        ax.set_ylim([-1.1, 1.1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.text(0.9, 0.85, cluster_n, transform=ax.transAxes, fontsize=8)

        fig.savefig(
            project_dir
            / "figures/sankey_insets"
            / f"{compound}_{section_type}_{cluster_n}_traces.png",
            transparent=True,
            facecolor="white",
            dpi=300,
        )
        plt.close()

    return


def cluster_sankey(df: pd.DataFrame, project_dir: Path):
    compound_colors = {
        "DHA": "#A3BE8C",
        "formaldehyde": "#81A1C1",
        "glycolaldehyde": "#EBCB8B",
    }
    for compound in ["DHA", "formaldehyde", "glycolaldehyde"]:
        c_order = ["#BF616A", "#B48EAD", compound_colors[compound]]
        compound_df = df[(df.compound == compound) | (df.compound == "NaOH")]
        compound_df["section"] = (
            compound_df["compound"] + "_" + compound_df["type"]
        )  # + df["clusters"].astype(str)
        wf_clusts = pd.pivot(
            compound_df, index="label", columns="section", values="clusters"
        ).fillna(0)
        count_df = wf_clusts.groupby(
            [f"{compound}_Combined", "NaOH_Isolation"], as_index=False
        ).count()
        count_df_2 = wf_clusts.groupby(
            [f"{compound}_Isolation", f"{compound}_Combined"], as_index=False
        ).count()
        # Create group offset per type
        count_df[f"{compound}_Combined"] += 9
        count_df_2[f"{compound}_Combined"] += 9
        count_df_2[f"{compound}_Isolation"] += 18

        # Since SNR filtering is applied per group not all traces appear in each column.
        # These 'hidden' group make it so group sizes are correct
        hide = [0, 9, 18]
        fig = go.Figure(
            data=go.Sankey(
                node=dict(
                    label=[
                        "",
                        *[f"Group {j}" for j in range(1, 9)],
                        "",
                        *[f"Group {j}" for j in range(1, 9)],
                        "",
                        *[f"Group {j}" for j in range(1, 9)],
                    ],
                    color=[
                        (c_order[i // 9] if not n in hide else "rgba(0,0,0,0)")
                        for i, n in enumerate([*range(9), *range(9), *range(9)])
                    ],
                    line={"width": 0},
                ),
                link=dict(
                    source=[
                        *count_df["NaOH_Isolation"],
                        *count_df_2[f"{compound}_Combined"],
                    ],
                    target=[
                        *count_df[f"{compound}_Combined"],
                        *count_df_2[f"{compound}_Isolation"],
                    ],
                    value=[
                        *count_df[f"{compound}_Isolation"],
                        *count_df_2[f"NaOH_Isolation"],
                    ],
                    color=[
                        (
                            "rgba(0,0,0,0)"
                            if n[0] in hide or n[1] in hide
                            else ("lightgray")
                        )
                        for n in zip(
                            [
                                *count_df["NaOH_Isolation"],
                                *count_df_2[f"{compound}_Combined"],
                            ],
                            [
                                *count_df[f"{compound}_Combined"],
                                *count_df_2[f"{compound}_Isolation"],
                            ],
                        )
                    ],
                ),
            )
        )
        fig.add_annotation(
            x=0.0,  # Adjust x-position for the first column
            y=1.1,  # Above the diagram
            text="NaOH 120 s",
            showarrow=False,
            font=dict(size=14),
        )

        fig.add_annotation(
            x=0.5,  # Adjust x-position for the second column
            y=1.1,
            text=f"NaOH + {compound}",
            showarrow=False,
            font=dict(size=14),
        )

        fig.add_annotation(
            x=1,  # Adjust x-position for the third column
            y=1.1,
            text=f"{compound} 75 s",
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(title=compound)
        # fig.show()

        fig.update_layout(width=500, height=500)
        fig.write_image(project_dir / f"figures/sankey_{compound}.png")  # , dpi=300)
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

    cluster_sankey(df_embedding, project_dir)
    # umap_sankey_insets_line(df_embedding, data, project_dir)
    return


if __name__ == "__main__":
    main()
