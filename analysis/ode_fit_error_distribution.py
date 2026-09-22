import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import os
from matplotlib.lines import Line2D

sns.set_theme(
    style="ticks",
    context="paper",
    font_scale=0.7,
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
    "#b48ead",
    "#a3be8c",
    "#d08770",
    "#8fbcbb",
    "#88c0d0",
    "#81a1c1",
    "#5e81ac",
    "#2e3440",
    "#3b4252",
    "#434c5e",
    "#4c566a",
    "#d8dee9",
    "#e5e9f0",
    "#eceff4",
]

sns.set_palette(nord_palette)


def main():
    project_dir = Path(os.environ["formose_collective_dynamics"])
    exp_code = "FAD011"
    df = pd.read_csv(project_dir / f"analysis_files/{exp_code}_errors.csv", index_col=0)
    df["input_compound"] = df["input_compound"].map(
        {"naoh": "NaOH", "cacl2": "CaCl$_2$"}
    )
    df["section"] = df["section"].map({0: "Isolation", 1: "Isolation", 2: "Combined"})
    df["section_label"] = df["input_compound"] + " " + df["section"]
    print(df)
    for section, group in df.groupby(["section", "section_label"]):
        print(section, group[group["test_mase"] < 1].shape, group["test_mase"].mean())
    fig, ax = plt.subplots(figsize=(7.1 / 2.54, 6.6 / 2.54))
    sns.kdeplot(
        df[df["section"] == "Isolation"],
        x="test_mase",
        hue="section_label",
        lw=2,
        alpha=0.8,
        ax=ax,
    )
    sns.kdeplot(
        df[df["section"] == "Combined"],
        x="test_mase",
        hue="section_label",
        lw=2,
        ax=ax,
        linestyle="--",
        alpha=0.8,
    )
    ax.set_xlabel("Test MASE")
    ax.set_xlim(left=0)
    linestyles = {"Isolation": "-", "Combined": "--"}
    color = {"NaOH": nord_palette[0], "CaCl$_2$": nord_palette[1]}
    labels = list(df.section_label.unique())
    labels.sort()
    labels = np.flip(labels)
    lines = [
        Line2D(
            [0],
            [0],
            color=color[label.split(" ")[0]],
            linestyle=linestyles[label.split(" ")[1]],
        )
        for label in labels
    ]
    plt.legend(lines, labels)
    for i, kdeline in enumerate(ax.lines):
        style = kdeline._linestyle

        if style == "-":
            group = df[df["section"] == "Isolation"]
        else:
            group = df[df["section"] == "Combined"]

        color = kdeline._color
        # Manually select the right line based on color
        if color == (0.9215686274509803, 0.796078431372549, 0.5450980392156862, 0.8):
            group = group[group["input_compound"] == "CaCl$_2$"]
        else:
            group = group[group["input_compound"] == "NaOH"]

        mean = group["test_mase"].median()
        xs = kdeline.get_xdata()
        ys = kdeline.get_ydata()
        height = np.interp(mean, xs, ys)
        ax.vlines(mean, 0, height, color=color, ls=style, lw=1, alpha=1)
    plt.tight_layout()
    plt.savefig(project_dir / "figures/FAD011_error_distribution.svg")
    plt.savefig(
        project_dir / "figures/FAD011_error_distribution.png", dpi=300, transparent=True
    )

    plt.show()


if __name__ == "__main__":
    main()
