import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
import tomllib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from matplotlib.ticker import MaxNLocator

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


def train_test_plot(
    y_data: pd.DataFrame,
    target_data: pd.DataFrame,
    predict_data: pd.DataFrame,
    train_idx: np.ndarray[int],
    test_idx: np.ndarray[int],
    ax: mpl.axes.Axes,
):
    ax.plot(y_data, target_data, "--", color="#BF616A", label="Target")
    ax.plot(train_idx, predict_data.loc[train_idx], color="#B48EAD", label="Train")
    ax.plot(test_idx, predict_data.loc[test_idx], color="#5E81AC", label="Test")
    ymin = ax.get_ylim()[0]
    ymax = ax.get_ylim()[1]
    ax.vlines(
        x=train_idx[-1],
        ymin=ymin,
        ymax=ymax,
        linestyles="dashed",
        alpha=0.6,
        color="#4C566A",
    )
    ax.hlines(
        y=ymin,
        xmin=y_data[0],
        xmax=train_idx[-1],
        color="#B48EAD",
        lw=6,
        alpha=0.8,
    )
    ax.hlines(
        y=ymin,
        xmin=train_idx[-1],
        xmax=y_data[-1],
        color="#5E81AC",
        lw=6,
        alpha=0.8,
    )
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(y_data[0], y_data[-1])
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=1))

    return ax


def main():
    project_dir = Path(os.environ["formose_collective_dynamics"])
    exp_code = "FAD011"
    exp_dir = project_dir / "data" / exp_code
    data_path = exp_dir / f"{exp_code}_processed.parquet"
    meta = tomllib.loads(Path(exp_dir / f"{exp_code}_meta.toml").read_text())
    naoh_target = pd.read_csv(
        project_dir / f"analysis_files/{exp_code}_naoh_ode_results.csv", index_col=0
    )
    naoh_target = naoh_target.loc[:, naoh_target.diff().sum() != 0]
    naoh_target -= naoh_target.iloc[550:5500, :].mean(axis=0)
    naoh_target /= naoh_target.iloc[550:5500, :].abs().max(axis=0)

    naoh_target.index -= meta["flow_time_correction"]
    cacl2_target = pd.read_csv(
        project_dir / f"analysis_files/{exp_code}_cacl2_ode_results.csv", index_col=0
    )
    cacl2_target.index -= meta["flow_time_correction"]
    cacl2_target = cacl2_target.loc[:, cacl2_target.diff().sum() != 0]
    cacl2_target -= cacl2_target.iloc[550:5500, :].mean(axis=0)
    cacl2_target /= cacl2_target.iloc[550:5500, :].abs().max(axis=0)

    target_dict = {"naoh": naoh_target, "cacl2": cacl2_target}
    flow_diff = {
        "naoh": 12600,
        "cacl2": 6300,
    }  # the time difference between the first and second flow set.

    df = pd.read_parquet(data_path)
    df["smooth_intensity"] = (
        df.groupby("label", sort=False, as_index=False)["normalized_intensity"]
        .rolling(15, center=True, win_type="triang")
        .mean()
        .normalized_intensity
    )
    df = df[~df.label.isin(["130.16_0.565", "186.22_0.681", "242.28_0.792"])]
    section_inputs = [["naoh"], ["cacl2"], ["naoh", "cacl2"]]
    outputs = pd.DataFrame()
    target_values = pd.DataFrame()
    example_compounds = ["FAD", "GLYp"]

    for i, time_interval in enumerate(meta["flow_timings"]):
        time_interval[0] += 600
        data = df[df["Time"].between(*time_interval)]

        for compound in section_inputs[i]:
            regressor = Pipeline(
                [
                    ("scaling", StandardScaler(with_std=False)),
                    (
                        "regression",
                        MultiOutputRegressor(
                            linear_model.LassoCV(max_iter=10000),
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
            y = target_dict[compound]
            if i == 2:
                y.index += flow_diff[compound]

            y_index = np.abs(data["Time"].unique()[:, None] - y.index.values).argmin(
                axis=1
            )
            train_flow = y.iloc[y_index]
            train_flow = train_flow[example_compounds]
            X = data.pivot(index="Time", columns="label", values="smooth_intensity")
            X.fillna(0, inplace=True)
            X_train, X_test, y_train, y_test = train_test_split(
                X, train_flow.index, test_size=0.4, shuffle=False
            )

            regressor.fit(X_train, train_flow.loc[y_train])
            y_predict = regressor.predict(X)
            y_predict = pd.DataFrame(
                y_predict,
                index=train_flow.index,
                columns=train_flow.columns,
            )
            y_predict["time"] = train_flow.index
            tf = train_flow
            tf["time"] = tf.index
            tf = pd.melt(
                tf,
                id_vars="time",
                value_vars=example_compounds,
                var_name="target_compound",
                value_name="target_value",
            )
            tf.time -= y_predict.time.min()
            tf.time /= 60
            if i < 2:
                tf["section"] = "Isolated"
            else:
                tf["section"] = "Combined"

            tf["input_compound"] = compound

            target_values = pd.concat((target_values, tf))
            y_predict["split"] = "train"
            y_predict.loc[y_test, "split"] = "test"
            y_predict = pd.melt(
                y_predict,
                id_vars=["time", "split"],
                value_vars=example_compounds,
                var_name="target_compound",
                value_name="predicted_value",
            )
            if i < 2:
                y_predict["section"] = "Isolated"
            else:
                y_predict["section"] = "Combined"
            y_predict["input_compound"] = compound
            y_predict["time"] -= y_predict.time.min()

            y_predict["time"] /= 60
            outputs = pd.concat((outputs, y_predict))

    fig, axs = plt.subplots(
        9,
        2,
        height_ratios=[1, 0.2, 1, 0.2, 0.2, 1, 0.2, 1, 0.2],
        figsize=(10 / 2.54, 7 / 2.54),
        layout="constrained",
    )
    compound_colors = {"naoh": "#BF616A", "cacl2": "#EBCB8B"}

    for j, target_compound in enumerate(example_compounds):
        for i, input_compound in enumerate(["cacl2", "naoh"]):
            sns.despine(ax=axs[4, i], bottom=True, left=True)
            axs[4, i].set_xticks([])
            axs[4, i].set_yticks([])
            for k, source in enumerate(["isolation", "combined"]):
                ax = axs[j * 2 + k * 5, i]
                data = outputs[
                    (outputs.target_compound == target_compound)
                    & (outputs.input_compound == input_compound)
                ]
                target_data = target_values[
                    (target_values.target_compound == target_compound)
                    & (target_values.input_compound == input_compound)
                ]
                linewidth = 1
                ax.plot(
                    target_data[target_data.section == "Isolated"].time,
                    target_data[target_data.section == "Isolated"].target_value,
                    # "--",
                    lw=linewidth,
                    color=compound_colors[input_compound],
                    label="Target",
                )
                if source == "isolation":
                    ax.plot(
                        data[
                            (data.split == "train") & (data.section == "Isolated")
                        ].time,
                        data[
                            (data.split == "train") & (data.section == "Isolated")
                        ].predicted_value,
                        "--",
                        lw=linewidth,
                        color="#5e81ac",
                        label="Train",
                        alpha=1,
                    )
                    ax.plot(
                        data[
                            (data.split == "test") & (data.section == "Isolated")
                        ].time,
                        data[
                            (data.split == "test") & (data.section == "Isolated")
                        ].predicted_value,
                        "--",
                        lw=linewidth,
                        color="#a3be8c",
                        label="Test",
                        alpha=1,
                    )

                if source == "combined":
                    ax.plot(
                        data[
                            (data.split == "train") & (data.section == "Combined")
                        ].time,
                        data[
                            (data.split == "train") & (data.section == "Combined")
                        ].predicted_value,
                        "--",
                        lw=linewidth,
                        color="#5e81ac",
                        label="Train",
                        alpha=1,
                    )
                    ax.plot(
                        data[
                            (data.split == "test") & (data.section == "Combined")
                        ].time,
                        data[
                            (data.split == "test") & (data.section == "Combined")
                        ].predicted_value,
                        "--",
                        lw=linewidth,
                        color="#a3be8c",
                        label="Test",
                        alpha=1,
                    )
                ymin = ax.get_ylim()[0]
                ymax = ax.get_ylim()[1]
                split_x = data[data["split"] == "train"].time.max()
                ax.vlines(
                    x=split_x,
                    ymin=ymin,
                    ymax=ymax,
                    linestyles="dashed",
                    alpha=0.6,
                    color="#4C566A",
                )
                ax.hlines(
                    y=ymin,
                    xmin=data.time.min(),
                    xmax=split_x,
                    color="#5e81ac",
                    lw=4,
                    alpha=0.8,
                )
                ax.hlines(
                    y=ymin,
                    xmin=split_x,
                    xmax=data.time.max(),
                    color="#a3be8c",
                    lw=4,
                    alpha=0.8,
                )
                ax.set_ylim(ymin, ymax)
                ax.set_xlim(data.time.min(), data.time.max())
                if i == 0:
                    ax.set_ylabel(f"Signal\n(a.u.)")

    for j, target_compound in enumerate(example_compounds):
        for i, input_compound in enumerate(["cacl2", "naoh"]):
            for k, source in enumerate(["Isolated", "Combined"]):
                ax = axs[j * 2 + k * 5 + 1, i]
                data = outputs[
                    (outputs.target_compound == target_compound)
                    & (outputs.input_compound == input_compound)
                    & (outputs.section == source)
                ]
                target_data = target_values[
                    (target_values.target_compound == target_compound)
                    & (target_values.input_compound == input_compound)
                    & (target_values.section == source)
                ]

                y_train_mean = np.mean(target_data[data.split == "train"].target_value)
                ase = np.abs(target_data.target_value - data.predicted_value) / np.mean(
                    np.abs(target_data.target_value - y_train_mean)
                )
                ax.plot(
                    data[data["split"] == "train"].time,
                    ase[data["split"] == "train"],
                    c="#5E81AC",
                )
                ax.plot(
                    data[data["split"] == "test"].time,
                    ase[data["split"] == "test"],
                    c="#A3BE8C",
                )
                if i == 0:
                    ax.set_ylabel("ASE")
                ax.set_xlim(data.time.min(), data.time.max())
                ymin = ax.get_ylim()[0]
                ymax = ax.get_ylim()[1]
                ax.set_ylim(0, 1.2)

    [ax.set_xlabel("") for ax in axs.flatten()]
    [ax.set_xticklabels([]) for ax in axs[:-1, :].flatten()]
    [axs[j, 0].set_xticks([]) for j in [0, 2, 5]]
    [axs[j, 1].set_xticks([]) for j in [0, 2, 5]]

    axs[-1, 0].set_xlabel("Time (min)")
    axs[-1, 1].set_xlabel("Time (min)")
    axs[0, 1].set_title("NaOH")
    axs[0, 0].set_title("CaCl$_2$")
    plt.tight_layout(h_pad=-0.4)
    plt.savefig(project_dir / f"figures/{exp_code}_ode_selected_compounds.svg")
    plt.savefig(project_dir / f"figures/{exp_code}_ode_selected_compounds.png", dpi=300)


if __name__ == "__main__":
    main()
