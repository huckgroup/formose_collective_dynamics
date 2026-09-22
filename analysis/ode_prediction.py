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
from matplotlib.ticker import LogFormatter, EngFormatter, MaxNLocator

sns.set_theme(
    style="ticks",
    context="paper",
    font_scale=0.7,
    rc={
        "axes.linewidth": 0.5,
        "axes.facecolor": "#eceff4",
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
    compound: str,
):
    c_color = {"cacl2": "#EBCB8B", "naoh": "#BF616A"}
    ax.plot(y_data, target_data, "--", color=c_color[compound], label="Target")
    ax.plot(train_idx, predict_data.loc[train_idx], color="#5E81AC", label="Train")
    ax.plot(test_idx, predict_data.loc[test_idx], color="#A3BE8C", label="Test")
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
        color="#5E81AC",
        lw=6,
        alpha=0.8,
    )
    ax.hlines(
        y=ymin,
        xmin=train_idx[-1],
        xmax=y_data[-1],
        color="#A3BE8C",
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
    naoh_target.index -= meta["flow_time_correction"]
    cacl2_target = pd.read_csv(
        project_dir / f"analysis_files/{exp_code}_cacl2_ode_results.csv", index_col=0
    )
    cacl2_target.index -= meta["flow_time_correction"]
    cacl2_target = cacl2_target.loc[:, cacl2_target.diff().sum() != 0]

    target_dict = {"naoh": naoh_target, "cacl2": cacl2_target}
    flow_diff = {
        "naoh": 12600,
        "cacl2": 6300,
    }  # the time difference between the first and second flow set.

    df = pd.read_parquet(data_path)
    df["smooth_intensity"] = (
        # print(
        df.groupby("label", sort=False, as_index=False)["normalized_intensity"]
        .rolling(15, center=True, win_type="triang")
        .mean()
        .normalized_intensity
    )
    df = df[~df.label.isin(["130.16_0.565", "186.22_0.681", "242.28_0.792"])]
    section_inputs = [["naoh"], ["cacl2"], ["naoh", "cacl2"]]
    errors = {
        "input_compound": [],
        "target_compound": [],
        "section": [],
        "test_mase": [],
        "train_mase": [],
    }
    regression_output = dict()

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
                            linear_model.LassoCV(
                                max_iter=10000,
                            ),
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
            y_predict["split"] = "train"
            y_predict.loc[y_test, "split"] = "test"

            fig, axs = plt.subplots(
                11,
                7,
                sharex=True,
                figsize=(18.2 / 2.54, 20 / 2.54),
                layout="constrained",
            )

            print(y.columns)
            y_predict.index -= train_flow.index[0]
            y_train -= train_flow.index[0]
            y_test -= train_flow.index[0]
            train_flow.index -= train_flow.index[0]
            for target_compound, ax in zip(y.columns, axs.flatten()):
                train_test_plot(
                    train_flow.index,
                    train_flow[target_compound],
                    predict_data=y_predict[target_compound],
                    train_idx=y_train,
                    test_idx=y_test,
                    ax=ax,
                    compound=compound,
                )
                axs[-1, 0].set_xlabel("Time (s)")
                ax.set_title(f"{target_compound}")

            section_l = ["Isolation", "Isolation", "Combined"]
            title_map = {"cacl2": "CaCl$_2$", "naoh": "NaOH"}
            fig.suptitle(f"{title_map[compound]} {section_l[i]}")
            plt.tight_layout(h_pad=-0.1, w_pad=-0.1)
            plt.savefig(
                project_dir / "figures" / f"{exp_code}_{compound}_{i}.png",
                dpi=300,
                transparent=True,
            )
            # plt.show()
            plt.close()

            for target_compound in y.columns:
                MAD_train = abs(
                    train_flow.loc[y_train, target_compound].values
                    - train_flow.loc[y_train, target_compound].values.mean()
                ).mean()
                train_mase = (
                    abs(
                        y_predict.loc[y_train, target_compound]
                        - train_flow.loc[y_train, target_compound]
                    ).mean()
                    / MAD_train
                )
                test_mase = (
                    abs(
                        y_predict.loc[y_test, target_compound]
                        - train_flow.loc[y_test, target_compound]
                    ).mean()
                    / MAD_train
                )
                errors["input_compound"].append(compound)
                errors["target_compound"].append(target_compound)
                errors["section"].append(i)
                errors["train_mase"].append(train_mase)
                errors["test_mase"].append(test_mase)
            regression_output[f"{compound}_{i}"] = regressor

    pd.DataFrame(errors).to_csv(project_dir / f"analysis_files/{exp_code}_errors.csv")


if __name__ == "__main__":
    main()
