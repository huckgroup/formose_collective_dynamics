import pandas as pd
from pathlib import Path
import os
import tomllib
from sklearn.feature_selection import mutual_info_regression
import numpy as np


def main():
    project_dir = Path(os.environ["formose_collective_dynamics"])
    exp_code = "FAD011"
    exp_dir = project_dir / "data" / exp_code
    data_path = exp_dir / f"{exp_code}_processed.parquet"
    meta = tomllib.loads(Path(exp_dir / f"{exp_code}_meta.toml").read_text())
    flow = pd.read_csv(exp_dir / f"flowprofiles/{exp_code}_concentration_changes.csv")
    flow.rename(lambda x: x.split(" ")[0], axis="columns", inplace=True)
    # flow time correction was calculated on DHA pulses preceding the experiment
    flow["time"] = flow["time"] - meta["flow_time_correction"]

    df = pd.read_parquet(data_path)
    df["smooth_intensity"] = (
        df.groupby("label", sort=False, as_index=False)["normalized_intensity"]
        .rolling(15, center=True, win_type="triang")
        .mean()
        .normalized_intensity
    )
    section_inputs = [["NaOH"], ["CaCl2"], ["NaOH", "CaCl2"]]
    dt_range = range(-300, 100, 5)
    correlation_dict = {
        "input_compound": [],
        "trace_label": [],
        "dt": [],
        "section": [],
        "mutual_info": [],
    }

    for i, time_interval in enumerate(meta["flow_timings"]):
        time_interval[0] += 480
        time_interval[-1] -= 480
        for compound in section_inputs[i]:
            for dt in dt_range:
                data = df[df["Time"].between(*time_interval)]
                y_index = np.abs(
                    data["Time"].unique()[:, None] - flow["time"].values
                ).argmin(axis=1)
                y_index += dt
                X = data.pivot(
                    index="Time", columns="label", values="smooth_intensity"
                ).fillna(0)
                flow[compound] = flow[compound].astype(str)
                mutual_info = mutual_info_regression(
                    X.values,
                    flow[compound].iloc[y_index].values,
                    random_state=230,
                )
                length = len(mutual_info)

                correlation_dict["input_compound"].extend([compound] * length)
                correlation_dict["trace_label"].extend(X.columns)
                correlation_dict["dt"].extend([dt] * length)
                correlation_dict["section"].extend([i] * length)
                correlation_dict["mutual_info"].extend(mutual_info)

    correlation_df = pd.DataFrame(correlation_dict)
    correlation_df.to_csv(
        project_dir / f"analysis_files/mutual_info_{exp_code}.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
