from pathlib import Path
import numpy as np
import pandas as pd
import os


def calculate_steady_state_std(data: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    intermediate_sections = list(zip(meta.end_time[:-1], meta.start_time[1:]))
    steady_state_std = pd.DataFrame()
    for i, equilibrium_period in enumerate(intermediate_sections):
        section_data = data[
            data["Time"].between(equilibrium_period[0], equilibrium_period[1])
        ]
        wf_section = section_data.pivot(
            index="Time", columns="label", values="normalized_intensity"
        ).fillna(0)
        wf_section.columns = wf_section.columns.astype(str)
        wf_section = (
            wf_section.rolling(15, center=True, win_type="triang")
            .mean()
            .iloc[50:-50, :]
        )
        std_dataframe = wf_section.std().to_frame(name="steady_state_std").reset_index()
        std_dataframe["section"] = i
        steady_state_std = pd.concat((steady_state_std, std_dataframe))
    steady_state_std = (
        steady_state_std[steady_state_std.section != 3]
        .groupby("label", as_index=False)["steady_state_std"]
        .mean()
    )
    return steady_state_std


def split_and_smooth_data(data: pd.DataFrame, meta: pd.DataFrame, flow) -> pd.DataFrame:
    wf_data_parts = []
    timings = pd.DataFrame()
    flow_maps = dict()
    time_maps = dict()
    for row_i in meta.itertuples():
        subset = data[data["Time"].between(row_i.start_time, row_i.end_time)]
        subset = subset.assign(label=subset["label"].astype(str) + f"-{row_i.name}")
        wf_data_local = (
            subset.pivot(
                index=["Time", "flow_idx"],
                columns="label",
                values="normalized_intensity",
            )
            .fillna(0)
            .reset_index()
        )
        flow_maps[row_i.name] = dict(zip(wf_data_local.index, wf_data_local.flow_idx))
        time_maps[row_i.name] = dict(
            zip(wf_data_local.index, wf_data_local.Time - row_i.start_time)
        )
        # timing_df = wf_data_local[["Time", "flow_idx"]].reset_index()
        # timing_df.rename({"index":"t_idx"})
        # timing_df["section_type"] = row_i["name"]
        # timings.append

        wf_data_local.drop(["Time", "flow_idx"], axis=1, inplace=True)
        wf_data_parts.append(wf_data_local)

    wf_data = pd.concat(wf_data_parts, axis=1)
    wf_data_smooth = (
        wf_data.rolling(15, center=True, win_type="triang").mean().iloc[100:-100, :]
    )

    lf_data = wf_data_smooth.melt(
        value_name="smooth_intensity", var_name="trace", ignore_index=False
    ).reset_index()
    lf_data["label"] = lf_data.trace.str.split("-").str[0]
    lf_data[["compound", "section"]] = (
        lf_data.trace.str.split("-").str[1].str.split(" ", expand=True)
    )
    lf_data["flow_idx"] = (
        lf_data.groupby(["compound", "section"], sort=False)["index"]
        .apply(lambda x: x.map(flow_maps[f"{x.name[0]} {x.name[1]}"]))
        .values
    )
    lf_data["Time"] = (
        lf_data.groupby(["compound", "section"], sort=False)["index"]
        .apply(lambda x: x.map(time_maps[f"{x.name[0]} {x.name[1]}"]))
        .values
    )
    return lf_data


def main():
    project_dir = Path(os.environ["formose_collective_dynamics"])
    exp_code = "FAD17_18"
    exp_dir = project_dir / "data" / exp_code

    data = pd.read_parquet(exp_dir / f"{exp_code}_processed.parquet")
    meta = pd.read_csv(exp_dir / "meta.csv")
    flow = pd.read_csv(exp_dir / "flowprofiles" / f"{exp_code}_flow_profiles.csv")
    flow = flow.rename(
        columns=dict(zip(flow.columns, flow.columns.str.split(" ").str[0])),
        inplace=False,
    )

    y_index = np.abs(data["Time"].unique()[:, None] - flow["Time"].values).argmin(
        axis=1
    )
    idx_map = dict(zip(data["Time"].unique(), y_index))
    data["flow_idx"] = data.Time.map(idx_map)

    processed_data = split_and_smooth_data(data, meta, flow)
    steady_state_std = calculate_steady_state_std(data, meta)
    processed_data["snr"] = (
        processed_data.groupby(["label", "compound", "section"])[
            "smooth_intensity"
        ].transform(lambda x: x.std())
        / pd.merge(processed_data, steady_state_std, on="label").steady_state_std
    )
    processed_data.to_parquet(
        project_dir / "data" / f"{exp_code}/{exp_code}_processed_data.parquet",
        index=False,
    )


if __name__ == "__main__":
    main()
