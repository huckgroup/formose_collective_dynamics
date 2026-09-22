import pandas as pd
from pathlib import Path
import os
from scipy.stats import pearsonr


def main():
    project_dir = Path(os.environ["formose_collective_dynamics"])
    exp_code = "FAD17_18"
    exp_dir = project_dir / "data" / exp_code
    concentration_changes = pd.read_csv(
        exp_dir / f"flowprofiles/{exp_code}_concentration_changes.csv"
    )
    concentration_changes.rename(
        lambda x: x.split(" ")[0], axis="columns", inplace=True
    )
    data = pd.read_parquet(
        project_dir / f"data/{exp_code}/{exp_code}_processed_data.parquet"
    )
    data.rename({"index": "time_idx"}, axis=1, inplace=True)
    df_embedding = pd.read_csv(project_dir / "analysis_files/umap_coordinates.csv")

    data = data[
        ~(
            ((data.compound == "NaOH") | (data.section == "Combined"))
            & (data["snr"] <= 1.5)
        )
    ]
    data = data[~((data.section == "Isolation") & (data["snr"] <= 1.5))]
    data = data[~data.label.isin(["130.16_0.565", "186.22_0.681", "242.28_0.792"])]
    data = data[data.compound != "CaCl2"]
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
    naoh_c_input = concentration_changes.loc[
        data[data.compound == "NaOH"].flow_idx.unique(), "NaOH"
    ]
    correlation_dict = {
        "trace_label": [],
        "pearsonr": [],
        "pearsonp": [],
        "cluster": [],
    }

    for cluster in [1, 2]:
        group = data[data.cluster == cluster]

        X = group.pivot(
            index="time_idx", columns="trace", values="smooth_intensity"
        ).fillna(0)

        pearson = [
            pearsonr(
                naoh_c_input.values,
                X.values[:, i],
            )
            for i in range(X.shape[1])
        ]
        length = len(pearson)

        correlation_dict["trace_label"].extend(X.columns)
        correlation_dict["pearsonr"].extend([p.statistic for p in pearson])
        correlation_dict["pearsonp"].extend([p.pvalue for p in pearson])
        correlation_dict["cluster"].extend([cluster] * length)
    df = pd.DataFrame(correlation_dict)
    print(df.groupby(["cluster"]).agg({"pearsonr": ["mean", "std"]}))


if __name__ == "__main__":
    main()
