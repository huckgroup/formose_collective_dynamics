from sklearn.preprocessing import MinMaxScaler
import umap
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import os

from pathlib import Path
import pandas as pd


def cluster_umap(df: pd.DataFrame):

    dist_matrix = pdist(df[["u1", "u2"]], metric="cityblock")
    threshold = 2.8
    Z = linkage(
        dist_matrix, method="average", metric="cityblock", optimal_ordering=True
    )
    dn = dendrogram(Z, color_threshold=threshold, no_plot=False)
    # Uncomment if you want to inspect the dendrogram
    # plt.show()

    df["clusters"] = "no cluster"
    df.loc[dn["leaves"], "clusters"] = dn["leaves_color_list"]
    df["clusters"] = df["clusters"].str[1:].astype(int)
    # Reorganize the clusters to improve the flow of the paper
    cluster_remap = {1: 2, 2: 8, 3: 4, 4: 1, 5: 3, 6: 6, 7: 7, 8: 5}
    df["clusters"] = df["clusters"].map(cluster_remap)

    return df  # , Z


def main():
    project_dir = Path(os.environ["formose_collective_dynamics"])
    exp_code = "FAD17_18"
    flow = pd.read_csv(
        project_dir
        / "data"
        / exp_code
        / "flowprofiles"
        / f"{exp_code}_flow_profiles.csv"
    )
    flow = flow.rename(
        columns=dict(zip(flow.columns, flow.columns.str.split(" ").str[0])),
        inplace=False,
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
    # CaCl2 data is not considered in the analysis as there is no combined and isolation input
    data = data[data.compound != "CaCl2"]
    data_snr_filtered = data.pivot(
        index="time_idx", columns="trace", values="smooth_intensity"
    )

    scaler = MinMaxScaler()
    X = scaler.fit_transform(data_snr_filtered.T)

    umapper = umap.UMAP(
        n_neighbors=25,
        min_dist=0.005,
        metric="correlation",
        n_components=2,
        random_state=3230,
    )
    embedding = umapper.fit_transform(X)

    df_embedding = pd.DataFrame(
        {
            "label": data_snr_filtered.columns.str.split("-").str[0],
            "compound": data_snr_filtered.columns.str.split("-")
            .str[1]
            .str.split(" ")
            .str[0],
            "type": data_snr_filtered.columns.str.split("-")
            .str[1]
            .str.split(" ")
            .str[1],
            "u1": embedding[:, 0],
            "u2": embedding[:, 1],
        }
    )
    df_embedding = cluster_umap(df_embedding)
    df_embedding.to_csv(
        project_dir / "analysis_files/umap_coordinates.csv",
        index=False,
    )

    return


if __name__ == "__main__":
    main()
