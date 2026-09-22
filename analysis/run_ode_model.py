import os
from pathlib import Path
import numpy as np
import pandas as pd

import amici

# Adapted from: https://github.com/huckgroup/Formose_reservoir_computation
#


def run_model(
    flux_df: pd.DataFrame,
    proxy_compound: str,
    change_frequency: int,
    initial_state: np.ndarray,
    model,
) -> pd.DataFrame:
    solver = model.create_solver()
    model.set_free_parameter_by_name("kf", 0.5)
    model.set_initial_state(initial_state)
    model.set_timepoints(np.linspace(0, 10, 1000))
    rdata = model.simulate(solver=solver)
    time_points = np.linspace(flux_df.index[0], flux_df.index[-1], len(flux_df))
    data_points = []
    for time, naoh in flux_df[::change_frequency].items():
        model.set_timepoints(np.linspace(0, change_frequency / 60, change_frequency))
        initial_states = rdata["x"][-1]
        initial_states[model.get_state_ids().index(f"{proxy_compound}_in")] = naoh
        model.set_initial_state(initial_states)
        rdata = model.simulate(solver=solver)
        data_points.append(rdata["x"][:, :86])

    data = np.array(data_points).reshape(len(time_points), 86)
    data = pd.DataFrame(
        data=data, columns=model.get_state_ids()[:86], index=time_points
    )
    return data


def main():
    project_dir = Path(os.environ["formose_collective_dynamics"])
    exp_code = "FAD011"
    exp_dir = project_dir / "data" / exp_code
    flow = pd.read_csv(exp_dir / f"flowprofiles/{exp_code}_flow_profiles.csv")
    flow["time"] = flow["Time (s)"]
    flow["time"] = flow["time"] / 60
    flow.rename(lambda x: x.split(" ")[0], axis="columns", inplace=True)
    fluctuations = flow[["time", "NaOH", "CaCl2"]]
    fluctuations[["NaOH", "CaCl2"]] /= 0.8186111
    fluctuations.set_index("time")

    naoh_flux = fluctuations.loc[2220:7619]["NaOH"]
    cacl2_flux = fluctuations.loc[8520:13919]["CaCl2"]
    model_module = amici.import_model_module(
        "CCM_ext_Gly_mod",
        project_dir / "analysis_files/amici-CCM_ext_Gly_mod",
    )
    model = model_module.get_model()
    ORIGINAL_STATES = model.get_initial_state()
    initial_states = list(ORIGINAL_STATES)

    initial_states[-1] = 2
    model.set_initial_state(initial_states)
    solver = model.create_solver()

    model.set_free_parameter_by_name("kf", 0.5)
    model.set_timepoints(np.linspace(0, 1000, 1000))
    rdata = model.simulate(solver=solver)

    initial_state = rdata["x"][-1]
    naoh_data = run_model(
        naoh_flux, "ATP", change_frequency=120, initial_state=initial_state, model=model
    )
    cacl2_data = run_model(
        cacl2_flux,
        "NADP",
        change_frequency=45,
        initial_state=initial_state,
        model=model,
    )
    cacl2_data.to_csv(project_dir / f"analysis_files/{exp_code}_cacl2_ode_results.csv")
    naoh_data.to_csv(project_dir / f"analysis_files/{exp_code}_naoh_ode_results.csv")


if __name__ == "__main__":
    main()
