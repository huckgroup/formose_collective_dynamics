# Emergence of Collective Dynamics in a Driven Prebiotic Chemical Network
This repository contains code and data relevant to the manuscript:
Emergence of Collective Dynamics in a Driven Prebiotic Chemical Network
T.J. de Jong, M.G. Baltussen, W.E. Robinson, W.T.S. Huck

# Installation
This code requires python 3.12, all requirements are listed in the pyproject.toml.
The recommended installation method is using `uv`.
First clone the directory locally, and make sure you have a c++ compiler installed ().  
On debian based systems run
```bash
sudo apt install build-essentials
```

Subsequently install the python environment using:
```bash
uv sync
```

> Amici (combined with tellurium and roadrunner) are needed to recreate the ODE results, if you do not wish to recreate this data they can be removed from the pyproject.toml to simplify installation.

You also need to add a formose_collective_dynamics environmental variable with the path to the project, the way to do this depends on your operating system.

```powershell
setx formose_collective_dynamics /path/to/directory/
```
for zsh run:
```zsh
echo "export formose_collective_dynamics=/path/to/directory/" >> ~/.zshrc
```
for bash run:
```bash
echo "export formose_collective_dynamics=/path/to/directory/" >> ~/.bashrc
```


# Structure
- `data` Contains preprocessed experimental files and other relevant metadata.
- `analysis` Contains code relevant 
- `analysis-files` Contains additional files produced by or required for analysis.
- `figures` Is the output container for the various figures produced by the code.

# Run instructions
To recreate the panels in figure to run in order:
```bash
uv run analysis/FAD17_18_preprocess_signal.py
```
This applies some standard preprocessing to the data.
```bash
uv run analysis/umap_embedding.py
```
This creates the umap embedding and clusters the traces.

```bash
uv run analysis/collective_dynamics_plot.py
```
Create the plots for figure 2

```bash
uv run analysis/sankey_diagrams.py
```
Creates the panels needed for figure 3

```bash
uv run analysis/compile_model.py
```
Compiles the amici model (requires amici and tellurium)

```bash
uv run analysis/run_ode_model.py
```
Runs the ode models with the dynamic inputs (requires amici)

```bash
uv run analysis/ode_prediction_example_fits.py
```
Creates the fitted traces for figure 5c and d


```bash
uv run analysis/ode_prediction.py
```
Creates the SM ODE figres and the results file for `ode_fit_error_distribution.py`

```bash
uv run analysis/ode_fit_error_distribution.py
```
Creates figure 5e
