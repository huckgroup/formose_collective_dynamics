import tellurium as te
import amici
import os
from pathlib import Path

# Adapted from: https://github.com/huckgroup/Formose_reservoir_computation


def main():
    project_dir = Path(os.environ["formose_collective_dynamics"])
    r = te.loadSBMLModel(
        "https://ftp.ebi.ac.uk/pub/databases/biomodels/repository/aaj/MODEL2010160002/3/CCM_ext_Gly.xml"
    )  # E. coli glycerol Oliveira 2020

    r.addParameter("kf", 1.0)
    for s, c in zip(r.getFloatingSpeciesIds(), r.getFloatingSpeciesConcentrations()):
        r.addSpeciesConcentration(
            f"{s}_in", compartment="cell", initConcentration=c, boundaryCondition=True
        )
        r.addReaction(f"F_{s}_in", [], [s], f"kf*{s}_in")
        r.addReaction(f"F_{s}_out", [s], [], f"kf*{s}")
    r.exportToSBML(project_dir / "analysis_files" / "CCM_ext_Gly_mod.xml")
    sbml_importer = amici.SbmlImporter(
        project_dir / "analysis_files" / "CCM_ext_Gly_mod.xml"
    )
    sbml_importer.sbml2amici(
        model_name="CCM_ext_Gly_mod",
        output_dir=project_dir / "analysis_files" / "amici-CCM_ext_Gly_mod",
    )

    return


if __name__ == "__main__":
    main()
