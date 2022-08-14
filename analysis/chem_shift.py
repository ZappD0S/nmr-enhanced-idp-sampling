import numpy as np
import subprocess
import pandas as pd
import mdtraj as mdt
import tempfile
from pathlib import Path


def get_lines_to_skip(filename):
    """Determine the number of comment lines in a SPARTA+ output file."""
    # format_string = """FORMAT %4d %4s %4s %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f"""
    format_string = r"FORMAT %4d %4s %4s %9.3f %9.3f %9.3f %9.3f %9.3f %s"
    handle = open(filename)
    for i, line in enumerate(handle):
        if line.find(format_string) != -1:
            return i + 2

    raise Exception("No format string found in SPARTA+ file!")


names = [
    "resSeq",
    "resName",
    "name",
    "SS_SHIFT",
    "SHIFT",
    "RC_SHIFT",
    "HM_SHIFT",
    "EF_SHIFT",
    "SIGMA",
]

# TODO: make configurable
faspr_bin = "/home/gzappavigna/FASPR/FASPR"
# spartaplus_bin = Path("/home/gzappavigna/SPARTA+/bin/SPARTA+.static.linux9")
sparta_bin = Path("/home/gzappavigna/SPARTA/src/SPARTA")


def run_faspr(tempdir, pdbs):

    pdbs = []

    for pdb_cg in enumerate(pdbs):
        pdb_all = pdb_cg.parent / (pdb_cg.stem + "_all.pdb")

        subprocess.run([faspr_bin, "-i", str(pdb_cg), "-o", str(pdb_all)], check=True)
        assert pdb_all.exists()

        pdbs.append(pdb_all)

    return pdbs


def run_sparta(tempdir, pdbs):
    data = []

    for pdb in pdbs:
        pred_tab = pdb.parent / (pdb.stem + "_pred.tab")

        popen = subprocess.Popen(
            [sparta_bin]
            + ["-in", str(pdb)]
            + ["-sum", str(pred_tab)]
            + ["-spartaDir", str(sparta_bin.parents[1])],
            cwd=tempdir,
            stdout=subprocess.DEVNULL,
        )

        data.append((popen, pred_tab))

    preds = []

    for popen, pred_tab in data:
        assert popen.wait() == 0
        assert pred_tab.exists()
        preds.append(pred_tab)

    return preds


def build_df(preds):
    dfs = []

    for i, pred_tab in enumerate(preds):
        lines_to_skip = get_lines_to_skip(pred_tab)

        df = pd.read_table(
            pred_tab,
            names=names,
            header=None,
            sep=r"\s+",
            skiprows=lines_to_skip,
        )
        df["frame"] = i
        dfs.append(df)

    df = pd.concat(dfs)

    # if rename_HN:
    #     df.name[df.name == "HN"] = "H"

    df = df.pivot_table(
        index=["resSeq", "name"],
        columns="frame",
        values="SHIFT",
    )

    return df


def do_stuff(
    traj,
    period: float,  # in ps
    cg=False,
):

    one_every = round(period / traj.timestep)
    inds = np.arange(0, traj.time.size, one_every)

    subtraj = traj[inds]

    with tempfile.TemporaryDirectory() as tempdir_:
        tempdir = Path(tempdir_)

        pdbs = []

        for i, frame in enumerate(subtraj):
            pdb = tempdir / f"trj{i}.pdb"
            frame.save(pdb)
            assert pdb.exists()
            pdbs.append(pdb)

        if cg:
            pdbs = run_faspr(tempdir, pdbs)

        preds = run_sparta(tempdir, pdbs)
        df = build_df(preds)

    return df
