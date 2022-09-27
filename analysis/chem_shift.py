import numpy as np
import subprocess
import pandas as pd
import mdtraj as mdt
import tempfile
from pathlib import Path
from MDAnalysis.lib.util import convert_aa_code
import concurrent.futures

random_coil_ca_cs = {
    "A": 52.84,
    "R": 56.42,
    "N": 53.23,
    "D": 54.18,
    "C": 57.53,
    "Q": 56.12,
    "E": 56.87,
    "G": 45.51,
    "H": 55.86,
    "I": 61.03,
    "L": 54.92,
    "K": 56.59,
    "M": 55.67,
    "F": 57.98,
    "P": 63.47,
    "S": 58.38,
    "T": 61.64,
    "W": 57.78,
    "Y": 57.97,
    "V": 62.06,
}

random_coil_ca_pp_cs = {
    "A": 50.5,
    "R": 54.0,
    "N": 51.3,
    "D": 52.2,
    "C": 56.4,
    "Q": 53.7,
    "E": 54.2,
    "G": 44.5,
    "H": 53.3,
    "I": 58.7,
    "L": 53.1,
    "K": 54.2,
    "M": 53.3,
    "F": 55.6,
    "P": 61.5,
    "S": 56.4,
    "T": 59.8,
    "W": 55.7,
    "Y": 55.8,
    "V": 59.8,
}


def compute_secondary_cs(residues, idx2cs):
    sec_cs = {}

    for idx, cs in idx2cs.items():
        try:
            next_is_pro = residues[idx + 1].name == "PRO"
        except IndexError:
            next_is_pro = False

        one_letter = convert_aa_code(residues[idx].name)

        if next_is_pro:
            rc_cs = random_coil_ca_pp_cs[one_letter]
        else:
            rc_cs = random_coil_ca_cs[one_letter]

        sec_cs[idx] = cs - rc_cs

    return sec_cs


def get_lines_to_skip(filename):
    """Determine the number of comment lines in a SPARTA+ output file."""
    format_string = """FORMAT %4d %4s %4s %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f"""
    # format_string = r"FORMAT %4d %4s %4s %9.3f %9.3f %9.3f %9.3f %9.3f %s"
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
spartaplus_bin = Path("/home/gzappavigna/SPARTA+/bin/SPARTA+.static.linux9")
# sparta_bin = Path("/home/gzappavigna/SPARTA/src/SPARTA")


def run_faspr(tempdir, pdbs):
    pdbs = []

    for pdb_cg in enumerate(pdbs):
        pdb_all = pdb_cg.parent / (pdb_cg.stem + "_all.pdb")

        subprocess.run([faspr_bin, "-i", str(pdb_cg), "-o", str(pdb_all)], check=True)
        assert pdb_all.exists()

        pdbs.append(pdb_all)

    return pdbs


def call_sparta(tempdir, idx, frame):
    stem = f"trj{idx}"
    pdb = tempdir / (stem + ".pdb")
    frame.save(pdb)
    assert pdb.exists()

    pred_tab = tempdir / (stem + "_pred.tab")

    subprocess.run(
        [spartaplus_bin]
        + ["-in", str(pdb)]
        # + ["-sum", str(pred_tab)]
        + ["-out", str(pred_tab)]
        + ["-spartaDir", str(spartaplus_bin.parents[1])],
        cwd=tempdir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    assert pred_tab.exists()

    # pred_in_tab = tempdir / "pred" / (stem + "_in.tab")
    # assert pred_in_tab.exists()
    # pred_in_tab.unlink()
    pdb.unlink()

    return pred_tab


def run_sparta(tempdir, traj):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []

        for i, frame in enumerate(traj):
            future = executor.submit(call_sparta, tempdir, i, frame)
            futures.append(future)

        dones, not_dones = concurrent.futures.wait(futures)

    assert len(not_dones) == 0
    preds = []

    for future in dones:
        preds.append(future.result())

    # for pdb in pdbs:
    #     pred_tab = pdb.parent / (pdb.stem + "_pred.tab")

    #     popen = subprocess.Popen(
    #         [sparta_bin]
    #         + ["-in", str(pdb)]
    #         + ["-sum", str(pred_tab)]
    #         + ["-spartaDir", str(sparta_bin.parents[1])],
    #         cwd=tempdir,
    #         stdout=subprocess.DEVNULL,
    #     )

    #     data.append((popen, pred_tab))

    # for popen, pred_tab in data:
    #     assert popen.wait() == 0
    #     assert pred_tab.exists()
    #     preds.append(pred_tab)

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


def compute_chem_shift(
    traj,
    period: float,  # in ps
    cg=False,
):

    one_every = round(period / traj.timestep)
    inds = np.arange(0, traj.time.size, one_every)

    subtraj = traj[inds]

    with tempfile.TemporaryDirectory() as tempdir_:
        tempdir = Path(tempdir_)

        # if cg:
        #     pdbs = run_faspr(tempdir, pdbs)

        preds = run_sparta(tempdir, subtraj)
        df = build_df(preds)

    return df
