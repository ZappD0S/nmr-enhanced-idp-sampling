from pathlib import Path
import subprocess

import mdtraj as mdt
import pandas as pd
from gmx_utils import build_gmx_env, xvg_find_first_line


def gmx_gyrate(xtc, top, gmxrc=None):
    env = None

    if gmxrc is not None:
        env = build_gmx_env(gmxrc)

    output = Path("/tmp/output.xvg")

    proc = subprocess.Popen(
        ["gmx", "gyrate", "-f", xtc, "-s", top, "-o", str(output)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
    )
    proc.communicate("Protein\n")

    with output.open() as f:
        lines_to_skip = xvg_find_first_line(f)

    # print(lines_to_skip)

    df = pd.read_csv(
        output,
        sep=" ",
        skiprows=lines_to_skip,
        skipinitialspace=True,
        names=["t", "Rg", "Rg_x", "Rg_y", "Rg_z"],
        index_col=0,
    )
    output.unlink()
    assert not output.exists()

    return df


def mdtraj_rg(xtc, top):
    traj = mdt.load(xtc, top=top)
    return mdt.compute_rg(traj)
