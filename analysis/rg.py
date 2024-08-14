# use gromacs "gmx gyrate" or mdtraj.compute_rg? well, the faster one..
from pathlib import Path
import shutil
import os
import shlex
import subprocess

import mdtraj as mdt
import pandas as pd
from parse import compile


def xvg_find_first_line(f):
    fmt = " {:10.3f} {:8.6f} {:8.6f} {:8.6f} {:8.6f}\n"
    pattern = compile(fmt)

    for i, line in enumerate(f):
        if pattern.parse(line) is not None:
            return i

    raise Exception


def gmx_gyrate(xtc, pdb_or_gro):
    gmxrc = "/usr/local/gromacs/bin/GMXRC"

    command = shlex.split(f"bash -c 'source {gmxrc} && env'")
    p = subprocess.run(command, text=True, capture_output=True)

    gmx_env = os.environ.copy()

    for line in p.stdout.splitlines():
        (key, _, value) = line.partition("=")
        gmx_env[key] = value

    output = Path("/tmp/output.xvg")

    proc = subprocess.Popen(
        ["gmx", "gyrate", "-f", xtc, "-s", pdb_or_gro, "-o", str(output)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        text=True,
        env=gmx_env,
    )
    proc.communicate("Protein\n")

    with output.open() as f:
        lines_to_skip = xvg_find_first_line(f)

    print(lines_to_skip)
    df = pd.read_csv(
        output,
        sep=" ",
        skiprows=lines_to_skip,
        skipinitialspace=True,
        names=["t", "Rg", "Rg_x", "Rg_y", "Rg_z"],
        index_col=0,
    )
    output.unlink()

    return df
