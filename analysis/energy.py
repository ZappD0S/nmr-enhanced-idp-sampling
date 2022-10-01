from pathlib import Path
import subprocess

import pandas as pd
from parse import compile
from gmx_utils import build_gmx_env, xvg_find_first_line


def parse_colnames(f):
    fmt = '@ s{:d} legend "{}"'
    pattern = compile(fmt)

    names = {}

    for line in f:
        if (match := pattern.parse(line.strip())) is not None:
            index, name = match
            names[index] = name

    return [names[i] for i in sorted(names)]


def gmx_energy(
    edr,
    tpr,
    params=[
        "Bond",
        "Angle",
        "Proper-Dih.",
        "Per.-Imp.-Dih.",
        "LJ",
        "Coulomb",
        "Pot",
        "Kin",
        "Tot",
        "Cons",
        "Temp",
        "Pressure",
    ],
    gmxrc=None,
):
    env = None
    if gmxrc is not None:
        env = build_gmx_env(gmxrc)

    output = Path("/tmp/output.xvg")

    proc = subprocess.Popen(
        ["gmx", "energy", "-f", edr, "-s", tpr, "-o", str(output)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
    )
    proc.communicate("\n".join(params) + "\n\n")

    with output.open() as f:
        lines = f.readlines()

    names = parse_colnames(lines)
    lines_to_skip = xvg_find_first_line(lines)

    print(lines_to_skip)
    print(names)
    df = pd.read_csv(
        output,
        sep=" ",
        skiprows=lines_to_skip,
        skipinitialspace=True,
        names=["t"] + names,
        index_col=0,
    )
    output.unlink()
    assert not output.exists()

    return df
