from pathlib import Path
import os
import shlex
import subprocess

import pandas as pd
from parse import compile


def xvg_find_first_line(f):
    for i, line in enumerate(f):
        stripped = line.lstrip()
        if not (stripped.startswith("#") or stripped.startswith("@")):
            return i

    raise Exception


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
):
    gmxrc = "/usr/local/gromacs/bin/GMXRC"

    command = shlex.split(f"bash -c 'source {gmxrc} && env'")
    p = subprocess.run(command, text=True, capture_output=True)

    gmx_env = os.environ.copy()

    for line in p.stdout.splitlines():
        (key, _, value) = line.partition("=")
        gmx_env[key] = value

    output = Path("/tmp/output.xvg")

    proc = subprocess.Popen(
        ["gmx", "energy", "-f", edr, "-s", tpr, "-o", str(output)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        env=gmx_env,
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
