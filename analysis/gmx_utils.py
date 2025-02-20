import os
import shlex
import subprocess


def xvg_find_first_line(f):
    for i, line in enumerate(f):
        stripped = line.lstrip()
        if not (stripped.startswith("#") or stripped.startswith("@")):
            return i

    raise Exception


def build_gmx_env(gmxrc):
    env = os.environ.copy()
    command = shlex.split(f"bash -c 'source {gmxrc} && env'")
    p = subprocess.run(command, text=True, capture_output=True)

    for line in p.stdout.splitlines():
        (key, _, value) = line.partition("=")
        env[key] = value
