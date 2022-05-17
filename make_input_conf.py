import argparse
import io
import re
import pathlib
import subprocess
from itertools import combinations
from math import floor
from random import randint

import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.util import convert_aa_code
from MDAnalysis.analysis.dihedrals import Ramachandran
from scipy import ndimage, stats

# from astropy.convolution import convolve
# from astropy.convolution.kernels import Gaussian2DKernel

from make_data_conf import write_data_config, Topology

# TODO: move the data dict either to a diffent file or to a json file

# lambda values in nm (note: internal units are A, multiply by 10)
# original values
# lambda_dict = {'A':0.730, 'R':0.000, 'N':0.432, 'D':0.378, 'C':0.595,
#                 'Q':0.514, 'E':0.459, 'G':0.649, 'H':0.514, 'I':0.973,
#                         'L':0.973, 'K':0.514, 'M':0.838, 'F':1.000, 'P':1.000,
#                                 'S':0.595, 'T':0.676, 'W':0.946, 'Y':0.865, 'V':0.892}
# values from bob best
lambda_dict = {
    "A": 0.51507,
    "R": 0.24025,
    "N": 0.78447,
    "D": 0.30525,
    "C": 0.46169,
    "Q": 0.29516,
    "E": 0.42621,
    "G": 1.24153,
    "H": 0.55537,
    "I": 0.83907,
    "L": 0.51207,
    "K": 0.47106,
    "M": 0.64648,
    "F": 1.17854,
    "P": 0.34128,
    "S": 0.11195,
    "T": 0.27538,
    "W": 0.97588,
    "Y": 1.04266,
    "V": 0.55645,
}

# side-chain r0 from M. Levitt J Mol Biol 1976 (derived from Clothia 1975)
# values in A
r0_dict = {
    "A": 4.6,
    "R": 6.8,
    "N": 5.7,
    "D": 5.6,
    "C": 5.0,
    "Q": 6.1,
    "E": 6.1,
    "H": 6.2,
    "I": 6.2,
    "G": 3.8,
    "L": 6.3,
    "K": 6.3,
    "M": 6.2,
    "F": 6.8,
    "P": 5.6,
    "S": 4.8,
    "T": 5.6,
    "W": 7.2,
    "Y": 6.9,
    "V": 5.8,
}

# side-chain epsilon from M. Levitt J Mol Biol 1976 (derived from number of heavy atoms)
# values in kcal/mol
epsilon_dict = {
    "A": 0.05,
    "R": 0.39,
    "N": 0.21,
    "D": 0.21,
    "C": 0.10,
    "Q": 0.27,
    "E": 0.27,
    "G": 0.025,
    "H": 0.33,
    "I": 0.21,
    "L": 0.21,
    "K": 0.27,
    "M": 0.21,
    "F": 0.39,
    "P": 0.16,
    "S": 0.10,
    "T": 0.16,
    "W": 0.56,
    "Y": 0.45,
    "V": 0.16,
}


# define non-bonded forces
def lj(r, epsilon, sigma):
    V = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
    F = 4 * epsilon / r * (12 * (sigma / r) ** 12 - 6 * (sigma / r) ** 6)

    return (V, F)


def lj86(r, epsilon, sigma):
    V = epsilon * (3 * (sigma / r) ** 8 - 4 * (sigma / r) ** 6)
    F = 24 * epsilon / r * ((sigma / r) ** 8 - (sigma / r) ** 6)

    return (V, F)


def phinb(r, eps, sigma, lmbd):
    VLJ, FLJ = lj(r, eps, sigma)

    if r < 2 ** (1 / 6) * sigma:
        V = VLJ + (1 - lmbd) * eps
        F = FLJ
    else:
        V = lmbd * VLJ
        F = lmbd * FLJ

    return (V, F)


def rep8(r, epsilon, sigma):
    V = epsilon * (3 * (sigma / r) ** 8)
    F = 24 * epsilon / r * ((sigma / r) ** 8)

    return (V, F)


def write_cmap_tables(stream: io.TextIOBase, cmaps_dict, bins):
    for res, (index, cmap) in cmaps_dict.items():
        stream.write(f"# residue {res.resname}{res.resid}, type {index}\n\n")

        for angle, row in zip(bins, cmap):
            stream.write(f"# {angle}\n\n")
            stream.write(" ".join(map(str, row)) + " \n\n")

        stream.write("\n")


def write_ashbaugh_hatch_tables(stream: io.TextIOBase, ashbaugh_hatch_tables):
    # TODO: comment?
    stream.write("# comment\n\n")

    for (entry, _, _), table in ashbaugh_hatch_tables.items():
        stream.write(
            f"{entry}\n"  #
            f"N {table.size} R {table["r"][0]} {table["r"][-1]}\n"
            f"\n"
        )

        for row in table:
            stream.write(" ".join(map(str, row)) + "\n")

        stream.write("\n\n")


def write_input_config(
    stream: io.TextIOBase,
    topo: Topology,
    args: argparse.Namespace,
    ah_tables,
    ah_table_fname: str,
    data_conf_fname: str,
    output_traj_fname: str,
    cmap_fname: str | None = None,
):
    # TODO: where does this value come from?
    kappa = 3.04 / np.sqrt(args.concentration)
    box_name = "mybox"
    cmap_name = "mycmap"
    Nsteps = floor(args.sim_time * 1e6 / args.time_step)
    bhw = args.box_half_width
    box_shape = " ".join(map(str, [-bhw, bhw] * 3))

    stream.write(
        f"# comment\n"  #
        f"\n"
        f"units real\n"
        f"atom_style full\n"
        f"package gpu 0 neigh no\n"
        f"region {box_name} block {box_shape}\n"
        f"create_box {topo.n_atom_types} {box_name}"
        f" bond/types {topo.n_bond_types}"
        f" angle/types {topo.n_angle_types}"
        f" dihedral/types {topo.n_dihedral_types}"
        # TODO: what's the point of the extra stuff per atom?
        f" extra/bond/per/atom 3"
        f" extra/angle/per/atom 3"
        f" extra/dihedral/per/atom 2\n"
        f"\n"
        f"special_bonds charmm\n"
        f"pair_style hybrid/overlay"
        f" table linear {args.ah_points}"
        f" coul/debye {1.0 / kappa} {args.CO_charges}\n"
    )

    stream.write("\n")

    for entry, index1, index2 in ah_tables:
        stream.write(
            f"pair_coeff {index1} {index2}"  #
            f" table {ah_table_fname} {entry} {args.ah_cutoff}\n"
        )

    stream.write("\n")

    stream.write(
        f"pair_coeff * * coul/debye\n"
        f"dielectric {args.dielectric}\n"
        f"bond_style harmonic\n"
        f"angle_style harmonic\n"
        f"dihedral_style fourier\n"
    )

    if cmap_fname is not None:
        stream.write(
            f"fix {cmap_name} all cmap {cmap_fname}\n"  #
            f"fix_modify {cmap_name} energy yes\n"
        )

    stream.write(
        f"read_data {data_conf_fname} add append"
        + (f" fix {cmap_name} crossterm CMAP" if cmap_fname is not None else "")
        + "\n"
    )

    stream.write("\n")

    stream.write(
        f"neighbor 2.0 bin\n"
        f"neigh_modify delay 5\n"
        f"\n"
        f"timestep {args.time_step}\n"
        f"thermo_style multi\n"
        f"thermo 50\n"
        f"\n"
        f"minimize 1.0e-4 1.0e-6 10000 100000\n"
        f"fix 1 all nve\n"
        # TODO: what is the random number?
        f"fix 2 all langevin {args.temp} {args.temp} {args.langevin_damp} {randint(1, 100000)}\n"
        f"\n"
        f"dump 1 all xtc 250 {output_traj_fname}\n"
        f"run {Nsteps}\n"
    )


def write_configs(
    subfolder: pathlib.Path,
    init_ag,
    topo,
    args: argparse.Namespace,
    u_ens_traj: mda.Universe | None = None,
):
    # for now we leave these hardcoded...
    name = "CG"
    output_traj_fname = "traj.xtc"

    main_config_file = subfolder / f"in.{name}"
    data_conf_file = subfolder / f"data.{name}"
    cmap_file = subfolder / f"{name}.cmap" if args.use_cmap else None
    ah_table_file = subfolder / "Ashbaugh-Hatch.table"

    ah_tables = build_ashbaugh_hatch_tables(topo.atom_type_to_index, args)

    with main_config_file.open("w") as f:
        write_input_config(
            f,
            topo,
            args,
            ah_tables,
            ah_table_file.name,
            data_conf_file.name,
            output_traj_fname,
            cmap_file.name if cmap_file is not None else None,
        )

    with open(data_conf_file, "w") as f:
        write_data_config(f, topo, init_ag, name, args.box_half_width, args.use_cmap)

    with ah_table_file.open("w") as f:
        write_ashbaugh_hatch_tables(f, ah_tables)

    if cmap_file is None:
        return

    if u_ens_traj is None:
        # TODO: add messagge..
        raise Exception

    cmaps_dict, bins = build_cmaps(u_ens_traj, topo.crossterm_type_to_index, args)

    with open(cmap_file, "w") as f:
        write_cmap_tables(f, cmaps_dict, bins)


def build_ashbaugh_hatch_tables(atom_type_to_index, args: argparse.Namespace):

    def compute_params(key):
        epsilon = args.epsilon

        match key[:2]:
            case (resname, "CA"):
                r = 3.32  # CT1+HB1 in C36
                e = epsilon
                l = 0.0
                name = resname + "_CA"
            case (resname, "CB" | "H"):
                one_letter_resname = convert_aa_code(resname)

                r = r0_dict[one_letter_resname]
                e = epsilon_dict[one_letter_resname]
                l = lambda_dict[one_letter_resname]
                name = resname + "_CB"
            case (None, "C"):
                r = 3.7  # C+O in C36
                e = epsilon
                l = 0.0
                name = "gen_C"
            case (None, "N"):
                r = 2.0745  # NH1+H in C36
                e = epsilon
                l = 0.0
                name = "gen_N"
            case ("PRO", "N"):
                r = 2.0745  # NH1+H in C36
                e = epsilon
                l = 0.0
                name = "gen_NPRO"
            case _:
                raise Exception

        return r, e, l, name

    tables = {}
    rs = np.linspace(args.ah_min_dist, args.ah_cutoff, args.ah_points)

    for key1, key2 in combinations(atom_type_to_index, 2):
        _, atom_type1, _ = key1
        _, atom_type2, _ = key2

        r1, e1, l1, name1 = compute_params(key1)
        r2, e2, l2, name2 = compute_params(key1)

        entry = name1 + "_" + name2

        rij = 0.5 * (r1 + r2) * args.scaling  # np.sqrt(r1*r2)
        eij = 0.5 * (e1 + e2)  # np.sqrt(e1*e2)
        # lij = 0.5 * (l1 + l2)

        # if any of the two atoms is a CA or a CB
        if {atom_type1, atom_type2} & {"CA", "CB"}:
            Es, Fs = rep8(rs, eij, rij)
        else:
            # Es, Fs = phinb(rs, eij, rij, lij)
            Es, Fs = lj86(rs, eij, rij)

        arrs_dict = {
            "index": np.arange(rs.size) + 1,
            "r": rs,
            "E": Es,
            "F": Fs,
        }

        dtype = np.dtype([(name, arr.dtype) for name, arr in arrs_dict.items()])
        table = np.empty(rs.size, dtype=dtype)

        for name, arr in arrs_dict.items():
            table[name] = arr

        index1 = atom_type_to_index[key1]
        index2 = atom_type_to_index[key2]
        tables[(entry, index1, index2)] = table

    return tables


def build_cmaps(u_ens_traj, crossterm_type_to_index, args: argparse.Namespace):
    RT = 1.98720425864083e-3 * args.temp
    # the last element of bins is just 180,
    # it's only needed for np.histogram2d to work correctly
    bins = np.linspace(-180, 180, args.cmap_points + 1)

    # TODO: is it phi-psi? it should be...
    target_rama = np.loadtxt(args.target_dist)

    r = Ramachandran(u_ens_traj).run()

    # r.results.angles is (n_frames, n_residues - 2, 2)
    angles = np.transpose(r.results.angles, [1, 2, 0])
    # after this it's (n_residues - 2, 2, n_frames)

    cmaps_dict = {}

    # TODO: instead of using histogram2d and smoothing with a gaussian filter
    # let's experiment with KDE

    for res, phi_psi_ens in zip(u_ens_traj.residues[1:-1], angles):
        ens_rama, _, _ = np.histogram2d(*phi_psi_ens, bins=bins, density=True)
        ens_rama[ens_rama == 0.0] = 1e-5

        # TODO: does this and astropy's convolve() produce the same output?
        ens_rama = ndimage.gaussian_filter(ens_rama, sigma=0.3, mode="wrap")
        # ens_rama = convolve(
        #     ens_rama,
        #     Gaussian2DKernel(x_stddev=0.3, y_stddev=0.3),
        #     # boundary="extend",
        #     boundary="wrap",
        # )

        cmap = -RT * np.log(ens_rama / target_rama)
        index = crossterm_type_to_index[res.resid]
        cmaps_dict[res] = (index, cmap)

    return cmaps_dict, bins[:-1]


def clean_pdb(input: io.TextIOBase, output: io.TextIOBase):
    pattern = re.compile(
        r"""^ATOM
            \s+
            \d+                         # atom index
            \s+
            \d?[A-Z]{1,2}\d?            # atom name
            \s+
            [A-Z]{3}                    # residue
            \s+
            \d+                         # residue id
            \s+
            (?:-?\d{1,3}\.\d{3}\s*){2}  # position
            -?\d{1,3}\.\d{3}            # position
            \s+
            \d\.\d{2}                   # other..
            \s+
            \d\.\d{2}                   # other..
            """,
        re.X,
    )

    for line in input:
        if (match := pattern.search(line)) is None:
            print(line)
            raise Exception

        output.write(match[0] + "\n")


def main():
    parser = argparse.ArgumentParser(description="What the program does")

    parser.add_argument("--topo-pdb", type=str)

    cmap_group = parser.add_argument_group("CMAP")
    cmap_group.add_argument("--use-cmap", action="store_true")
    cmap_group.add_argument("--target-dist", type=argparse.FileType("r"))
    cmap_group.add_argument(
        "--ens-traj",
        type=str,
        help="""this is not an actual trajectory.
        It's just the ASTEROIDS ensemble conformations put together in an XTC file""",
    )
    cmap_group.add_argument("--cmap-points", nargs="?", type=int, default=24)

    parser.add_argument("init_confs", nargs="+", type=pathlib.Path)
    parser.add_argument(
        "-o", "--output-dir", nargs="?", default=pathlib.Path.cwd(), type=pathlib.Path
    )

    parser.add_argument("--lmp-path", nargs="?", help="LAMMPS binary path")
    parser.add_argument(
        "--n-tasks", nargs="?", type=int, default=6, help="number of parallel tasks"
    )

    ah_group = parser.add_argument_group("Ashbaugh-Hatch")
    ah_group.add_argument("--ah-min-dist", nargs="?", type=float, default=0.5)
    ah_group.add_argument("--ah-cutoff", nargs="?", type=float, default=30.0)
    ah_group.add_argument(
        "--epsilon",
        nargs="?",
        type=float,
        default=0.2,
        help="energy scale of non-bonded interactions, in Kcal/mol.",
    )
    ah_group.add_argument("--ah-points", nargs="?", type=int, default=7501)

    phys_params_group = parser.add_argument_group("Physical parameters")
    phys_params_group.add_argument(
        "-ld", "--langevin-damp", nargs="?", type=float, default=5
    )
    phys_params_group.add_argument(
        "-conc", "--concentration", nargs="?", type=float, default=0.15
    )
    phys_params_group.add_argument("--CO-charges", nargs="?", type=float, default=30.0)
    phys_params_group.add_argument("--dielectric", nargs="?", type=float, default=78.5)
    phys_params_group.add_argument("-T", "--temp", nargs="?", type=float, default=298.0)
    phys_params_group.add_argument("--scaling", nargs="?", type=float, default=0.66)

    sim_params_group = parser.add_argument_group("Simulation parameters")
    sim_params_group.add_argument(
        "--sim-time", nargs="?", type=float, default=500.0, help="in nanoseconds"
    )
    sim_params_group.add_argument(
        "-ts", "--time-step", nargs="?", type=float, default=4.0, help="in femtoseconds"
    )
    sim_params_group.add_argument(
        "--box-half-width",
        nargs="?",
        type=int,
        default=200,
        help="in nanometers (check..)",
    )

    args = parser.parse_args()

    if args.use_cmap and not (args.target_dist and args.ens_traj):
        parser.error("TODO")

    args.output_dir.mkdir(exist_ok=True)
    topo_json_file = args.output_dir / "topo.json"
    cg_pdb = args.output_dir / "cg.pdb"

    if args.topo_pdb is not None:
        u_topo = mda.Universe(args.topo_pdb, format="PDB", guess_bonds=True)
        topo = Topology.from_pdb(u_topo)

        ref_ag = u_topo.atoms[list(topo.atom_to_type)]
        cg_pdb.touch(exist_ok=True)
        ref_ag.write(cg_pdb)

        with topo_json_file.open("w") as f:
            topo.to_json(f)
    else:
        try:
            with topo_json_file.open() as f:
                topo = Topology.read_json(f)
        except FileNotFoundError:
            # TODO add message..
            raise Exception

    if args.use_cmap:
        output_dir = args.output_dir / "cmap"
        u_ens_traj = mda.Universe(cg_pdb, args.ens_traj)
    else:
        output_dir = args.output_dir / "ref"
        u_ens_traj = None

    output_dir.mkdir(exist_ok=True)

    for init_conf in args.init_confs:
        cleaned_pdb = io.StringIO()

        with init_conf.open() as raw_pdb:
            clean_pdb(raw_pdb, cleaned_pdb)

        cleaned_pdb.seek(0)
        u_init = mda.Universe(cleaned_pdb, format="PDB")

        # select the atoms used in the simulation
        init_ag = u_init.atoms[list(topo.atom_to_type)]

        subfolder = output_dir / init_conf.stem
        subfolder.mkdir(exist_ok=True)

        write_configs(subfolder, init_ag, topo, args, u_ens_traj)

        subprocess.run(
            [
                "mpirun",
                "-n",
                str(args.n_tasks),
                (args.lmp_path if args.lmp_path is not None else "lmp"),
                "-sf",
                "gpu",
                "-in",
                "in.CG",
            ],
            stdout=subprocess.DEVNULL,
            cwd=subfolder,
        )


if __name__ == "__main__":
    main()
