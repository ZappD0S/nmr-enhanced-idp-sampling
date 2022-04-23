import argparse
from itertools import combinations
import numpy as np
from math import floor
import MDAnalysis as mda
from MDAnalysis.lib.util import convert_aa_code

from MDAnalysis.analysis.dihedrals import Ramachandran

# from astropy.convolution import convolve
# from astropy.convolution.kernels import Gaussian2DKernel
from random import randint
from scipy import ndimage

from make_data_conf import build_data_config

parser = argparse.ArgumentParser(description="What the program does")

# parser.add_argument(
#     "integers", metavar="N", type=int, nargs="+", help="an integer for the accumulator"
# )
# parser.add_argument(
#     "--sum",
#     dest="accumulate",
#     action="store_const",
#     const=sum,
#     default=max,
#     help="sum the integers (default: find the max)",
# )

# parser.add_argument("filename")  # positional argument
# parser.add_argument("-c", "--count")  # option that takes a value
# parser.add_argument("-v", "--verbose", action="store_true")  # on/off flag

parser.add_argument("-ld", "--langevin-damp", type=float, default=5)
parser.add_argument("-conc", "--concentration", type=float, default=0.15)
parser.add_argument("--CO-charges", type=float, default=30.0)
parser.add_argument("--dielectric", type=float, default=78.5)
parser.add_argument("-nbtp", "--non-bonded-table-points", type=float, default=78.5)
parser.add_argument("--non-bonded-cutoff", type=float, default=78.5)
parser.add_argument(
    "--epsilon",
    type=float,
    default=78.5,
    help="energy scale of non-bonded interactions, in Kcal/mol.",
)

parser.add_argument("-T", "--temp", type=float, default=298.0)


parser.add_argument(
    "-ts", "--time-step", type=float, default=4.0, help="in femtoseconds."
)

# args = parser.parse_args()
# print(args.filename, args.count, args.verbose)

# langdamp = float(sys.argv[1])
langdamp = 5.0
# TODO: we need to rebuild the data.CG file for every conformers that we simulate
# but only that, we don't need to rebuild the topology, the only thing that changes
# are the positions in the "Atoms" section

pdb = "CG.pdb"
conc = 0.15  # salt, M
CO_charges = 30.0  # cut-off Coulomb/Debye/GPU
diel = 78.5  # for water, dielectric constant
NP = 7501  # number of points in non-bonded tables
CO_NB = 30.0  # cut-off non-bonded interactions, A
epsilon = 0.2  # Kcal/mol, energy scale of non-bonded interactions
rminNB = 0.5  # A
rmaxNB = 30.0  # A
# langdamp = 200.0 # in fs
T = 298.0  #
trajT = 500  # ns
tstep = 4.0  # fs
XTC = "/data/vschnapka/202310-CMAP-HPS/MeV-NT/MeV_NT_ens/ensemble_200_1/ensemble_200_1.xtc"
PDB = "/data/vschnapka/202310-CMAP-HPS/MeV-NT/MeV_NT_ens/1_CG.pdb"
REFfile = "/data/vschnapka/202310-CMAP-HPS/reference_dih/all-rama-ref.out"
NCMAP = 24  # number of points to build correction maps (NCMAPxNCMAP)
scaling = 0.66  # scaling factor of ML vdW radii (this is 0.66 in FM)

RT = 1.98720425864083e-3 * T

# TODO: move the data dict either to a diffent file or to a json file

# sigma values in nm (note: internal units are A, multiply by 10)
# sigma_dict = {
#     "A": 0.504,
#     "R": 0.656,
#     "N": 0.568,
#     "D": 0.558,
#     "C": 0.548,
#     "Q": 0.602,
#     "E": 0.592,
#     "G": 0.450,
#     "H": 0.608,
#     "I": 0.618,
#     "L": 0.618,
#     "K": 0.636,
#     "M": 0.618,
#     "F": 0.636,
#     "P": 0.556,
#     "S": 0.518,
#     "T": 0.562,
#     "W": 0.678,
#     "Y": 0.646,
#     "V": 0.586,
# }

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


def build_configs(name, box_half_width):
    box_name = "mybox"

    Nsteps = int(floor(trajT * 1e6 / tstep))

    # input
    topo_ref_pdb = "CG.pdb"

    ref_dist_file = "/data/vschnapka/202310-CMAP-HPS/reference_dih/all-rama-ref.out"
    # TODO: this should be the same as topo_ref_pdb, right?
    ens_pdb_file = "/data/vschnapka/202310-CMAP-HPS/MeV-NT/MeV_NT_ens/1_CG.pdb"
    ens_traj_file = "/data/vschnapka/202310-CMAP-HPS/MeV-NT/MeV_NT_ens/ensemble_200_1/ensemble_200_1.xtc"

    output_traj_file = "traj.xtc"

    # TODO: turn this into Path objects?
    main_config_file = f"in.{name}"
    data_conf_file = f"data.{name}"
    cmap_file = f"{name}.cmap"
    ashbaugh_hatch_table_file = "Ashbaugh-Hatch.table"

    with open(data_conf_file, "w") as f:
        topo_dicts = build_data_config(name, f, topo_ref_pdb, box_half_width)

    n_atom_types = len(topo_dicts["atom_type_to_index"])
    n_bond_types = len(topo_dicts["bond_type_to_index"])
    n_angle_types = len(topo_dicts["angle_type_to_index"])
    n_dih_types = len(topo_dicts["dihedral_type_to_index"])

    kappa = 3.04 / np.sqrt(conc)

    cmap_dict, bins = build_cmap(
        ref_dist_file,
        ens_pdb_file,
        ens_traj_file,
        topo_dicts["crossterm_type_to_index"],
    )

    with open(cmap_file, "w") as f:
        for res, (index, cmap) in cmap_dict.items():
            f.write(f"# residue {res.resname}{res.resid}, type {index}\n\n")

            for angle, row in zip(bins, cmap):
                f.write(f"# {angle}\n\n")
                f.write(" ".join(map(str, row)) + " \n\n")

            f.write("\n")

    ashbaugh_hatch_tables = build_ashbaugh_hatch_tables(
        topo_dicts["atom_type_to_index"]
    )

    with open(ashbaugh_hatch_table_file, "w") as f:
        # TODO: comment
        f.write("# comment\n\n")

        for (entry, _, _), table in ashbaugh_hatch_tables.items():
            f.write(
                f"{entry}\n"  #
                f"N {NP} R {rminNB} {rmaxNB}\n"
                f"\n"
            )

            for row in table:
                f.write(" ".join(map(str, row)) + "\n")

            f.write("\n\n")

    with open(main_config_file, "w") as f:
        f.write(
            f"# comment\n"  #
            f"\n"
            f"units real\n"
            f"atom_style full\n"
            f"region {box_name} block "
            + " ".join(map(str, [-box_half_width, box_half_width] * 3))
            + "\n"
            f"create_box {n_atom_types} {box_name}"
            f" bond/types {n_bond_types}"
            f" angle/types {n_angle_types}"
            f" dihedral/types {n_dih_types}"
            f" extra/bond/per/atom 3"
            f" extra/angle/per/atom 3"
            f" extra/dihedral/per/atom 2\n"
            f"\n"
            f"special_bonds charmm\n"
            f"pair_style hybrid/overlay"
            f" table linear {NP}"
            f" coul/debye {1.0 / kappa} {CO_charges}\n"
        )

        f.write("\n")

        for entry, index1, index2 in ashbaugh_hatch_tables:
            f.write(
                f"pair_coeff {index1} {index2}"
                f" table {ashbaugh_hatch_table_file} {entry} {CO_NB}\n"
            )

        f.write("\n")

        f.write(
            f"pair_coeff * * coul/debye\n"
            f"dielectric {diel}\n"
            f"fix cmap all cmap {cmap_file}\n"
            f"fix_modify cmap energy yes\n"
            f"bond_style harmonic\n"
            f"angle_style harmonic\n"
            f"dihedral_style fourier\n"
            f"read_data {data_conf_file} add append fix cmap crossterm CMAP\n"
            f"\n"
            f"neighbor 2.0 bin\n"
            f"neigh_modify delay 5\n"
            f"\n"
            f"timestep {tstep}\n"
            f"thermo_style multi\n"
            f"thermo 50\n"
            f"\n"
            f"minimize 1.0e-4 1.0e-6 10000 100000\n"
            f"fix 1 all nve\n"
            # TODO: what is the random number?
            f"fix 2 all langevin {T} {T} {langdamp} {randint(1, 100000)}\n"
            f"\n"
            f"dump 1 all xtc 250 {output_traj_file}\n"
            f"run {Nsteps}\n"
        )


def build_ashbaugh_hatch_tables(atom_type_to_index):

    def compute_params(key):
        match key:
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

        return (r, e, l, name)

    tables = {}
    rs = np.linspace(rminNB, rmaxNB, NP)

    for key1, key2 in combinations(atom_type_to_index, 2):
        _, atom_type1 = key1
        _, atom_type2 = key2

        (r1, e1, l1, name1) = compute_params(key1)
        (r2, e2, l2, name2) = compute_params(key1)

        entry = name1 + "_" + name2

        rij = 0.5 * (r1 + r2) * scaling  # np.sqrt(r1*r2)
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


def build_cmap(ref_dist_file, ens_pdb_file, ens_traj_file, crossterm_type_to_index):
    # the last element of bins is just 180,
    # it's only needed for np.histogram2d to work correctly
    bins = np.linspace(-180, 180, NCMAP + 1)

    # TODO: is it phi-psi? it should be...
    ref_rama = np.loadtxt(ref_dist_file)

    u = mda.Universe(ens_pdb_file, ens_traj_file)
    r = Ramachandran(u).run()

    # r.results.angles is (n_frames, n_residues - 2, 2)
    angles = np.transpose(r.results.angles, [1, 2, 0])
    # after this it's (n_residues - 2, 2, n_frames)

    cmaps_dict = {}

    for res, (phi_ens, psi_ens) in zip(u.residues[1:-1], angles):
        ens_rama, _, _ = np.histogram2d(phi_ens, psi_ens, bins=bins, density=True)
        ens_rama[ens_rama == 0.0] = 1e-5

        # TODO: does this and astropy's convolve() produce the same output?
        ens_rama = ndimage.gaussian_filter(ens_rama, sigma=0.3, mode="wrap")
        # ens_rama = convolve(
        #     ens_rama,
        #     Gaussian2DKernel(x_stddev=0.3, y_stddev=0.3),
        #     boundary="extend",
        # )

        cmap = -RT * np.log(ens_rama / ref_rama)

        index = crossterm_type_to_index[res.resid]

        cmaps_dict[res] = (index, cmap)

    return cmaps_dict, bins[:-1]
