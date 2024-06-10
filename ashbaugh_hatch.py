import argparse
import io
from itertools import combinations

import numpy as np
from MDAnalysis.lib.util import convert_aa_code


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

        # TODO: why is the potential only repulsive when one of the atom is a CA or CB?

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
