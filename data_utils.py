from itertools import pairwise

charges_dict = {
    "A": 0,
    "R": 1,
    "N": 0,
    "D": -1,
    "C": 0,
    "Q": 0,
    "E": -1,
    "G": 0,
    "H": 0.5,
    "I": 0,
    "L": 0,
    "K": 1,
    "M": 0,
    "F": 0,
    "P": 0,
    "S": 0,
    "T": 0,
    "W": 0,
    "Y": 0,
    "V": 0,
}

residue_masses_dict = {
    "A": 15.0347,
    "R": 100.1431,
    "N": 58.0597,
    "D": 59.0445,
    "C": 47.0947,
    "Q": 72.0865,
    "E": 73.0713,
    "G": 1.0079,
    "H": 81.0969,
    "I": 57.1151,
    "L": 57.1151,
    "K": 72.1297,
    "M": 75.1483,
    "F": 91.1323,
    "P": 41.0725,
    "S": 31.0341,
    "T": 45.0609,
    "W": 130.1689,
    "Y": 107.1317,
    "V": 43.0883,
}

# side-chain r0 from M. Levitt J Mol Biol 1976 (derived from Clothia 1975)
# values in angstrom
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


def get_atom_id(atom):
    return (int(atom.resid), atom.name)


def filter_atoms(ag, atom_ids):
    [seg] = ag.segments
    segid = seg.segid
    return sum(
        ag.select_atoms("atom " + " ".join([segid] + [str(x) for x in atom_id]))
        for atom_id in atom_ids
    )


def is_terminus(atom):
    # TODO: do we need the int() ?
    def is_c_terminus(atom):
        u = atom.universe
        return (
            atom.type == "C" and atom.name == "C" and int(atom.resid) == len(u.residues)
        )

    def is_n_terminus(atom):
        return atom.type == "N" and atom.name == "N" and int(atom.resid) == 1

    return is_c_terminus(atom) or is_n_terminus(atom)


def build_cmap_atoms(dihedrals):
    atoms_list = []

    left_dihedrals = [
        dihedral
        for dihedral in dihedrals
        # consider only dihedrals that involve backbone atoms
        if all(atom.name in {"CA", "C", "N"} for atom in dihedral)
    ]

    not_matched = []

    while True:
        dihedral1 = next(
            (
                dihedral
                for dihedral in left_dihedrals
                # C-N-CA-C
                if dihedral[2].name == "CA"
            ),
            None,
        )

        if dihedral1 is None:
            break

        left_dihedrals.remove(dihedral1)
        found = False

        for dihedral2 in left_dihedrals:
            if dihedral1[1:] != dihedral2[:-1]:
                continue

            assert all(
                atom1.residue == atom2.residue
                for atom1, atom2 in pairwise(dihedral1[1:])
            )

            found = True
            atoms = dihedral1[:] + dihedral2[-1:]
            assert len(atoms) == 5
            atoms_list.append(atoms)

            left_dihedrals.remove(dihedral2)
            break

        if not found:
            not_matched.append(dihedral1)

    not_matched.extend(left_dihedrals)
    # print([[atom.name for atom in dih] for dih in not_matched])
    # print(len(atoms_list))
    # assert len(not_matched) == 2

    return atoms_list
