import io
from more_itertools import pairwise, triplewise
import MDAnalysis as mda
from MDAnalysis.topology.tables import masses as atom_masses_dict
from MDAnalysis.lib.util import convert_aa_code
from MDAnalysis.core.topologyobjects import Bond, Angle

kspring = 200  # in E/distance^2 units

# charges
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

# masses
# these are the masses of side chains
residue_masses_dict = {
    "A": 15.0347,
    "R": 100.1431,
    "N": 58.0597,
    "D": 59.0445,
    "C": 47.0947,
    "Q": 72.0865,
    "E": 73.0713,
    "G": 1.0079 + 12.011 / 2.0,  #'G':1.0079,
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


# for each C_alpha there are three dihedral angles (possible torsions of the backbone).
# For the first and last C_alpha there would be an additional two additional angles
# (a phi and a psi, for N- and C- terminals) but we ingnore them
#
#    ┌─psi
#    │
#    │   ┌─omega
#    │   │
# Cb │   │ H ┌─phi
# │  │   │ │ │
# │  ▼   ▼ │ ▼
# Ca───C───N───Ca
#      │       │
#      │       │
#      O       Cb


def write_list(stream, header, data):
    stream.write(header + "\n")
    stream.write("\n")

    for row in data:
        stream.write(" ".join(map(str, row)) + "\n")

    stream.write("\n")


def build_data_config(name, stream: io.TextIOBase, topo_ref_pdb, box_half_width):
    u = mda.Universe(topo_ref_pdb, guess_bonds=True)

    atom_to_type, atom_type_to_index = build_atom_to_type_dict(u)
    atom_type_masses = build_atom_type_masses(atom_type_to_index)
    atoms = build_atoms_list(u, atom_to_type, atom_type_to_index)

    bond_to_type, bond_type_to_index = build_bond_to_type_dict(u)
    bonds = build_bonds_list(bond_to_type, bond_type_to_index)
    bond_coeffs = build_bond_coeffs(bond_type_to_index)

    angle_to_type, angle_type_to_index = build_angle_to_type_dict(u, bond_to_type)
    angles = build_angles_list(angle_to_type, angle_type_to_index)
    angle_coeffs = build_angle_coeffs(angle_type_to_index)

    dihedral_to_type, dihedral_type_to_index = build_dihedral_to_type_dict(
        u, angle_to_type
    )
    dihedral_coeffs = build_dihedral_coeffs(dihedral_type_to_index)
    dihedrals = build_dihedrals_list(dihedral_to_type, dihedral_type_to_index)

    crossterm_to_type, crossterm_type_to_index = build_cmap_crossterms_to_type_dict(u)
    cmap_5atoms = build_cmap_5atoms_list(crossterm_to_type, crossterm_type_to_index)

    stream.write(
        f"LAMMPS {name} input data\n"  #
        f"\n"
        f"{len(atom_to_type)} atoms\n"
        f"{len(bond_to_type)} bonds\n"
        f"{len(angle_to_type)} angles\n"
        f"{len(dihedral_to_type)} dihedrals\n"
        f"0 impropers\n"
        f"{len(cmap_5atoms)} crossterms"
        f"\n"
        f"{len(atom_type_to_index)} atom types\n"
        f"{len(bond_type_to_index)} bond types\n"
        f"{len(angle_type_to_index)} angle types\n"
        f"{len(dihedral_type_to_index)} dihedral types\n"
        f"\n"
        f"{-box_half_width} {box_half_width} xlo xhi\n"
        f"{-box_half_width} {box_half_width} ylo yhi\n"
        f"{-box_half_width} {box_half_width} zlo zhi\n"
        f"\n"
    )

    write_list(stream, "Masses", atom_type_masses)
    write_list(stream, "Bond Coeffs", bond_coeffs)
    write_list(stream, "Angle Coeffs", angle_coeffs)
    write_list(stream, "Dihedral Coeffs", dihedral_coeffs)
    write_list(stream, "Atoms", atoms)
    write_list(stream, "Bonds", bonds)
    write_list(stream, "Angles", angles)
    write_list(stream, "Dihedrals", dihedrals)
    write_list(stream, "CMAP", cmap_5atoms)

    return dict(
        atom_to_type=atom_to_type,
        atom_type_to_index=atom_type_to_index,
        bond_to_type=bond_to_type,
        bond_type_to_index=bond_type_to_index,
        angle_to_type=angle_to_type,
        angle_type_to_index=angle_type_to_index,
        dihedral_to_type=dihedral_to_type,
        dihedral_type_to_index=dihedral_type_to_index,
        crossterm_to_type=crossterm_to_type,
        crossterm_type_to_index=crossterm_type_to_index,
    )


def build_atom_to_type_dict(u):
    atom_to_type = {}
    atom_type_to_index = {}

    for atom in u.atoms:
        if atom.type == "C":
            if atom.name in {"CA", "CB"}:
                key = (atom.resname, atom.name)
            else:
                assert atom.name == "C"
                key = (None, atom.type)
        elif atom.type == "N":
            if atom.resname == "PRO":
                key = (atom.resname, atom.type)
            else:
                key = (None, atom.type)
        elif atom.type == "H":
            assert atom.resname == "GLY"
            key = (atom.resname, atom.type)
        else:
            raise Exception

        if key not in atom_type_to_index:
            index = len(atom_type_to_index) + 1  # start from 1
            atom_type_to_index[key] = index

        atom_to_type[atom] = key

    unique_residues = set(u.residues.resnames)
    assert len(atom_type_to_index) == 2 * len(unique_residues) + 2 + (
        1 if "PRO" in unique_residues else 0
    )
    return atom_to_type, atom_type_to_index


def build_atom_type_masses(atom_type_to_index):
    atom_type_masses = []
    amd = atom_masses_dict

    for key, index in atom_type_to_index.items():
        match key:
            case (None, "N"):
                mass = amd["N"] + amd["H"]
            case ("PRO", "N"):
                mass = amd["N"]
            case (None, "C"):
                mass = amd["C"] + amd["O"]
            case ("GLY", "CA"):
                # TODO: why divide by 2?
                mass = amd["C"] / 2.0 + amd["H"]
            case (resname, "CA"):
                mass = amd["C"] + amd["H"]
            case (resname, "CB") | (resname, "H"):
                mass = residue_masses_dict[convert_aa_code(resname)]
            case _:
                raise Exception

        atom_type_masses.append((index, mass))

    return atom_type_masses


def build_atoms_list(u, atom_to_type, atom_type_to_index):
    atom_specs = []

    for i, (atom, key) in enumerate(atom_to_type.items()):
        type_index = atom_type_to_index[key]

        if atom.type == "N":
            if atom.resid == 1 and len(atom.bonds) == 1:
                # N-terminus
                charge = 1
            else:
                charge = 0
        elif atom.type == "C":
            if atom.name == "CB":
                charge = charges_dict[convert_aa_code(atom.resname)]
            elif atom.resid == len(u.residues) and len(atom.bonds) == 1:
                # C-terminus
                charge = -1
            else:
                # generic CA or C
                charge = 0
        elif atom.type == "H":
            assert atom.resname == "GLY"
            charge = charges_dict[convert_aa_code(atom.resname)]
        else:
            raise Exception

        atom_specs.append(
            (i + 1, 1, type_index, charge, *atom.position.tolist(), 0, 0, 0)
        )

    return atom_specs


def build_bond_to_type_dict(u):
    bond_to_type = {}
    bond_type_to_index = {}

    # we treat the H of glycine as if it were the CB of the other residues
    CB_H_set = frozenset({"CB", "H"})

    for bond in u.bonds:
        pair = {}

        for atom in bond:
            name = atom.name if atom.type == "C" else atom.type

            if name in CB_H_set:
                name = CB_H_set

            pair[name] = atom

        # this type of bond doesn't actually exist,
        # if they appear they're just an artifact of the guessing algorithm
        if pair.keys() == {"C", CB_H_set}:
            continue

        if "CA" in pair:
            CA = pair["CA"]
            key = (CA.resname, tuple(pair))
        else:
            key = (None, tuple(pair))

        if key not in bond_type_to_index:
            index = len(bond_type_to_index) + 1
            bond_type_to_index[key] = index

        bond_to_type[bond] = key

    # for each AA type we have specific N-CA, CA-CB (CA-H for glycine) and CA-C bonds
    # plus a generic C-N bond
    unique_residues = set(u.residues.resnames)
    assert len(bond_type_to_index) == 3 * len(unique_residues) + 1
    assert len(bond_to_type) == len(u.atoms) - 1

    return bond_to_type, bond_type_to_index


def build_bond_coeffs(bond_type_to_index):
    bond_coeffs = []

    for (resname, pair), index in bond_type_to_index.items():
        pair_set = set(pair)

        if pair_set == {"N", "CA"}:
            eq_bond_len = 1.455
        elif pair_set == {"CA", frozenset({"CB", "H"})}:
            eq_bond_len = 1.09 if resname == "GLY" else 1.53
        elif pair_set == {"CA", "C"}:
            eq_bond_len = 1.524
        elif pair_set == {"C", "N"}:
            eq_bond_len = 1.334
        else:
            raise Exception

        bond_coeffs.append((index, kspring, eq_bond_len))

    return bond_coeffs


def build_bonds_list(bond_to_type, bond_type_to_index):
    bonds = []

    for i, (bond, bond_key) in enumerate(bond_to_type.items()):
        bond_index = bond_type_to_index[bond_key]

        atom_indices = [atom.ix + 1 for atom in bond]
        bonds.append((i + 1, bond_index, *atom_indices))

    return bonds


def build_angle_to_type_dict(u, bond_to_type):
    angle_to_type = {}
    angle_type_to_index = {}

    def merge_pairs(pairs):
        out = []

        for (a, b1), (b2, c) in pairwise(pairs):
            assert b1 == b2

            if not out:
                out.append(a)

            out.extend((b1, c))

        return out

    for angle in u.angles:
        missing_bond = False
        pair_types = []

        for pair in pairwise(angle):
            # the hash of Bond class depends on the order of the two indices...
            indices = [atom.ix for atom in pair]
            possible_bonds = (
                (Bond(perm, universe=u), flipped)
                for perm, flipped in zip([indices, indices[::-1]], [False, True])
            )

            match = next(
                (
                    (bond_to_type[bond], flipped)
                    for bond, flipped in possible_bonds
                    if bond in bond_to_type
                ),
                None,
            )

            if match is None:
                missing_bond = True
                break

            (_, pair_type), flipped = match
            pair_types.append(pair_type[::-1] if flipped else pair_type)

        if missing_bond:
            continue

        key = tuple(merge_pairs(pair_types))

        if key not in angle_type_to_index:
            index = len(angle_type_to_index) + 1
            angle_type_to_index[key] = index

        angle_to_type[angle] = key

    assert len(angle_to_type) == 5 * len(u.residues) - 2
    # N-CA-CO, CA-CO-N, CO-N-CA, N-CA-CB, C-CA-CB
    assert len(angle_type_to_index) == 5

    return angle_to_type, angle_type_to_index


def build_angles_list(angle_to_type, angle_type_to_index):
    angles = []

    for i, (angle, key) in enumerate(angle_to_type.items()):
        angle_index = angle_type_to_index[key]
        atom_indices = [atom.ix + 1 for atom in angle]

        angles.append((i + 1, angle_index, *atom_indices))

    return angles


def build_angle_coeffs(angle_type_to_index):
    angle_coeffs = []

    CB_H_set = frozenset({"CB", "H"})

    for key, index in angle_type_to_index.items():
        match key:
            case ("N", "CA", s) if s == CB_H_set:  # 4
                angle_coeffs.append((index, 10.0, 110.5))
            case ("N", "CA", "C"):  # 2
                angle_coeffs.append((index, 10.0, 109.0))
            case ("CA", "C", "N"):  # 3
                angle_coeffs.append((index, 10.0, 116.2))
            case (s, "CA", "C") if s == CB_H_set:  # 5
                angle_coeffs.append((index, 10.0, 111.2))
            case ("C", "N", "CA"):  # 1
                angle_coeffs.append((index, 10.0, 121.4))
            case _:
                raise Exception

    return angle_coeffs


def build_dihedral_to_type_dict(u, angle_to_type):
    dihedral_to_type = {}
    dihedral_type_to_index = {}

    for dihedral in u.dihedrals:
        if any(
            atom.type == "H" or (atom.type == "C" and atom.name == "CB")
            for atom in dihedral
        ):
            continue

        missing_angle = False

        for triplet in triplewise(dihedral):
            angle = Angle([atom.ix for atom in triplet], universe=u)

            if angle not in angle_to_type:
                missing_angle = True
                break

        if missing_angle:
            continue

        # TODO: maybe build the dih keys based on the angle keys, as we did in the previous case?
        key = tuple(atom.name if atom.type == "C" else atom.type for atom in dihedral)

        if key not in dihedral_type_to_index:
            index = len(dihedral_type_to_index) + 1
            dihedral_type_to_index[key] = index

        dihedral_to_type[dihedral] = key

    assert len(dihedral_to_type) == 3 * (len(u.residues) - 1)
    # CO-N-CA-CO, CA-CO-N-CA, N-CA-CO-N
    assert len(dihedral_type_to_index) == 3

    return dihedral_to_type, dihedral_type_to_index


def build_dihedral_coeffs(dih_type_to_index):
    dihedral_coeffs = []

    for key, index in dih_type_to_index.items():
        match key:
            case ("N", "CA", "C", "N"):  # psi
                dihedral_coeffs.append((index, 1, 0.6, 1, 0.0))
            case ("CA", "C", "N", "CA"):  # omega
                dihedral_coeffs.append((index, 2, 6.10883, 1, 0.0, 10.46, 2, 180.0))
            case ("C", "N", "CA", "C"):  # phi
                dihedral_coeffs.append((index, 1, 0.2, 1, 180.0))
            case _:
                raise Exception

    return dihedral_coeffs


def build_dihedrals_list(dihedral_to_type, dihedral_type_to_index):
    dihedrals_list = []

    for i, (dihedral, key) in enumerate(dihedral_to_type.items()):
        dihedral_index = dihedral_type_to_index[key]
        atom_indices = [atom.ix + 1 for atom in dihedral]

        dihedrals_list.append((i + 1, dihedral_index, *atom_indices))

    return dihedrals_list


def build_cmap_crossterms_to_type_dict(u):
    crossterm_to_type = {}
    crossterm_type_to_index = {}

    left_dihedrals = [
        dihedral
        for dihedral in u.dihedrals
        # consider only dihedrals that involve backbone atoms
        if all(atom.name in {"CA", "C", "N"} for atom in dihedral)
    ]

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

        for dihedral2 in left_dihedrals:
            if dihedral1[1:] != dihedral2[:-1]:
                continue

            assert all(
                atom1.residue == atom2.residue
                for atom1, atom2 in pairwise(dihedral1[1:])
            )

            key = dihedral1[1].resid

            assert key not in crossterm_type_to_index
            index = len(crossterm_type_to_index) + 1
            crossterm_type_to_index[key] = index

            atoms = dihedral1[:] + dihedral2[-1]
            crossterm_to_type[atoms] = key

            left_dihedrals.remove(dihedral2)
            break

    assert len(crossterm_to_type) == len(u.residues) - 2
    assert len(crossterm_type_to_index) == len(u.residues) - 2

    return crossterm_to_type, crossterm_type_to_index


def build_cmap_5atoms_list(crossterm_to_type, crossterm_type_to_index):
    cmap_5atoms = []

    for i, (atoms, key) in enumerate(crossterm_to_type.items()):
        crossterm_index = crossterm_type_to_index[key]
        atom_indices = [atom.ix + 1 for atom in atoms]

        cmap_5atoms.append((i + 1, crossterm_index, *atom_indices))

    return cmap_5atoms
