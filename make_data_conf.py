import io
import ujson
from ast import literal_eval

import MDAnalysis as mda
from MDAnalysis.topology.tables import masses as atom_masses_dict
from MDAnalysis.core.groups import Atom
from MDAnalysis.lib.util import convert_aa_code
from more_itertools import pairwise, triplewise


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


def write_data_config(
    stream: io.TextIOBase,
    topo: "Topology",
    init_ag: mda.AtomGroup,
    name: str,
    box_half_width: int,
    use_cmap: bool,
):
    stream.write(
        f"LAMMPS {name} input data\n"  #
        f"\n"
        f"{topo.n_atoms} atoms\n"
        f"{topo.n_bonds} bonds\n"
        f"{topo.n_angles} angles\n"
        f"{topo.n_dihedrals} dihedrals\n"
        f"0 impropers\n"
    )

    if use_cmap:
        stream.write(f"{topo.n_crossterms} crossterms\n")

    stream.write(
        f"\n"
        f"{topo.n_atom_types} atom types\n"
        f"{topo.n_bond_types} bond types\n"
        f"{topo.n_angle_types} angle types\n"
        f"{topo.n_dihedral_types} dihedral types\n"
        f"\n"
        f"{-box_half_width} {box_half_width} xlo xhi\n"
        f"{-box_half_width} {box_half_width} ylo yhi\n"
        f"{-box_half_width} {box_half_width} zlo zhi\n"
        f"\n"
    )

    write_list(stream, "Masses", topo.build_atom_type_masses())
    write_list(stream, "Bond Coeffs", topo.build_bond_coeffs())
    write_list(stream, "Angle Coeffs", topo.build_angle_coeffs())
    write_list(stream, "Dihedral Coeffs", topo.build_dihedral_coeffs())
    write_list(stream, "Atoms", topo.build_atoms_list(init_ag))
    write_list(stream, "Bonds", topo.build_bonds_list())
    write_list(stream, "Angles", topo.build_angles_list())
    write_list(stream, "Dihedrals", topo.build_dihedrals_list())

    if use_cmap:
        write_list(stream, "CMAP", topo.build_cmap_crossterms_list())

    return topo


def get_atom_id(atom: Atom):
    return (atom.segid, int(atom.resid), atom.name)


# TODO: find more specific name...
class Topology:
    def __init__(self, topo_dicts):
        self._topo_dicts = topo_dicts
        self._atom_to_type = topo_dicts["atom_to_type"]
        self._atom_to_index = topo_dicts["atom_to_index"]
        self._atom_type_to_index = topo_dicts["atom_type_to_index"]
        self._bond_to_type = topo_dicts["bond_to_type"]
        self._bond_type_to_index = topo_dicts["bond_type_to_index"]
        self._angle_to_type = topo_dicts["angle_to_type"]
        self._angle_type_to_index = topo_dicts["angle_type_to_index"]
        self._dihedral_to_type = topo_dicts["dihedral_to_type"]
        self._dihedral_type_to_index = topo_dicts["dihedral_type_to_index"]
        self._crossterm_to_type = topo_dicts["crossterm_to_type"]
        self._crossterm_type_to_index = topo_dicts["crossterm_type_to_index"]

    def to_json(self, stream):
        # TODO: stringify keys
        return ujson.dump(self._topo_dicts, stream, indent=1)

    @classmethod
    def read_json(cls, stream):
        def convert_lists_to_tuples(x):
            if not isinstance(x, list):
                return x

            return tuple(map(convert_lists_to_tuples, x))

        raw_topo_dicts = ujson.load(stream)
        topo_dicts = {}

        for name, old in raw_topo_dicts.items():
            new = {}

            for k, v in old.items():
                new[literal_eval(k)] = convert_lists_to_tuples(v)

            topo_dicts[name] = new

        return cls(topo_dicts)

    @classmethod
    def from_pdb(cls, u):
        (
            atom_to_type,
            atom_type_to_index,
        ) = cls._build_atom_to_type_dict(u)

        atom_to_index = {key: index for index, key in enumerate(atom_to_type)}

        (
            bond_to_type,
            bond_type_to_index,
        ) = cls._build_bond_to_type_dict(u, atom_to_type)

        (
            angle_to_type,
            angle_type_to_index,
        ) = cls._build_angle_to_type_dict(u, bond_to_type)

        (
            dihedral_to_type,
            dihedral_type_to_index,
        ) = cls._build_dihedral_to_type_dict(u, angle_to_type)

        (
            crossterm_to_type,
            crossterm_type_to_index,
        ) = cls._build_cmap_crossterms_to_type_dict(u)

        topo_dicts = dict(
            atom_to_type=atom_to_type,
            atom_to_index=atom_to_index,
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

        return cls(topo_dicts)

    @property
    def atom_to_type(self):
        return self._atom_to_type

    @property
    def atom_to_index(self):
        return self._atom_to_index

    @property
    def atom_type_to_index(self):
        return self._atom_type_to_index

    @property
    def bond_to_type(self):
        return self._bond_to_type

    @property
    def bond_type_to_index(self):
        return self._bond_type_to_index

    @property
    def angle_to_type(self):
        return self._angle_to_type

    @property
    def angle_type_to_index(self):
        return self._angle_type_to_index

    @property
    def dihedral_to_type(self):
        return self._dihedral_to_type

    @property
    def dihedral_type_to_index(self):
        return self._dihedral_type_to_index

    @property
    def crossterm_to_type(self):
        return self._crossterm_to_type

    @property
    def crossterm_type_to_index(self):
        return self._crossterm_type_to_index

    @property
    def n_atoms(self):
        return len(self._atom_to_type)

    @property
    def n_bonds(self):
        return len(self._bond_to_type)

    @property
    def n_angles(self):
        return len(self._angle_to_type)

    @property
    def n_dihedrals(self):
        return len(self._dihedral_to_type)

    @property
    def n_crossterms(self):
        return len(self._crossterm_to_type)

    @property
    def n_atom_types(self):
        return len(self._atom_type_to_index)

    @property
    def n_bond_types(self):
        return len(self._bond_type_to_index)

    @property
    def n_angle_types(self):
        return len(self._angle_type_to_index)

    @property
    def n_dihedral_types(self):
        return len(self._dihedral_type_to_index)

    @staticmethod
    def _build_atom_to_type_dict(u):
        atom_to_type = {}
        atom_type_to_index = {}

        for atom in u.atoms:
            if atom.type == "C":
                if atom.name in {"CA", "CB"}:
                    key = (atom.resname, atom.name, False)
                elif atom.name == "C":
                    is_terminal = (
                        int(atom.resid) == len(u.residues) and len(atom.bonds) == 1
                    )
                    key = (None, atom.type, is_terminal)
                else:
                    key = None
            elif atom.type == "N" and atom.name == "N":
                is_terminal = int(atom.resid) == 1 and len(atom.bonds) == 1
                if atom.resname == "PRO":
                    key = (atom.resname, atom.type, is_terminal)
                else:
                    key = (None, atom.type, is_terminal)
            elif atom.resname == "GLY" and atom.type == "H" and atom.name == "2HA":
                key = (atom.resname, atom.type, False)
            else:
                key = None

            if key is None:
                continue

            if key not in atom_type_to_index:
                index = len(atom_type_to_index) + 1  # start from 1
                atom_type_to_index[key] = index

            atom_to_type[get_atom_id(atom)] = key

        unique_residues = set(u.residues.resnames)
        # residue-specific CA and CB,
        # 2 generic atoms for backbone C and N,
        # 2 atoms for the terminals,
        # one special N for proline (if present)
        assert len(atom_type_to_index) == 2 * len(unique_residues) + 2 + 2 + (
            1 if "PRO" in unique_residues else 0
        )
        return atom_to_type, atom_type_to_index

    @staticmethod
    def _build_bond_to_type_dict(u, atom_to_type):
        bond_to_type = {}
        bond_type_to_index = {}

        for bond in u.bonds:
            pair = {}
            missing_atom = False

            for atom in bond:
                if get_atom_id(atom) not in atom_to_type:
                    missing_atom = True
                    break

                # TODO: this is not really necessary anymore...
                name = atom.name if atom.type == "C" else atom.type

                # we treat the H of glycine as if it were the CB of the other residues
                if name in {"CB", "H"}:
                    name = "CB_H"

                pair[name] = atom

            if missing_atom:
                continue

            # this type of bond doesn't actually exist,
            # if they appear they're just an artifact of the guessing algorithm
            if pair.keys() == {"C", "CB_H"}:
                continue

            if "CA" in pair:
                CA = pair["CA"]
                key = (CA.resname, tuple(pair))
            else:
                key = (None, tuple(pair))

            if key not in bond_type_to_index:
                index = len(bond_type_to_index) + 1
                bond_type_to_index[key] = index

            bond_inds = tuple(get_atom_id(atom) for atom in bond)
            bond_to_type[bond_inds] = key

        # for each AA type we have specific N-CA, CA-CB (CA-H for glycine) and CA-C bonds
        # plus a generic C-N bond
        unique_residues = set(u.residues.resnames)
        assert len(bond_type_to_index) == 3 * len(unique_residues) + 1
        assert len(bond_to_type) == len(u.atoms) - 1

        return bond_to_type, bond_type_to_index

    @staticmethod
    def _build_angle_to_type_dict(u, bond_to_type):
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
                bond_inds = tuple(get_atom_id(atom) for atom in pair)

                match = next(
                    (
                        (bond_to_type[correct], flipped)
                        for flipped in [True, False]
                        if (correct := bond_inds[::-1] if flipped else bond_inds)
                        in bond_to_type
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

            angle_inds = tuple(get_atom_id(atom) for atom in angle)
            angle_to_type[angle_inds] = key

        assert len(angle_to_type) == 5 * len(u.residues) - 2
        # N-CA-CO, CA-CO-N, CO-N-CA, N-CA-CB, C-CA-CB
        assert len(angle_type_to_index) == 5

        return angle_to_type, angle_type_to_index

    @staticmethod
    def _build_dihedral_to_type_dict(u, angle_to_type):
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
                angle_inds = tuple(get_atom_id(atom) for atom in triplet)

                if angle_inds not in angle_to_type:
                    missing_angle = True
                    break

            if missing_angle:
                continue

            # TODO: maybe build the dih keys based on the angle keys, as we did in the previous case?
            key = tuple(
                atom.name if atom.type == "C" else atom.type for atom in dihedral
            )

            if key not in dihedral_type_to_index:
                index = len(dihedral_type_to_index) + 1
                dihedral_type_to_index[key] = index

            dih_inds = tuple(get_atom_id(atom) for atom in dihedral)
            dihedral_to_type[dih_inds] = key

        assert len(dihedral_to_type) == 3 * (len(u.residues) - 1)
        # CO-N-CA-CO, CA-CO-N-CA, N-CA-CO-N
        assert len(dihedral_type_to_index) == 3

        return dihedral_to_type, dihedral_type_to_index

    @staticmethod
    def _build_cmap_crossterms_to_type_dict(u):
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

                key = int(dihedral1[1].resid)

                assert key not in crossterm_type_to_index
                index = len(crossterm_type_to_index) + 1
                crossterm_type_to_index[key] = index

                atoms = dihedral1[:] + dihedral2[-1]
                crossterm_inds = tuple(get_atom_id(atom) for atom in atoms)
                crossterm_to_type[crossterm_inds] = key

                left_dihedrals.remove(dihedral2)
                break

        assert len(crossterm_to_type) == len(u.residues) - 2
        assert len(crossterm_type_to_index) == len(u.residues) - 2

        return crossterm_to_type, crossterm_type_to_index

    def filter_cg_atoms(self, ag):
        return sum(
            ag.select_atoms(f"atom " + " ".join(map(str, atom_id)))
            for atom_id in self.atom_to_type
        )

    def build_atom_type_masses(self):
        atom_type_masses = []
        amd = atom_masses_dict
        rmd = residue_masses_dict

        for key, index in self.atom_type_to_index.items():
            match key:
                case (None, "N", is_terminal):
                    mass = amd["N"] + (3 * amd["H"] if is_terminal else amd["H"])
                case ("PRO", "N", False):
                    mass = amd["N"]
                case (None, "C", is_terminal):
                    mass = amd["C"] + (2 * amd["O"] if is_terminal else amd["O"])
                case ("GLY", "CA", False):
                    mass = amd["C"] / 2.0 + amd["H"]
                case (resname, "CA", False):
                    mass = amd["C"] + amd["H"]
                case ("GLY", "H", False):
                    mass = rmd[convert_aa_code(resname)] + amd["C"] / 2.0
                case (resname, "CB" | "H", False):
                    mass = rmd[convert_aa_code(resname)]
                case _:
                    raise Exception

            atom_type_masses.append((index, mass))

        return atom_type_masses

    def build_atoms_list(self, ag):
        atom_specs = []

        cg_ag = self.filter_cg_atoms(ag)

        # we assume that the center of the box is always (0, 0, 0)
        com = cg_ag.center_of_mass()
        cg_ag.translate(-com)

        atom_id_to_obj = {get_atom_id(atom): atom for atom in cg_ag}

        for i, (atom_id, type_key) in enumerate(self.atom_to_type.items()):
            type_index = self.atom_type_to_index[type_key]

            match type_key:
                case (_, "N", is_terminal):
                    charge = 1 if is_terminal else 0
                case (None, "C", is_terminal):
                    charge = -1 if is_terminal else 0
                case (resname, "CA", False):
                    charge = 0
                case (resname, "CB" | "H", False):
                    charge = charges_dict[convert_aa_code(resname)]
                case _:
                    raise Exception

            # atom = init_ag.select_atoms(f"atom " + " ".join(atom_id))[0]
            atom = atom_id_to_obj[atom_id]
            atom_specs.append(
                (i + 1, 1, type_index, charge, *atom.position.tolist(), 0, 0, 0)
            )

        return atom_specs

    def build_bond_coeffs(self):
        bond_coeffs = []
        # TODO: for now  we leave this hard-coded, then we'll see..
        kspring = 200  # in E/distance^2 units

        for (resname, pair), index in self.bond_type_to_index.items():
            pair_set = set(pair)

            if pair_set == {"N", "CA"}:
                eq_bond_len = 1.455
            elif pair_set == {"CA", "CB_H"}:
                eq_bond_len = 1.09 if resname == "GLY" else 1.53
            elif pair_set == {"CA", "C"}:
                eq_bond_len = 1.524
            elif pair_set == {"C", "N"}:
                eq_bond_len = 1.334
            else:
                raise Exception

            bond_coeffs.append((index, kspring, eq_bond_len))

        return bond_coeffs

    def build_bonds_list(self):
        bonds = []

        for i, (bond_atom_ids, bond_key) in enumerate(self.bond_to_type.items()):
            bond_atom_inds = [self.atom_to_index[atom_id] for atom_id in bond_atom_ids]
            bond_index = self.bond_type_to_index[bond_key]
            bonds.append((i + 1, bond_index) + tuple(i + 1 for i in bond_atom_inds))

        return bonds

    def build_angles_list(self):
        angles = []

        for i, (angle_atoms_ids, key) in enumerate(self.angle_to_type.items()):
            angle_atom_inds = [
                self.atom_to_index[atom_id] for atom_id in angle_atoms_ids
            ]
            angle_index = self.angle_type_to_index[key]
            angles.append((i + 1, angle_index) + tuple(i + 1 for i in angle_atom_inds))

        return angles

    def build_angle_coeffs(self):
        angle_coeffs = []

        for key, index in self.angle_type_to_index.items():
            match key:
                case ("N", "CA", "CB_H"):  # 4
                    angle_coeffs.append((index, 10.0, 110.5))
                case ("N", "CA", "C"):  # 2
                    angle_coeffs.append((index, 10.0, 109.0))
                case ("CA", "C", "N"):  # 3
                    angle_coeffs.append((index, 10.0, 116.2))
                case ("CB_H", "CA", "C"):  # 5
                    angle_coeffs.append((index, 10.0, 111.2))
                case ("C", "N", "CA"):  # 1
                    angle_coeffs.append((index, 10.0, 121.4))
                case _:
                    raise Exception

        return angle_coeffs

    def build_dihedral_coeffs(self):
        dihedral_coeffs = []

        for key, index in self.dihedral_type_to_index.items():
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

    def build_dihedrals_list(self):
        dihedrals_list = []

        for i, (dih_atom_ids, key) in enumerate(self.dihedral_to_type.items()):
            dih_atom_inds = [self.atom_to_index[atom_id] for atom_id in dih_atom_ids]
            dihedral_index = self.dihedral_type_to_index[key]
            dihedrals_list.append(
                (i + 1, dihedral_index) + tuple(i + 1 for i in dih_atom_inds)
            )

        return dihedrals_list

    def build_cmap_crossterms_list(self):
        cmap_crossterms = []

        for i, (atom_ids, key) in enumerate(self.crossterm_to_type.items()):
            atom_inds = [self._atom_to_index[atom_id] for atom_id in atom_ids]
            crossterm_index = self.crossterm_type_to_index[key]
            cmap_crossterms.append(
                (i + 1, crossterm_index) + tuple(i + 1 for i in atom_inds)
            )

        return cmap_crossterms
