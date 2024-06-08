from data_utils import (
    filter_atoms,
    get_atom_id,
    is_terminus,
    r0_dict,
    epsilon_dict,
    residue_masses_dict,
    charges_dict,
    build_cmap_atoms,
)
from base_data_builder import BaseDataBuilder
from pathlib import Path
import MDAnalysis as mda
from lark import Lark
from MDAnalysis.topology.tables import masses as atom_masses_dict
from MDAnalysis.lib.util import convert_aa_code


class CharmmDataBuilder(BaseDataBuilder):
    @property
    def n_atoms(self):
        return len(self._atom_to_type)

    @property
    def n_bonds(self):
        return len(self._bonds_list)

    @property
    def n_angles(self):
        return len(self._angles_list)

    @property
    def n_dihedrals(self):
        return len(self._dih_list)

    @property
    def n_crossterms(self):
        raise NotImplementedError

    @property
    def n_atom_types(self):
        return len(self._atom_type_to_index)

    @property
    def n_bond_types(self):
        return len(self._bond_coeffs)

    @property
    def n_angle_types(self):
        return len(self._angle_coeffs)

    @property
    def n_dihedral_types(self):
        return len(self._dih_coeffs)

    # @property
    # def atom_types(self):
    #     return self._atom_type_to_index.keys()

    def __init__(
        self, lark_grammar: Path, chm2lmp_data: Path, all_ag: mda.AtomGroup
    ) -> None:
        super().__init__()

        parser = Lark.open(lark_grammar, parser="lalr")
        self._ref_all_ag = all_ag
        self._ref_cg_ag = self._build_cg_ag(all_ag)

        self._all_ind_to_atom = {
            atom.ix: get_atom_id(atom) for atom in self._ref_all_ag
        }
        self._cg_atom_to_ind = {
            get_atom_id(atom): i for i, atom in enumerate(self._ref_cg_ag)
        }

        data_tree = parser.parse(chm2lmp_data.read_text())

        (
            self._atom_to_type,
            self._atom_type_to_index,
            self._atom_type_to_coeffs,
        ) = self._build_atom_to_type_dict(data_tree)

        bond_coeffs_tree = next(data_tree.find_data("bond_coeffs"))
        bonds_tree = next(data_tree.find_data("bonds_list"))

        self._bond_coeffs, self._bonds_list = self._coeffs_inters_lists(
            bond_coeffs_tree, bonds_tree
        )

        angle_coeffs_tree = next(data_tree.find_data("angle_coeffs"))
        angles_tree = next(data_tree.find_data("angles_list"))

        self._angle_coeffs, self._angles_list = self._coeffs_inters_lists(
            angle_coeffs_tree, angles_tree
        )

        dih_coeffs_tree = next(data_tree.find_data("dihedral_coeffs"))
        dihs_tree = next(data_tree.find_data("dihedrals_list"))

        self._dih_coeffs, self._dih_list = self._coeffs_inters_lists(
            dih_coeffs_tree, dihs_tree
        )

        imp_coeffs_tree = next(data_tree.find_data("improper_coeffs"))
        imps_tree = next(data_tree.find_data("impropers_list"))

        imp_coeffs, imp_list = self._coeffs_inters_lists(imp_coeffs_tree, imps_tree)
        assert len(imp_coeffs) == 0 and len(imp_list) == 0

    def _build_cg_ag(self, all_ag):
        cg_atoms = []

        for atom in all_ag:
            if atom.name in {"CA", "CB", "C", "N"} or (
                atom.resname == "GLY" and atom.name == "2HA"
            ):
                cg_atoms.append(atom)

        return mda.AtomGroup(cg_atoms)

    def _build_coeffs_dict(self, coeffs_tree):
        coeffs_dict = {}

        for coeffs in coeffs_tree.children:
            index_tok, *coeff_toks = coeffs.children
            index = int(index_tok.value)
            coeffs = []

            for tok in coeff_toks:
                match tok.type:
                    case "SIGNED_INT":
                        parse = int
                    case "SIGNED_FLOAT":
                        parse = float
                    case _:
                        raise Exception

                coeffs.append(parse(tok.value))

            coeffs_dict[index] = coeffs

        return coeffs_dict

    def _coeffs_inters_lists(self, coeffs_tree, inter_tree):
        all_coeffs_dict = self._build_coeffs_dict(coeffs_tree)
        all_to_cg_type_inds = {}
        cg_coeffs_list = []
        cg_inters_list = []

        cg_type = 0

        # NOTE: here the "all" refers to all-atom (as opposed to coarse-grained..)
        for inter_row in inter_tree.children:
            _, all_type_tok, *atom_ind_toks = inter_row.children
            all_type = int(all_type_tok.value)
            all_atom_inds = [int(tok.value) - 1 for tok in atom_ind_toks]
            cg_atom_inds = []

            missing = False

            for ind in all_atom_inds:
                atom_id = self._all_ind_to_atom[ind]

                if atom_id not in self._cg_atom_to_ind:
                    missing = True
                    break

                cg_atom_inds.append(self._cg_atom_to_ind[atom_id] + 1)

            if missing:
                continue

            assert len(cg_atom_inds) == len(all_atom_inds)

            if all_type not in all_to_cg_type_inds:
                cg_type += 1
                all_to_cg_type_inds[all_type] = cg_type
                coeffs = all_coeffs_dict[all_type]
                cg_coeffs_list.append((cg_type, *coeffs))

            cg_inters_list.append(
                (len(cg_inters_list) + 1, all_to_cg_type_inds[all_type], *cg_atom_inds)
            )

        return cg_coeffs_list, cg_inters_list

    def build_pair_coeffs(self):
        pair_coeffs = []

        for atom_type, coeffs in self._atom_type_to_coeffs.items():
            index = self._atom_type_to_index[atom_type]
            pair_coeffs.append((index, *coeffs))

        return pair_coeffs

    def _build_atom_to_type_dict(self, tree):
        pair_coeffs_tree = next(tree.find_data("pair_coeffs"))
        atoms_list_tree = next(tree.find_data("atoms_list"))

        pair_coeffs_dict = self._build_coeffs_dict(pair_coeffs_tree)

        bb_atoms = []
        other_atoms = []

        atom_to_coeffs = {}

        for atom, row in zip(self._ref_all_ag, atoms_list_tree.children, strict=True):
            tokens = row.children
            all_atom_type = int(tokens[2].value)
            pair_coeffs = tuple(pair_coeffs_dict[all_atom_type])
            atom_to_coeffs[get_atom_id(atom)] = pair_coeffs

        for atom in self._ref_cg_ag:
            if atom.name in {"CA", "C", "N"}:
                bb_atoms.append(atom)
            else:
                other_atoms.append(atom)

        bb_atoms = mda.AtomGroup(bb_atoms)
        other_atoms = mda.AtomGroup(other_atoms)

        atom_type_to_ids = {}
        atom_type_to_resnames = {}

        n_terms = 0

        for atom in bb_atoms:
            pair_coeffs = atom_to_coeffs[get_atom_id(atom)]
            is_term = is_terminus(atom)
            if is_term:
                n_terms += 1
            # this is just to have a separate entry for proline's backbone nitrogen
            is_npro = atom.type == "N" and atom.resname == "PRO"
            key = (pair_coeffs, atom.name, is_term, is_npro)

            atom_type_to_ids.setdefault(key, []).append(get_atom_id(atom))
            atom_type_to_resnames.setdefault(key, set()).add(atom.resname)

        assert n_terms == 2
        # swap coeffs with resnames set
        atom_type_to_coeffs = {}

        for key, resnames in atom_type_to_resnames.items():
            pair_coeffs, atom_name, is_term, _ = key
            new_key = (tuple(sorted(resnames)), atom_name, is_term)
            assert new_key not in atom_type_to_coeffs
            atom_type_to_coeffs[new_key] = pair_coeffs

            assert new_key not in atom_type_to_ids
            atom_type_to_ids[new_key] = atom_type_to_ids.pop(key)

        for atom in other_atoms:
            assert atom.name == "CB" or (atom.resname == "GLY" and atom.name == "2HA")

            key = ((atom.resname,), atom.name, False)
            atom_type_to_ids.setdefault(key, []).append(get_atom_id(atom))

            if key in atom_type_to_coeffs:
                continue

            one_letter_resname = convert_aa_code(atom.resname)
            sigma = r0_dict[one_letter_resname]
            epsilon = epsilon_dict[one_letter_resname]
            # TODO: check order and units...
            pair_coeffs = (epsilon, sigma) * 2
            atom_type_to_coeffs[key] = pair_coeffs

        atom_to_type = {
            atom_id: atom_type
            for atom_type, atom_ids in atom_type_to_ids.items()
            for atom_id in atom_ids
        }
        atom_type_to_index = {
            atom_type: i + 1 for i, atom_type in enumerate(atom_type_to_coeffs)
        }

        return atom_to_type, atom_type_to_index, atom_type_to_coeffs

    def build_atom_type_masses(self):
        atom_type_masses = []
        amd = atom_masses_dict
        rmd = residue_masses_dict

        for atom_type, index in self._atom_type_to_index.items():
            match atom_type:
                case _, "C", is_term:
                    mass = amd["C"] + (2 * amd["O"] if is_term else amd["O"])
                case ["PRO"], "N", False:
                    mass = amd["N"]
                case _, "N", is_term:
                    mass = amd["N"] + (3 * amd["H"] if is_term else amd["H"])
                case _, "CA", False:
                    mass = amd["C"] + amd["H"]
                case [resname], "CB" | "2HA", False:
                    mass = rmd[convert_aa_code(resname)]
                case _:
                    raise Exception

            atom_type_masses.append((index, mass))

        return atom_type_masses

    def filter_cg_atoms(self, ag):
        return filter_atoms(ag, (get_atom_id(atom) for atom in self._ref_cg_ag))

    def build_atoms_list(self, ag):
        # what about the charges?
        # we could keep the CHARMM charge for the CA C N
        # and add the rest of the residue partial charges for the CB
        atoms_list = []

        # cg_ag = filter_atoms(ag, map(get_atom_id, self._ref_cg_ag))
        cg_ag = self.filter_cg_atoms(ag)

        # we assume that the center of the box is always (0, 0, 0)
        com = cg_ag.center_of_mass()
        cg_ag.translate(-com)

        for atom in cg_ag:
            atom_type = self._atom_to_type[get_atom_id(atom)]

            match atom_type:
                case (_, "N", is_term):
                    charge = 1 if is_term else 0
                case (_, "C", is_term):
                    charge = -1 if is_term else 0
                case (_, "CA", False):
                    charge = 0
                case ([resname], "CB" | "2HA", False):
                    charge = charges_dict[convert_aa_code(resname)]
                case _:
                    raise Exception

            type_index = self._atom_type_to_index[atom_type]
            index = self._cg_atom_to_ind[get_atom_id(atom)] + 1

            atoms_list.append(
                (
                    index,
                    atom.resid,
                    type_index,
                    charge,
                    *atom.position.tolist(),
                    0,
                    0,
                    0,
                )
            )

        return atoms_list

    def build_bond_coeffs(self):
        return self._bond_coeffs

    def build_bonds_list(self):
        return self._bonds_list

    def build_angle_coeffs(self):
        return self._angle_coeffs

    def build_angles_list(self):
        return self._angles_list

    def build_dihedral_coeffs(self):
        return self._dih_coeffs

    def build_dihedrals_list(self):
        return self._dih_list

    def build_cmap_crossterms_list(self):
        dihedrals = []
        for row in self._dih_list:
            _, _, *dih_inds = row
            dih_atoms = self._ref_cg_ag[dih_inds]
            dihedrals.append(dih_atoms)

        cmap_atoms_list = build_cmap_atoms(dihedrals)

        for cmap_atoms in cmap_atoms_list:

            atom_inds = [self._cg_atom_to_ind[get_atom_id(atom)] for atom in cmap_atoms]

        # TODO: we need a dict that maps cmap type to resids list (or set..)

        raise NotImplementedError
