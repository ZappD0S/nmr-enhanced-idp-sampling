from collections import defaultdict
from itertools import pairwise
from math import sqrt

import numpy as np
from data_utils import (
    filter_atoms,
    get_atom_id,
    is_terminus,
    r0_dict,
    epsilon_dict,
    residue_masses_dict,
    build_cmap_atoms,
)
from base_data_builder import BaseDataBuilder
from pathlib import Path
import MDAnalysis as mda
from lark import Lark
from MDAnalysis.topology.tables import masses as atom_masses_dict
from MDAnalysis.lib.util import convert_aa_code
from itertools import combinations, combinations_with_replacement


class CharmmDataBuilder(BaseDataBuilder):
    @property
    def cg_atom_ids(self):
        return self._cg_atom_ids

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
    def n_impropers(self):
        return len(self._imp_list)

    @property
    def n_crossterms(self):
        return len(self._crossterm_to_type)

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

    @property
    def n_improper_types(self):
        return len(self._imp_coeffs)

    @property
    def crossterm_ind_to_resids(self):
        for key, resids in self._crossterm_type_to_resids.items():
            ind = self._crossterm_type_to_ind[key]

            yield ind, resids

    def __init__(self, chm2lmp_data: Path, all_ag: mda.AtomGroup) -> None:
        super().__init__()

        # TODO: the .lark file needs to be in the same folder..
        parser = Lark.open("data_parser/data_grammar.lark", parser="lalr")
        self._all_ag = all_ag
        self._cg_ag = self._build_cg_ag(all_ag)

        cg_mask = self._build_cg_mask(all_ag)
        all_atom_ids = [get_atom_id(atom) for atom in all_ag]
        self._cg_atom_ids = [
            atom_id
            for atom_id, is_cg in zip(all_atom_ids, cg_mask, strict=True)
            if is_cg
        ]

        assert all(atom.ix == i for i, atom in enumerate(self._all_ag))

        self._all_ind_to_atom = {atom.ix: get_atom_id(atom) for atom in self._all_ag}
        self._cg_atom_to_ind = {
            get_atom_id(atom): i for i, atom in enumerate(self._cg_ag)
        }

        data_tree = parser.parse(chm2lmp_data.read_text())

        (
            self._atom_to_type,
            self._atom_type_to_index,
            self._atom_type_to_coeffs,
            self._atom_charges_dict,
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

        self._imp_coeffs, self._imp_list = self._coeffs_inters_lists(
            imp_coeffs_tree, imps_tree
        )

        (
            self._crossterm_to_type,
            self._crossterm_type_to_ind,
            self._crossterm_type_to_resids,
        ) = self._build_cmap_to_type_dict()

    @staticmethod
    def _is_cg_atom(atom):
        return (atom.name in {"CA", "CB", "C", "N", "O", "HA", "HN"}) or (
            (atom.resname == "GLY") and (atom.name in {"HA1", "HA2"})
        )

    def _build_cg_ag(self, all_ag):
        cg_atoms = []

        for atom in all_ag:
            if self._is_cg_atom(atom):
                cg_atoms.append(atom)

        return mda.AtomGroup(cg_atoms)

    def _build_cg_mask(self, all_ag):
        mask = []

        for atom in all_ag:
            is_cg = self._is_cg_atom(atom)
            mask.append(is_cg)

        return np.array(mask, dtype=bool)

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
        raise NotImplementedError
        # pair_coeffs = []

        # for atom_type, coeffs in self._atom_type_to_coeffs.items():
        #     index = self._atom_type_to_index[atom_type]
        #     pair_coeffs.append((index, *coeffs))

        # return pair_coeffs

    def build_mixed_pair_coeffs(self):
        mixed_pair_coeffs = []

        for (key_i, i), (key_j, j) in combinations_with_replacement(
            self._atom_type_to_index.items(), 2
        ):
            eps_i, sigma_i = self._atom_type_to_coeffs[key_i]
            eps_j, sigma_j = self._atom_type_to_coeffs[key_j]

            pair_inds = (i, j) if i < j else (j, i)
            eps_ij = (eps_i * eps_j) ** 0.5

            _, atom_name_i, _ = key_i
            _, atom_name_j, _ = key_j

            # see https://www.desmos.com/calculator/rtdnjrfqse

            if atom_name_i == "CB" and atom_name_j == "CB":
                # sidechain-sidechain interaction
                sigma_i /= 2 / sqrt(3)
                sigma_j /= 2 / sqrt(3)

                sigma_ij = (sigma_i * sigma_j) ** 0.5
                nm_exps = (8, 6)
            else:
                if atom_name_i == "CB":
                    sigma_i /= 2 ** (1 / 6)

                if atom_name_j == "CB":
                    sigma_j /= 2 ** (1 / 6)

                sigma_ij = 0.5 * (sigma_i + sigma_j)
                nm_exps = (12, 6)

            mixed_pair_coeffs.append(
                tuple(ind + 1 for ind in pair_inds)
                + ("mie/cut", eps_ij, sigma_ij)
                + nm_exps
            )

        return mixed_pair_coeffs

    def _build_atom_to_type_dict(self, tree):
        pair_coeffs_tree = next(tree.find_data("pair_coeffs"))
        atoms_list_tree = next(tree.find_data("atoms_list"))

        pair_coeffs_dict = self._build_coeffs_dict(pair_coeffs_tree)
        atom_specs_dict = self._build_coeffs_dict(atoms_list_tree)

        bb_atoms = []
        cb_atoms = []

        atom_to_coeffs = {}
        atom_charges_dict = defaultdict(float)

        for atom, row in zip(self._all_ag, atom_specs_dict.values(), strict=True):
            resid, all_atom_type, charge, *_ = row

            is_bb = self._is_cg_atom(atom) and not atom.name == "CB"
            name = atom.name if is_bb else "CB"
            atom_charges_dict[(resid, name)] += charge

            # take only the first two, which are epsilon and sigma
            pair_coeffs = pair_coeffs_dict[all_atom_type][:2]
            atom_to_coeffs[get_atom_id(atom)] = tuple(pair_coeffs)

        eps = np.finfo(float).eps
        for key, charge in atom_charges_dict.items():
            if np.abs(charge) < eps:
                atom_charges_dict[key] = 0.0

        for atom in self._cg_ag:
            if atom.name == "CB":
                cb_atoms.append(atom)
            else:
                bb_atoms.append(atom)

        bb_atoms = mda.AtomGroup(bb_atoms)
        cb_atoms = mda.AtomGroup(cb_atoms)

        atom_type_to_ids = {}
        atom_type_to_resnames = {}

        n_terms = 0

        for atom in bb_atoms:
            pair_coeffs = atom_to_coeffs[get_atom_id(atom)]
            is_term = is_terminus(atom)
            if is_term:
                n_terms += 1

            key = (pair_coeffs, atom.name, is_term)

            atom_type_to_ids.setdefault(key, []).append(get_atom_id(atom))
            atom_type_to_resnames.setdefault(key, set()).add(atom.resname)

        assert n_terms == 2

        # swap coeffs with resnames set
        atom_type_to_coeffs = {}

        for key, resnames in atom_type_to_resnames.items():
            pair_coeffs, atom_name, is_term = key
            new_key = (tuple(sorted(resnames)), atom_name, is_term)
            assert new_key not in atom_type_to_coeffs
            atom_type_to_coeffs[new_key] = pair_coeffs

            assert new_key not in atom_type_to_ids
            atom_type_to_ids[new_key] = atom_type_to_ids.pop(key)

        for atom in cb_atoms:
            assert atom.name == "CB"
            key = ((atom.resname,), "CB", False)
            atom_type_to_ids.setdefault(key, []).append(get_atom_id(atom))

            if key in atom_type_to_coeffs:
                continue

            one_letter_resname = convert_aa_code(atom.resname)
            sigma = r0_dict[one_letter_resname]
            epsilon = epsilon_dict[one_letter_resname]
            pair_coeffs = (epsilon, sigma)
            atom_type_to_coeffs[key] = pair_coeffs

        atom_to_type = {
            atom_id: atom_type
            for atom_type, atom_ids in atom_type_to_ids.items()
            for atom_id in atom_ids
        }
        atom_type_to_index = {
            atom_type: i for i, atom_type in enumerate(atom_type_to_coeffs)
        }

        return atom_to_type, atom_type_to_index, atom_type_to_coeffs, atom_charges_dict

    def build_atom_type_masses(self):
        atom_type_masses = []
        amd = atom_masses_dict
        rmd = residue_masses_dict

        for atom_type, index in self._atom_type_to_index.items():
            match atom_type:
                case _, "C", is_term:
                    mass = amd["C"] + (2 * amd["O"] if is_term else 0)
                case _, "N", is_term:
                    mass = amd["N"] + (3 * amd["H"] if is_term else amd["H"])
                case _, "O", False:
                    mass = amd["O"]
                case _, "CA", False:
                    mass = amd["C"]
                case _, ("HA" | "HN" | "HA1" | "HA2"), False:
                    mass = amd["H"]
                case [resname], "CB", False:
                    mass = rmd[convert_aa_code(resname)]
                case _:
                    raise Exception

            atom_type_masses.append((index + 1, mass))

        return atom_type_masses

    def filter_cg_atoms(self, ag):
        return filter_atoms(ag, (get_atom_id(atom) for atom in self._cg_ag))

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
            atom_id = get_atom_id(atom)
            atom_type = self._atom_to_type[atom_id]
            charge = self._atom_charges_dict[atom_id]

            type_index = self._atom_type_to_index[atom_type]
            index = self._cg_atom_to_ind[get_atom_id(atom)] + 1

            atoms_list.append(
                (
                    index,
                    atom.resid,
                    type_index + 1,
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

    def build_improper_coeffs(self):
        return self._imp_coeffs

    def build_angles_list(self):
        # TODO: remove last two columns and use style harmonic
        return self._angles_list

    def build_dihedral_coeffs(self):
        harmonic_dih_coeffs = []

        for row in self._dih_coeffs:
            index, *coeffs, _ = row
            harmonic_dih_coeffs.append((index, 1, *coeffs))

        return harmonic_dih_coeffs

    def build_dihedrals_list(self):
        return self._dih_list

    def build_impropers_list(self):
        return self._imp_list

    def _build_cmap_to_type_dict(self):
        dihedrals = []
        for row in self._dih_list:
            _, _, *dih_inds = row
            assert len(dih_inds) == 4
            dih_atoms = self._cg_ag[[ind - 1 for ind in dih_inds]]
            dihedrals.append(dih_atoms)

        crossterm_atoms_list = build_cmap_atoms(dihedrals)

        crossterm_to_type = {}
        crossterm_type_to_ind = {}
        crossterm_type_to_resids = {}

        for crossterm_atoms in crossterm_atoms_list:
            assert all(
                atom1.residue == atom2.residue
                for atom1, atom2 in pairwise(crossterm_atoms[1:-1])
            )

            # each residue of the chain has its own cmap crossterm
            key = int(crossterm_atoms[1].resid)
            crossterm_atom_ids = tuple(get_atom_id(atom) for atom in crossterm_atoms)

            crossterm_to_type[crossterm_atom_ids] = key

            if key not in crossterm_type_to_ind:
                crossterm_type_to_ind[key] = len(crossterm_type_to_ind)

            crossterm_type_to_resids.setdefault(key, []).append(
                crossterm_atoms[1].resid
            )

        return crossterm_to_type, crossterm_type_to_ind, crossterm_type_to_resids

    def build_cmap_crossterms_list(self):
        cmap_crossterms_list = []

        for i, (crossterm_ids, key) in enumerate(self._crossterm_to_type.items()):
            crossterm_type_ind = self._crossterm_type_to_ind[key]

            atom_inds = [self._cg_atom_to_ind[atom_id] + 1 for atom_id in crossterm_ids]

            cmap_crossterms_list.append((i + 1, crossterm_type_ind + 1, *atom_inds))

        return cmap_crossterms_list
