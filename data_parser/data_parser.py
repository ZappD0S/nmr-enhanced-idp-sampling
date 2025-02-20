# %%
from lark import Lark
from pathlib import Path
import MDAnalysis as mda
from collections import Counter
from MDAnalysis import AtomGroup

lark_grammar_file = Path("data_grammar.lark")
parser = Lark.open(lark_grammar_file, parser="lalr")
tree = parser.parse(open("../ch2lmp_test/step1_pdbreader.data").read())

u = mda.Universe("../ch2lmp_test/step1_pdbreader.pdb")

cg_atoms = []

for atom in u.atoms:
    if (atom.resname == "GLY" and atom.name == "2HA") or atom.name in {
        "CA",
        "CB",
        "C",
        "N",
    }:
        cg_atoms.append(atom)

cg_atoms = AtomGroup(cg_atoms)


# %%

pair_coeffs_tree = next(tree.find_data("pair_coeffs"))

pair_coeffs_dict = {}

for coeffs in pair_coeffs_tree.children:
    tokens = coeffs.children
    atom_type = int(tokens[0].value)
    pair_coeffs_dict[atom_type] = [float(tok.value) for tok in tokens[1:]]

atoms_list_tree = next(tree.find_data("atoms_list"))

atom_name_to_pair_coeffs = {}
pair_coeff_to_key = {}

for i, row in enumerate(atoms_list_tree.children):
    tokens = row.children
    _, _, atom_type_tok, *coeffs_toks = tokens
    atom_type = int(atom_type_tok.value)

    atom = u.atoms[i]
    pair_coeffs = tuple(pair_coeffs_dict[atom_type])

    if atom.name in {"CA", "C", "N"}:
        # atom_name_to_pair_coeffs.setdefault(atom.name, set()).add(pair_coeffs)
        atom_name_to_pair_coeffs.setdefault(atom.name, []).append(pair_coeffs)
        pair_coeff_to_key.setdefault((atom.type, pair_coeffs), set()).add(atom.resname)


assert atom_name_to_pair_coeffs.keys() == {"CA", "C", "N"}

for name, rows in atom_name_to_pair_coeffs.items():
    print(name)
    c = Counter(rows)
    # print(c)
    print(c.most_common()[0][0])
    print()


# atom_name_to_pair_coeffs

# %%
def build_stuff(coeffs_tree, inter_tree):
    all_coeffs_dict = {}

    for coeffs in coeffs_tree.children:
        tokens = coeffs.children
        all_coeffs_dict[int(tokens[0].value)] = [float(tok.value) for tok in tokens[1:]]

    all_to_cg_type_inds = {}
    cg_coeffs_dict = {}
    cg_inter_list = []

    cg_type = 0

    for bond in inter_tree.children:
        tokens = bond.children
        bond_type = int(tokens[1])
        atom_inds = [int(tok.value) for tok in tokens[2:]]

        if not all(i - 1 in cg_atoms.ix for i in atom_inds):
            continue

        if bond_type not in all_to_cg_type_inds:
            cg_type += 1
            all_to_cg_type_inds[bond_type] = cg_type


        cg_coeffs_dict[cg_type] = all_coeffs_dict[bond_type]

        cg_inter_list.append([cg_type, tuple(atom_inds)])

    return cg_coeffs_dict, cg_inter_list

bond_coeffs_tree = next(tree.find_data("bond_coeffs"))
bonds_tree = next(tree.find_data("bonds_list"))

bond_coeffs_dict, bonds_list = build_stuff(bond_coeffs_tree, bonds_tree)

angle_coeffs_tree = next(tree.find_data("angle_coeffs"))
angles_tree = next(tree.find_data("angles_list"))

angle_coeffs_dict, angles_list = build_stuff(angle_coeffs_tree, angles_tree)

dih_coeffs_tree = next(tree.find_data("dihedral_coeffs"))
dihs_tree = next(tree.find_data("dihedrals_list"))

dih_coeffs_dict, dih_list = build_stuff(dih_coeffs_tree, dihs_tree)

imp_coeffs_tree = next(tree.find_data("improper_coeffs"))
imps_tree = next(tree.find_data("impropers_list"))

# this should be empty, and it is..
imp_coeffs_dict, imp_list = build_stuff(imp_coeffs_tree, imps_tree)


# key_to_bond_type = {}
# bond_type_to_key = {}

    # atom1, atom2 = bond_atoms = u.atoms[atom_inds]
    # resnames_set = {atom1.resname, atom2.resname}
    # neigh_resids = []

    # for i in [-1, 1]:
    #     neigh_resid = atom1.resid + i
    #     if neigh_resid < 0 or neigh_resid > len(u.residues):
    #         continue

    #     neigh_resids.append(neigh_resid)

    # neigh_resnames = u.residues[[resid - 1 for resid in neigh_resids]].resnames

    # if atom1.name == "C" and atom2.name == "N":
    #     if "PRO" in resnames_set:
    #         if "PRO" in neigh_resnames:
    #             key = ("PROPRO", ("C", "N"))
    #         else:
    #             key = ("PRO", ("C", "N"))
    #     else:
    #         key = (None, ("C", "N"))
    # elif (atom1.name == "CB" and atom2.name == "CA") or (atom1.name == "CA" and atom2.name == "CB"):
    #     if "PRO" in resnames_set:
    #         print(neigh_resnames)
    #         if "PRO" in neigh_resnames:
    #             key = ("PROPRO", ("CA", "CB"))
    #         else:
    #             key = ("PRO", ("CA", "CB"))
    #     elif "ALA" in resnames_set:
    #         key = ("ALA", ("CA", "CB"))
    #     else:
    #         key = ((atom1.resname, atom2.resname), tuple(neigh_resnames), ("CA", "CB"))
    #         # key = (None, ("CA", "CB"))
    # elif atom1.name == "N" and atom2.name == "CA":
    #         key = (None, ("N", "CA"))
    # elif atom1.name == "C" and atom2.name == "CA":
    #         key = (None, ("C", "CA"))
    # else:
    #     print(atom1.name, atom2.name)
    #     raise Exception

    # key_to_bond_type.setdefault(key, set()).add(bond_type)
    # bond_type_to_key.setdefault(bond_type, set()).add(key)
    # key_to_bond_type.setdefault(tuple((atom.resname, atom.name) for atom in bond_atoms), set()).add(bond_type)

    # print(atom_inds)
    # print(bond_coeffs_dict[bond_type])
    # print()


# dihedral_list = next(tree.find_data("dihedral_list"))
# dih_types = []

# for row in dihedral_list.children:
#     tokens = row.children
#     dih_type = int(tokens[1].value)
#     dih_atom_inds = [int(tok.value) - 1 for tok in tokens[2:6]]

#     dih_atoms = u.atoms[dih_atom_inds]

#     if dih_atoms.names.tolist() == ["C", "N", "CA", "C"]:
#         dih_types.append(dih_type)
#         print(dih_atoms.names)
#         print(dih_atoms[1].resname, dih_type)

# print(set(dih_types))
# print(Counter(dih_types))
# print(u.atoms[dih_atom_inds].names)
