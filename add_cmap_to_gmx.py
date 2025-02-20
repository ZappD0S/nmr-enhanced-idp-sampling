import argparse
from copy import copy
from pathlib import Path
import numpy as np
import parmed as pmd
import mdtraj as mdt
from parmed.topologyobjects import CmapType, Cmap

from data_utils import build_cmap_atoms
from cmap import build_phipsi, build_cmap


# def build_pmd_to_mdt_atom_map(pmd_top, mdt_top):
#     id_to_mdt_atom = {(atom.residue.index, atom.name): atom for atom in mdt_top.atoms}

#     residueNameReplacements = mdt.formats.pdb.PDBTrajectoryFile._residueNameReplacements
#     atomNameReplacements = mdt.formats.pdb.PDBTrajectoryFile._atomNameReplacements

#     pmd_atoms = [
#         atom for atom in pmd_top.atoms if atom.residue.name not in {"SOL", "CL", "NA"}
#     ]

#     pmd_to_mdt_atom = {}

#     for pmd_atom in pmd_atoms:
#         res = pmd_atom.residue
#         mdt_resname = residueNameReplacements[res.name]
#         mdt_name = atomNameReplacements[mdt_resname].get(pmd_atom.name, pmd_atom.name)
#         mdt_atom = id_to_mdt_atom[(res.idx, mdt_name)]

#         pmd_to_mdt_atom[pmd_atom] = mdt_atom

#     return pmd_to_mdt_atom


def build_cmaps(ens_phipsi, ref_phipsi):
    assert ens_phipsi.shape[0] == ref_phipsi.shape[0]

    cmaps = []
    n_res = ens_phipsi.shape[0]

    for i in range(1, n_res - 1):
        sel_ens_phipsi = ens_phipsi[i]
        sel_ref_phipsi = ref_phipsi[i]

        cmap, *_ = build_cmap(
            sel_ens_phipsi, sel_ref_phipsi, basepath=f"./plots/resid{i + 1}"
        )

        cmaps.append(cmap)

    cmaps = np.stack(cmaps, axis=0)
    nans_arr = np.full((1,) + cmaps.shape[1:], np.nan)

    return np.concatenate([nans_arr, cmaps, nans_arr], axis=0)


def add_cmap_to_pmd_top(pmd_top, cmaps, cmap_atoms_list):
    cmap_types = []
    cmap_objs = []

    def add_idx_to_atom(atom, i):
        atom_type = copy(atom.atom_type)
        new_name = atom_type.name + f"_{i}"
        atom_type.name = new_name
        atom.atom_type = atom_type
        atom.type = new_name

    c_first = cmap_atoms_list[0][0]
    add_idx_to_atom(c_first, 0)

    n_last = cmap_atoms_list[-1][-1]
    add_idx_to_atom(n_last, len(cmap_atoms_list) + 1)

    for i, cmap_atoms in enumerate(cmap_atoms_list):
        for atom in cmap_atoms[1:-1]:
            add_idx_to_atom(atom, i + 1)

    for cmap_atoms in cmap_atoms_list:
        # the third (index is 2) is the CA
        cmap_ind = cmap_atoms[2].residue.idx
        assert cmap_atoms[2].name == "CA"

        cmap = cmaps[cmap_ind]

        cmap_type = CmapType(resolution=24, grid=cmap.flatten().tolist())
        cmap_type.used = True
        cmap_obj = Cmap(*cmap_atoms, type=cmap_type)
        cmap_types.append(cmap_type)
        cmap_objs.append(cmap_obj)

    cmap_pmd_top = copy(pmd_top)

    cmap_pmd_top.cmap_types.clear()
    cmap_pmd_top.cmap_types.extend(cmap_types)
    cmap_pmd_top.cmap_types.claim()
    cmap_pmd_top.cmap_types.index_members()

    cmap_pmd_top.cmaps.clear()
    cmap_pmd_top.cmaps.extend(cmap_objs)
    cmap_pmd_top.cmaps.claim()
    cmap_pmd_top.cmaps.index_members()

    return cmap_pmd_top


def get_line_of_type(lines: list[str], target):
    for i, line in enumerate(lines):
        tokens = line.split()

        if tokens != ["[", "moleculetype", "]"]:
            continue

        names = []

        for line in lines[i + 1 :]:
            stripped = line.lstrip()

            if stripped.startswith(";") or stripped == "":
                continue

            if stripped.startswith("["):
                break

            names.append(stripped.split()[0])

        if len(names) != 1:
            raise Exception

        name = names[0]

        if name == target:
            return i

    raise Exception


def add_macros_to_topol(topol_file):
    topol_file = Path(topol_file)

    with topol_file.open("r") as f:
        lines = f.readlines()

    i_sol = get_line_of_type(lines, "SOL")

    lines[i_sol:i_sol] = ["#ifdef POSRES\n", '#include "posre.itp"\n', "#endif\n", "\n"]

    i_ions = min(get_line_of_type(lines, ion_type) for ion_type in ["NA", "CL"])

    lines[i_ions:i_ions] = [
        "#ifdef POSRES_WATER\n",
        "; Position restraint for each water oxygen\n",
        "[ position_restraints ]\n",
        ";  i funct       fcx        fcy        fcz\n",
        "   1    1       1000       1000       1000\n",
        "#endif\n",
        "\n",
    ]

    topol_file.rename(topol_file.with_suffix(topol_file.suffix + ".bak"))

    with topol_file.open("w") as f:
        f.writelines(lines)


def add_cmap_to_topol(topol_file, xyz_file, ens_trajs, ref_trajs):

    ref_phipsi = np.concatenate([build_phipsi(frame) for frame in ref_trajs], axis=1)
    ens_phipsi = np.concatenate([build_phipsi(frame) for frame in ens_trajs], axis=1)

    pmd_top = pmd.load_file(topol_file, xyz=xyz_file)
    cmap_atoms_list = build_cmap_atoms(
        [[getattr(dih, f"atom{i+1}") for i in range(4)] for dih in pmd_top.dihedrals]
    )

    cmaps = build_cmaps(ens_phipsi, ref_phipsi)
    cmap_pmd_top = add_cmap_to_pmd_top(pmd_top, cmaps, cmap_atoms_list)

    topol_file = Path(topol_file)
    cmap_topol_file = topol_file.with_stem(topol_file.stem + "_cmap")

    cmap_pmd_top.save(str(cmap_topol_file), overwrite=True)
    fixed_cmap_topol_file = add_macros_to_topol(cmap_topol_file)

    return fixed_cmap_topol_file


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("-ens", type=str, nargs="+", required=True)
    build_parser.add_argument("-ref", type=str, nargs="+", required=True)
    build_parser.add_argument("--ref-stride", type=int, default=10)
    build_parser.add_argument("-o", type=str, required=True)

    inject_parser = subparsers.add_parser("inject")
    inject_parser.add_argument("-top", type=str, required=True)
    inject_parser.add_argument("-xyz", type=str, required=True)
    inject_parser.add_argument("-cmap", type=str, required=True)
    inject_parser.add_argument("-o", type=str, required=True)

    args = parser.parse_args()

    if args.command == "build":
        ens_trajs = []
        for pdb in args.ens:
            ens_trajs.append(mdt.load(pdb))

        ref_trajs = []
        for i in range(0, len(args.ref), 2):
            pdb, xtc = args.ref[i : i + 2]
            ref_trajs.append(mdt.load(xtc, top=pdb, stride=args.ref_stride))

        ens_phipsi = np.concatenate(
            [build_phipsi(frame) for frame in ens_trajs], axis=1
        )
        ref_phipsi = np.concatenate(
            [build_phipsi(frame) for frame in ref_trajs], axis=1
        )

        cmaps = build_cmaps(ens_phipsi, ref_phipsi)
        np.save(args.o, cmaps)

    elif args.command == "inject":

        pmd_top = pmd.load_file(args.top, xyz=args.xyz)
        cmap_atoms_list = build_cmap_atoms(
            [
                [getattr(dih, f"atom{i+1}") for i in range(4)]
                for dih in pmd_top.dihedrals
            ]
        )

        cmaps = np.load(args.cmap)
        cmap_pmd_top = add_cmap_to_pmd_top(pmd_top, cmaps, cmap_atoms_list)

        topol_file = Path(args.top)

        cmap_topol_file = topol_file.with_stem(topol_file.stem + "_cmap")
        cmap_pmd_top.save(str(cmap_topol_file), overwrite=True)
        add_macros_to_topol(cmap_topol_file)
