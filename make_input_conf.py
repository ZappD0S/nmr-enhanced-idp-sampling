import argparse
import io
import re
import pathlib
import subprocess
from math import floor
from random import randint
import tempfile

import numpy as np
import MDAnalysis as mda
import mdtraj as mdt
from mdtraj.formats import PDBTrajectoryFile, XTCTrajectoryFile


from write_data_config import write_data_config
from base_data_builder import BaseDataBuilder
from charmm_data_builder import CharmmDataBuilder


def write_cmap_tables(stream: io.TextIOBase, cmaps_dict, angles):
    for index, cmap in sorted(cmaps_dict.items(), key=lambda kv: kv[0]):
        stream.write(f"# type {index + 1}\n\n")

        # for angle, col in zip(angles, cmap.T, strict=True):
        for angle, col in zip(angles, cmap, strict=True):
            stream.write(f"# {angle}\n")

            for i in range(0, len(col), 5):
                chunk = col[i : i + 5]
                stream.write(" ".join(map("{:.6f}".format, chunk)) + " \n")

            stream.write("\n")

        stream.write("\n")


def write_input_config(
    stream: io.TextIOBase,
    args: argparse.Namespace,
    data_conf_fname: str,
    output_traj_fname: str,
    cmap_fname: str | None = None,
):
    cmap_name = "mycmap"
    Nsteps = floor(args.sim_time * 1e6 / args.time_step)

    # inv_kappa = np.sqrt(args.conc) / 3.04
    inv_kappa = 1.4

    stream.write(
        f"# comment\n"  #
        f"\n"
        f"units real\n"
        f"package gpu 0 neigh no\n"
        f"atom_style full\n"
        f"bond_style harmonic\n"
        f"angle_style charmm\n"
        f"dihedral_style fourier\n"
        f"improper_style harmonic\n"
        f"special_bonds charmm\n"
        f"\n"
        f"pair_style hybrid/overlay mie/cut 8.0 coul/debye {inv_kappa} 8.0\n"
        # f"pair_style lj/charmm/coul/long 8.0 10.0\n"
        # f"pair_modify mix arithmetic\n"
        # f"suffix off\n"
        # f"kspace_style pppm 1e-4\n"
        # f"suffix on\n"
    )

    stream.write("\n")

    if cmap_fname is not None:
        stream.write(
            f"fix {cmap_name} all cmap {cmap_fname}\n"  #
            f"fix_modify {cmap_name} energy yes\n"
        )

    stream.write(
        f"read_data {data_conf_fname}"
        + (f" fix {cmap_name} crossterm CMAP" if cmap_fname is not None else "")
        + "\n"
    )

    stream.write("\n")

    stream.write(
        f"pair_coeff * * coul/debye\n"
        f"neighbor 2.0 bin\n"
        f"neigh_modify delay 5 every 1\n"
        f"\n"
        f"timestep {args.time_step}\n"
        f"thermo_style multi\n"
        f"thermo 1000\n"
        f"\n"
        # f"minimize 1.0e-4 1.0e-6 10000 100000\n"
        f"minimize 0.0 1.0e-8 1000 100000\n"
        f"fix 1 all nve\n"
        # TODO: make the seed configurable
        # f"fix 3 all langevin {args.temp} {args.temp} $(100.0*dt) 12345\n"
        f"fix 3 all langevin {args.temp} {args.temp} 100.0 12345\n"
        # f"fix 3 all langevin {0.0} {args.temp} $(100.0*dt) 12345\n"
        # f"fix 2 all shake 0.0001 5 0 m 1.0 a 232\n"
        f"\n"
        f"dump 1 all xtc 250 {output_traj_fname}\n"
        f"run {Nsteps}\n"
        # f"run 1000000\n"
    )


def write_configs(
    subfolder: pathlib.Path,
    init_ag: mda.AtomGroup,
    data_builder: BaseDataBuilder,
    args: argparse.Namespace,
    ens_ref_traj=None,
):
    # for now we leave these hardcoded...
    name = "CG"
    output_traj_fname = "traj.xtc"

    use_cmap = ens_ref_traj is not None

    main_config_file = subfolder / f"in.{name}"
    data_conf_file = subfolder / f"data.{name}"
    cmap_file = subfolder / f"{name}.cmap" if use_cmap else None

    with main_config_file.open("w") as f:
        write_input_config(
            f,
            args,
            data_conf_file.name,
            output_traj_fname,
            cmap_file.name if use_cmap else None,
        )

    with open(data_conf_file, "w") as f:
        write_data_config(f, data_builder, init_ag, name, args.box_half_width, use_cmap)

    if cmap_file is None:
        return

    if ens_ref_traj is None:
        # TODO: add messagge..
        raise Exception

    cmaps_dict, angles = build_cmaps(ens_ref_traj, data_builder, args)

    with open(cmap_file, "w") as f:
        write_cmap_tables(f, cmaps_dict, angles)


def old_clean_pdb(input: io.TextIOBase, output: io.TextIOBase):
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


def clean_pdb(input: io.TextIOBase, output: io.TextIOBase):
    subprocess.run(
        "cut -c -79 | pdb_tidy | pdb_element | pdb_chain -A",
        shell=True,
        stdin=input,
        stdout=output,
    )


def clean_pdb2(input):
    return subprocess.run(
        "cut -c -79 | pdb_tidy | pdb_element | pdb_chain -A",
        shell=True,
        input=input,
        capture_output=True,
    ).stdout


def build_parser():
    parser = argparse.ArgumentParser(description="What the program does")

    parser.add_argument("--topo-pdb", type=str, required=True)
    parser.add_argument("--ref-data", type=pathlib.Path, required=True)
    parser.add_argument("--aster-ens-confs", nargs="+", type=pathlib.Path)

    parser.add_argument("--skip-ref", action="store_true")
    parser.add_argument("--skip-cmap", action="store_true")

    cmap_group = parser.add_argument_group("CMAP")
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

    phys_params_group = parser.add_argument_group("Physical parameters")
    phys_params_group.add_argument(
        "-ld", "--langevin-damp", nargs="?", type=float, default=100.0
    )
    phys_params_group.add_argument("--conc", nargs="?", type=float, default=0.15)
    # phys_params_group.add_argument("--CO-charges", nargs="?", type=float, default=30.0)
    phys_params_group.add_argument("--dielectric", nargs="?", type=float, default=78.5)
    phys_params_group.add_argument("-T", "--temp", nargs="?", type=float, default=300.0)

    sim_params_group = parser.add_argument_group("Simulation parameters")
    sim_params_group.add_argument("--dry-run", action="store_true")
    sim_params_group.add_argument(
        "--sim-time", nargs="?", type=float, default=300.0, help="in nanoseconds"
    )
    sim_params_group.add_argument(
        "-ts", "--time-step", nargs="?", type=float, default=2.0, help="in femtoseconds"
    )
    sim_params_group.add_argument(
        "--box-half-width",
        nargs="?",
        type=int,
        default=200,
        help="in nanometers (check..)",
    )

    return parser


def run_sim(cwd, args):
    subprocess.run(
        [
            (args.lmp_path if args.lmp_path is not None else "lmp"),
            "-sf",
            "gpu",
            "-in",
            "in.CG",
        ],
        stdout=subprocess.DEVNULL,
        check=True,
        cwd=cwd,
    )


def write_ens_xtc(path, cg_atom_ids, ens_pdb_files):
    xyz_list = []

    for pdb_file in ens_pdb_files:
        with (
            pdb_file.open() as raw_pdb,
            tempfile.NamedTemporaryFile("w+") as cleaned_pdb,
        ):
            clean_pdb(raw_pdb, cleaned_pdb)
            cleaned_pdb.seek(0)
            pdb_traj = PDBTrajectoryFile(cleaned_pdb.name, standard_names=False)

        topo = pdb_traj.topology
        # mdtraj_rename = {"H": "HN", "HA3": "HA1"}

        mdtraj_id_to_atom = {
            (atom.residue.resSeq, atom.name): atom for atom in topo.atoms
        }

        inds = [mdtraj_id_to_atom[atom_id].index for atom_id in cg_atom_ids]

        xyz_list.append(pdb_traj.positions[0, inds])

    xyz = np.stack(xyz_list, axis=0)
    xyz = np.asarray(xyz, dtype=np.float32)

    xtc_traj = XTCTrajectoryFile(str(path), "w")
    xtc_traj.write(xyz)
    xtc_traj.close()


def run_ref_sim(init_ag_map, data_builder, args):  #
    for init_conform_name, init_ag in init_ag_map.items():
        conform_subdir = args.output_dir / init_conform_name
        conform_subdir.mkdir(exist_ok=True)

        ref_subdir = conform_subdir / "ref"
        ref_subdir.mkdir(exist_ok=True)

        write_configs(ref_subdir, init_ag, data_builder, args, None)
        if not args.dry_run:
            run_sim(ref_subdir, args)


def run_cmap_sim(init_ag_map, data_builder, args, ens_traj_file, cg_pdb_file):
    for init_conform_name, init_ag in init_ag_map.items():
        conform_subdir = args.output_dir / init_conform_name
        assert conform_subdir.exists()

        ref_subdir = conform_subdir / "ref"
        assert ref_subdir.exists()

        cmap_subdir = conform_subdir / "cmap"
        cmap_subdir.mkdir(exist_ok=True)

        ref_traj_file = ref_subdir / "traj.xtc"
        assert ref_traj_file.exists()
        ref_traj = mdt.load(ref_traj_file, top=cg_pdb_file)
        ens_traj = mdt.load(ens_traj_file, top=cg_pdb_file)

        write_configs(cmap_subdir, init_ag, data_builder, args, (ens_traj, ref_traj))
        if not args.dry_run:
            run_sim(cmap_subdir, args)


def main():
    parser = build_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True)

    u_ref = mda.Universe(args.topo_pdb)

    data_builder = CharmmDataBuilder(pathlib.Path(args.ref_data), u_ref.atoms)

    cg_atoms = data_builder.filter_cg_atoms(u_ref.atoms)
    cg_pdb_file = args.output_dir / "cg.pdb"
    cg_atoms.write(cg_pdb_file)

    init_ag_map = {}

    for init_conform in args.init_confs:
        with (
            init_conform.open() as raw_pdb,
            tempfile.NamedTemporaryFile("w+") as cleaned_pdb,
        ):
            clean_pdb(raw_pdb, cleaned_pdb)
            cleaned_pdb.seek(0)
            u_init = mda.Universe(cleaned_pdb.name, format="PDB")

        init_ag = data_builder.filter_cg_atoms(u_init.atoms)
        init_conform_name = init_conform.stem
        init_ag_map[init_conform_name] = init_ag

    if not args.skip_ref:
        run_ref_sim(init_ag_map, data_builder, args)

    # TODO: option to combine all ref traj.xtc files into one?

    if not args.skip_cmap:
        ens_traj_file = args.output_dir / "ensemble.xtc"

        if args.aster_ens_confs is not None:
            write_ens_xtc(ens_traj_file, data_builder.cg_atom_ids, args.aster_ens_confs)
        else:
            assert ens_traj_file.exists()

        run_cmap_sim(init_ag_map, data_builder, args, ens_traj_file, cg_pdb_file)


if __name__ == "__main__":
    main()
