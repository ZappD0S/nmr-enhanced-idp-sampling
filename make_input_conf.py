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
from scipy import constants
import mdtraj as mdt
from scipy.stats import gaussian_kde
from KDEpy import FFTKDE


from write_data_config import write_data_config
from base_data_builder import BaseDataBuilder
from charmm_data_builder import CharmmDataBuilder


def write_cmap_tables(stream: io.TextIOBase, cmaps_dict, angles):
    for index, cmap in sorted(cmaps_dict.items(), key=lambda kv: kv[0]):
        stream.write(f"# type {index}\n\n")

        for angle, row in zip(angles, cmap):
            stream.write(f"# {angle}\n\n")
            stream.write(" ".join(map(str, row)) + " \n\n")

        stream.write("\n")


def write_input_config(
    stream: io.TextIOBase,
    # data_builder: BaseDataBuilder,
    args: argparse.Namespace,
    # ah_tables,
    # ah_table_fname: str,
    data_conf_fname: str,
    output_traj_fname: str,
    cmap_fname: str | None = None,
):
    # TODO: where does this value come from?
    # kappa = 3.04 / np.sqrt(args.concentration)
    # box_name = "mybox"
    cmap_name = "mycmap"
    Nsteps = floor(args.sim_time * 1e6 / args.time_step)

    stream.write(
        f"# comment\n"  #
        f"\n"
        f"units real\n"
        f"package gpu 0 neigh no\n"
        f"atom_style full\n"
        f"bond_style harmonic\n"
        f"angle_style charmm\n"
        f"dihedral_style charmm\n"
        f"improper_style harmonic\n"
        f"special_bonds charmm\n"
        f"\n"
        f"pair_style nm/cut/coul/long 12.0 15.0\n"
        # f"pair_style lj/charmm/coul/long 8.0 10.0\n"
        # f"pair_modify mix arithmetic\n"
        # f"suffix off\n"
        f"kspace_style pppm 1e-4\n"
        # f"suffix on\n"
    )

    stream.write("\n")

    # for entry, index1, index2 in ah_tables:
    #     stream.write(
    #         f"pair_coeff {index1} {index2}"  #
    #         f" table {ah_table_fname} {entry} {args.ah_cutoff}\n"
    #     )

    # stream.write("\n")

    # stream.write(
    #     f"pair_coeff * * coul/debye\n"
    #     f"dielectric {args.dielectric}\n"
    # )

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
        f"neighbor 2.0 bin\n"
        f"neigh_modify delay 5 every 1\n"
        f"\n"
        f"timestep {args.time_step}\n"
        f"thermo_style multi\n"
        f"thermo 1000\n"
        f"\n"
        f"minimize 1.0e-4 1.0e-6 10000 100000\n"
        f"fix 1 all nve\n"
        # f"fix 2 all shake 0.0001 5 0 m 1.0 a 232\n"
        # TODO: what is the random number? -> it's almost certainly the seed..
        f"fix 3 all langevin {args.temp} {args.temp} {args.langevin_damp} {randint(1, 100000)}\n"
        f"\n"
        f"dump 1 all xtc 250 {output_traj_fname}\n"
        f"run {Nsteps}\n"
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


def build_cmaps(ens_ref_traj, data_builder: BaseDataBuilder, args: argparse.Namespace):
    def make_grid(cmap_points):
        eps = 180 * np.finfo(float).eps
        angles = np.linspace(-180 - eps, 180 + eps, cmap_points + 1, endpoint=True)
        return np.stack(np.meshgrid(angles, angles, indexing="ij"), axis=-1), angles

    def check_inds(topo, phi_inds, psi_inds):
        resnames = [res.name for res in topo.residues][1:-1]

        for resname, phi_ind, psi_ind in zip(
            resnames, phi_inds[:-1], psi_inds[1:], strict=True
        ):
            assert all(phi_ind[1:] == psi_ind[:-1])

            atom1 = topo.atom(phi_ind[2])
            atom2 = topo.atom(psi_ind[1])

            assert atom1 == atom2
            assert atom1.name == "CA"

            assert atom1.residue.name == atom2.residue.name == resname

    def build_phipsi(traj):
        phi_inds, phi = mdt.compute_phi(traj)
        psi_inds, psi = mdt.compute_psi(traj)

        check_inds(traj.topology, phi_inds, psi_inds)

        zero_pad = np.zeros([phi.shape[0], 1])
        phi = np.concatenate([zero_pad, phi], axis=1)
        psi = np.concatenate([psi, zero_pad], axis=1)
        phipsi = np.stack([phi, psi], axis=-1).transpose([1, 0, 2])

        return phipsi

    R = constants.R / (constants.calorie * 1e3)

    ens_traj, ref_traj = ens_ref_traj

    ens_phipsi = build_phipsi(ens_traj)
    ref_phipsi = build_phipsi(ref_traj)

    cmaps_dict = {}

    grid, angles = make_grid(args.cmap_points)

    for key, resids in data_builder.crossterm_type_to_resids.items():
        sel_ens_phipsi = ens_phipsi[resids].reshape(-1, 2)

        scipy_kde = gaussian_kde(sel_ens_phipsi.T, bw_method="silverman")
        fft_kde = FFTKDE(bw=scipy_kde.silverman_factor())
        fft_kde.fit(sel_ens_phipsi)
        sel_ens_dens = fft_kde.evaluate(grid.reshape(-1, 2)).reshape(grid.shape[:2])
        sel_ens_dens = sel_ens_dens[:-1, :-1]

        sel_ref_phipsi = ref_phipsi[resids].reshape(-1, 2)

        scipy_kde = gaussian_kde(sel_ref_phipsi.T, bw_method="silverman")
        fft_kde = FFTKDE(bw=scipy_kde.silverman_factor())
        fft_kde.fit(sel_ref_phipsi)
        sel_ref_dens = fft_kde.evaluate(grid.reshape(-1, 2)).reshape(grid.shape[:2])
        sel_ref_dens = sel_ref_dens[:-1, :-1]

        cmap = -R * args.temp * (np.log(sel_ens_dens) - np.log(sel_ref_dens))
        ind = data_builder.crossterm_type_to_index[key]
        cmaps_dict[ind] = cmap

    return cmaps_dict, angles[:-1]


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


def build_parser():
    parser = argparse.ArgumentParser(description="What the program does")

    parser.add_argument("--topo-pdb", type=str, required=True)
    parser.add_argument("--ref-data", type=pathlib.Path, required=True)

    cmap_group = parser.add_argument_group("CMAP")
    # cmap_group.add_argument("--use-cmap", action="store_true")
    # cmap_group.add_argument("--target-dist", type=argparse.FileType("r"))
    cmap_group.add_argument(
        "--ens-traj",
        type=str,
        required=True,
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

    # ah_group = parser.add_argument_group("Ashbaugh-Hatch")
    # ah_group.add_argument("--ah-min-dist", nargs="?", type=float, default=0.5)
    # ah_group.add_argument("--ah-cutoff", nargs="?", type=float, default=30.0)
    # ah_group.add_argument(
    #     "--epsilon",
    #     nargs="?",
    #     type=float,
    #     default=0.2,
    #     help="energy scale of non-bonded interactions, in Kcal/mol.",
    # )
    # ah_group.add_argument("--ah-points", nargs="?", type=int, default=7501)

    phys_params_group = parser.add_argument_group("Physical parameters")
    phys_params_group.add_argument(
        "-ld", "--langevin-damp", nargs="?", type=float, default=100.0
    )
    phys_params_group.add_argument(
        "-conc", "--concentration", nargs="?", type=float, default=0.15
    )
    # phys_params_group.add_argument("--CO-charges", nargs="?", type=float, default=30.0)
    phys_params_group.add_argument("--dielectric", nargs="?", type=float, default=78.5)
    phys_params_group.add_argument("-T", "--temp", nargs="?", type=float, default=300.0)
    # phys_params_group.add_argument("--scaling", nargs="?", type=float, default=0.66)

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
            "mpirun",
            "-n",
            str(args.n_tasks),
            (args.lmp_path if args.lmp_path is not None else "lmp"),
            "-sf",
            "gpu",
            "-in",
            "in.CG",
        ],
        stdout=subprocess.DEVNULL,
        cwd=cwd,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True)

    u_ref = mda.Universe(args.topo_pdb)

    data_builder = CharmmDataBuilder(pathlib.Path(args.ref_data), u_ref.atoms)

    cg_atoms = data_builder.filter_cg_atoms(u_ref.atoms)
    cg_atoms.write(args.output_dir / "cg.pdb")

    init_ag_map = {}

    for init_conform in args.init_confs:
        with init_conform.open() as raw_pdb, tempfile.TemporaryFile("w+") as cleaned_pdb:
            clean_pdb(raw_pdb, cleaned_pdb)
            cleaned_pdb.seek(0)
            u_init = mda.Universe(cleaned_pdb, format="PDB")

        init_ag = data_builder.filter_cg_atoms(u_init.atoms)
        init_conform_name = init_conform.stem
        init_ag_map[init_conform_name] = init_ag

    ref_trajs_map = {}

    # run ref
    for init_conform_name, init_ag in init_ag_map.items():
        conform_subdir = args.output_dir / init_conform_name
        conform_subdir.mkdir(exist_ok=True)

        ref_subdir = conform_subdir / "ref"
        ref_subdir.mkdir(exist_ok=True)

        write_configs(ref_subdir, init_ag, data_builder, args, None)
        run_sim(ref_subdir, args)

        ref_trajs_map[init_conform_name] = ref_subdir / "traj.xtc"

    # TODO: option to combine all ref traj.xtc files into one?

    # run cmap
    for init_conform_name, init_ag in init_ag_map.items():
        conform_subdir = args.output_dir / init_conform_name
        assert conform_subdir.exists()

        ref_subdir = conform_subdir / "ref"
        assert conform_subdir.exists()

        cmap_subdir = conform_subdir / "cmap"
        cmap_subdir.mkdir(exist_ok=True)

        ref_traj_file = ref_trajs_map[init_conform_name]
        ref_traj = mdt.load(ref_traj_file, top=...)
        ens_traj = mdt.load(args.ens_traj, top=...)

        write_configs(cmap_subdir, init_ag, data_builder, args, (ens_traj, ref_traj))
        run_sim(cmap_subdir, args)


if __name__ == "__main__":
    main()
