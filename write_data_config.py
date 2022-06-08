import io
import MDAnalysis as mda
from base_data_builder import BaseDataBuilder


def write_list(stream, header, data):
    stream.write(header + "\n")
    stream.write("\n")

    for row in data:
        stream.write(" ".join(map(str, row)) + "\n")

    stream.write("\n")


def write_data_config(
    stream: io.TextIOBase,
    data_build: BaseDataBuilder,
    init_ag: mda.AtomGroup,
    name: str,
    box_half_width: int,
    use_cmap: bool,
):
    stream.write(
        f"LAMMPS {name} input data\n"  #
        f"\n"
        f"{data_build.n_atoms} atoms\n"
        f"{data_build.n_bonds} bonds\n"
        f"{data_build.n_angles} angles\n"
        f"{data_build.n_dihedrals} dihedrals\n"
        f"0 impropers\n"
    )

    if use_cmap:
        stream.write(f"{data_build.n_crossterms} crossterms\n")

    stream.write(
        f"\n"
        f"{data_build.n_atom_types} atom types\n"
        f"{data_build.n_bond_types} bond types\n"
        f"{data_build.n_angle_types} angle types\n"
        f"{data_build.n_dihedral_types} dihedral types\n"
        f"\n"
        f"{-box_half_width} {box_half_width} xlo xhi\n"
        f"{-box_half_width} {box_half_width} ylo yhi\n"
        f"{-box_half_width} {box_half_width} zlo zhi\n"
        f"\n"
    )

    write_list(stream, "Masses", data_build.build_atom_type_masses())
    write_list(stream, "Pair Coeffs", data_build.build_pair_coeffs())
    write_list(stream, "Bond Coeffs", data_build.build_bond_coeffs())
    write_list(stream, "Angle Coeffs", data_build.build_angle_coeffs())
    write_list(stream, "Dihedral Coeffs", data_build.build_dihedral_coeffs())
    write_list(stream, "Atoms", data_build.build_atoms_list(init_ag))
    write_list(stream, "Bonds", data_build.build_bonds_list())
    write_list(stream, "Angles", data_build.build_angles_list())
    write_list(stream, "Dihedrals", data_build.build_dihedrals_list())

    if use_cmap:
        write_list(stream, "CMAP", data_build.build_cmap_crossterms_list())

    return data_build
