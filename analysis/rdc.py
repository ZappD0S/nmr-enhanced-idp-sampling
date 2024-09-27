from pathlib import Path
import subprocess
import tempfile
from parse import compile, parse
import numpy as np
from scipy.constants import value, h, mu_0


PALES_PATH = "/home/gzappavigna/pales/pales"

gammas = {
    "N": -2.7126 * 1e7,
    "HN": value("proton gyromag. ratio"),
    "CA": 6.7262 * 1e7,
    "C": 6.7262 * 1e7,
    "HA": value("proton gyromag. ratio"),
    "H": value("proton gyromag. ratio"),
}

bond = {
    ("N", "HN"): 1.02 * 1e-10,
    ("N", "H"): 1.02 * 1e-10,
    ("CA", "HA"): 1.12 * 1e-10,
    ("CA", "C"): 1.52 * 1e-10,
}


# def unit_vector(vector, axis):
#     """Returns the unit vector of the vector."""
#     assert vector.shape[axis] == 3
#     return vector / np.linalg.norm(vector, axis=axis, keepdims=True)


# def angle_between(v1, v2):
#     """Returns the angle in radians between vectors 'v1' and 'v2'::

#     >>> angle_between((1, 0, 0), (0, 1, 0))
#     1.5707963267948966
#     >>> angle_between((1, 0, 0), (1, 0, 0))
#     0.0
#     >>> angle_between((1, 0, 0), (-1, 0, 0))
#     3.141592653589793
#     """
#     assert v1.shape[-1] == 3
#     assert v2.shape[-1] == 3

#     v1_u = unit_vector(v1, axis=-1)
#     v2_u = unit_vector(v2, axis=-1)
#     # return np.arccos(np.clip(np.dot(v1_u, v2_u.T), -1.0, 1.0))
#     return np.arccos(np.clip(np.sum(v1_u * v2_u, axis=-1), -1.0, 1.0))


# def projection(v1, v2):
#     """This function computes the vector that connects the the projection of v1 on v2 to v1.
#     Essentially it computes an orthogonal vector to v2 and is in the plane defined by v1 and v2
#     """
#     assert v1.shape[-1] == 3
#     assert v2.shape[-1] == 3

#     v1_u = unit_vector(v1)
#     v2_u = unit_vector(v2)
#     # return v1_u - np.dot(v1_u, v2_u.T) * v2_u
#     proj = v1_u - np.sum(v1_u * v2_u, axis=-1, keepdims=True) * v2_u
#     assert np.allclose(np.sum(proj * v2_u, axis=-1), 0.0)

#     return proj


def cart2sph(vecs):
    x, y, z = np.moveaxis(vecs, -1, 0)

    x_2 = x**2
    y_2 = y**2
    z_2 = z**2

    xy = np.sqrt(x_2 + y_2)

    r = np.sqrt(x_2 + y_2 + z_2)
    # in sph_harm, azimuthal angle must be in the [0, 2 pi] interval
    azim = np.arctan2(y, x) + np.pi
    assert np.all((0 <= azim) & (azim <= 2 * np.pi))
    polar = np.arctan2(xy, z)
    assert np.all((0 <= polar) & (polar <= np.pi))

    return r, azim, polar


def parse_pales_eig(lines):
    eigval_fmt = "DATA EIGENVALUES (Axx,Ayy,Azz)    {:2.4e} {:2.4e} {:2.4e}"
    eigvec_fmt = "DATA EIGENVECTORS {:1}AXIS {:2.4e} {:2.4e} {:2.4e}"

    eigvals_list = []
    eigvecs_dict = {}

    for line in lines:
        if (res := parse(eigval_fmt, line)) is not None:
            eigvals_list.append(np.array(list(res)))
        elif (res := parse(eigvec_fmt, line)) is not None:
            axis, *vec = list(res)
            eigvecs_dict[axis] = np.array(vec)

    assert len(eigvals_list) == 1
    assert eigvecs_dict.keys() == {"X", "Y", "Z"}

    eigvals = eigvals_list[0]
    eigvecs = np.column_stack([eigvecs_dict[axis] for axis in ["X", "Y", "Z"]])

    return eigvals, eigvecs


def compute_align_tensor(tempdir, pdbs):
    # TODO: use concurrent.futures.ThreadPoolExecutor

    eigvals_list = []
    eigvecs_list = []

    with tempfile.TemporaryDirectory(delete=False) as tempdir:
        tempdir = Path(tempdir)

        for pdb in pdbs:
            proc = subprocess.Popen(
                [PALES_PATH, "-pdb", str(pdb)],
                cwd=tempdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )

            output, _ = proc.communicate()
            assert proc.returncode == 0

            with open(tempdir / (pdb.stem + ".pales"), "w") as f:
                f.write(output)

            eigvals, eigvecs = parse_pales_eig(output.splitlines())
            eigvals_list.append(eigvals)
            eigvecs_list.append(eigvecs)

    return np.array(eigvals_list), np.array(eigvecs_list)


def compute_nh_indices(traj):
    top = traj.topology
    bonds_set = {tuple(bond) for bond in top.bonds}

    n_inds = []
    h_inds = []

    for res in list(top.residues):
        # if res.index == 0 or res.name == "PRO":
        if res.name == "PRO":
            continue

        n_atom = next(atom for atom in res.atoms if atom.name == "N")
        h_atom = next(atom for atom in res.atoms if atom.name == "H")
        assert (n_atom, h_atom) in bonds_set or (h_atom, n_atom) in bonds_set

        n_inds.append(n_atom.index)
        h_inds.append(h_atom.index)

    return np.array(n_inds), np.array(h_inds)


def compute_rdc(traj):
    # TODO: select frames
    subtraj = traj

    with tempfile.TemporaryDirectory() as tempdir_:
        tempdir = Path(tempdir_)

        pdbs = []

        for i, frame in enumerate(subtraj):
            pdb = tempdir / f"trj{i}.pdb"
            frame.save(pdb)
            assert pdb.exists()
            pdbs.append(pdb)

        eigvals, eigvecs = compute_align_tensor(tempdir, pdbs)

    n_inds, h_inds = compute_nh_indices(subtraj)

    n_coords = subtraj.xyz[:, n_inds]
    h_coords = subtraj.xyz[:, h_inds]
    nh_vecs = h_coords - n_coords

    nh_vecs = np.moveaxis(nh_vecs, 1, 0)

    # if we multiply a vector by the transpose of the matrix whose columns are the alignment tensor eigenvectors
    # we take a vector expressed in the laboratory frame to the basis formed by the alignment tensor eigenvectors
    # eigvecs.T @ nh_vec
    rot_nh_vecs = np.einsum("ijk,jkl->ijl", nh_vecs, eigvecs)

    _, phis, thetas = cart2sph(rot_nh_vecs)

    Axx, Ayy, Azz = np.moveaxis(eigvals, 1, 0)

    Aa = Azz
    Ar = 2 / 3 * (Axx - Ayy)

    Dmax = -(gammas["N"] * gammas["H"] * h * mu_0) / (
        8 * np.pi**3 * bond[("N", "H")] ** 3
    )

    rdc = np.mean(
        Dmax
        * (
            0.5 * Aa[None, ...] * (3 * np.cos(thetas) ** 2 - 1)
            + 0.75 * Ar[None, ...] * np.sin(thetas) ** 2 * np.cos(2 * phis)
        ),
        axis=1,
    )

    return rdc
