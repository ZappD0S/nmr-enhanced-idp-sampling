from pathlib import Path
import subprocess
import tempfile
import pandas as pd
from parse import compile, parse
import numpy as np
from scipy.constants import value, h, mu_0
import concurrent.futures


PALES_PATH = "/home/gzappavigna/software/pales/pales"

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


eigval_fmt = compile("DATA EIGENVALUES (Axx,Ayy,Azz)   {:2.4e} {:2.4e} {:2.4e}")
eigvec_fmt = compile("DATA EIGENVECTORS {:1}AXIS {:2.4e} {:2.4e} {:2.4e}")


def parse_pales_eig(lines):

    eigvals_list = []
    eigvecs_dict = {}

    for line in lines:
        if (res := eigval_fmt.parse(line)) is not None:
            eigvals_list.append(np.array(list(res)))
        elif (res := eigvec_fmt.parse(line)) is not None:
            axis, *vec = list(res)
            eigvecs_dict[axis] = np.array(vec)

    assert len(eigvals_list) == 1
    assert eigvecs_dict.keys() == {"X", "Y", "Z"}

    eigvals = eigvals_list[0]
    eigvecs = np.column_stack([eigvecs_dict[axis] for axis in ["X", "Y", "Z"]])

    return eigvals, eigvecs


def call_pales(tempdir, key, subframe):
    pdb = tempdir / ("frame" + "_".join(str(k) for k in key) + ".pdb")
    subframe.save(pdb)
    assert pdb.exists()

    out = subprocess.run(
        [PALES_PATH, "-pdb", str(pdb)],
        cwd=tempdir,
        text=True,
        check=True,
        capture_output=True,
    )

    pdb.unlink()

    return parse_pales_eig(out.stdout.splitlines())


def compute_align_tensors(subframes_dict):
    results_dict = {}

    with (
        concurrent.futures.ProcessPoolExecutor() as executor,
        tempfile.TemporaryDirectory() as tempdir_,
    ):
        tempdir = Path(tempdir_)
        futures_to_key = {}

        for key, subframe in subframes_dict.items():
            fut = executor.submit(call_pales, tempdir, key, subframe)
            futures_to_key[fut] = key

        for fut in concurrent.futures.as_completed(futures_to_key):
            eigvals, eigvecs = fut.result()
            key = futures_to_key[fut]
            results_dict[key] = (eigvals, eigvecs)

    return results_dict


def compute_nh_indices(top):
    bonds_set = {tuple(bond) for bond in top.bonds}

    resids = []
    n_inds = []
    h_inds = []

    for res in list(top.residues):
        if res.name == "PRO":
            continue

        n_atom = next(atom for atom in res.atoms if atom.name == "N")
        h_atom = next(atom for atom in res.atoms if atom.name == "H")
        assert (n_atom, h_atom) in bonds_set or (h_atom, n_atom) in bonds_set

        resids.append(res.index)
        n_inds.append(n_atom.index)
        h_inds.append(h_atom.index)

    return np.array(resids), np.array(n_inds), np.array(h_inds)


def compute_rdc(traj, win):

    assert win % 2 == 1
    half_win = win // 2
    n_frames = len(traj)
    n_resid = None

    subframes_dict = {}

    # TODO: this can be parallelized...
    for i, frame in enumerate(traj):
        if n_resid is None:
            n_resid = frame.n_residues
        else:
            assert frame.n_residues == n_resid

        for j in range(half_win, n_resid - half_win):
            inds = [
                atom.index
                for atom in frame.top.atoms
                if (
                    (j - half_win) <= atom.residue.index
                    and atom.residue.index <= (j + half_win)
                )
            ]
            subframe = frame.atom_slice(inds)
            assert subframe.n_atoms > 0
            subframes_dict[(i, j)] = subframe

    retults_dict = compute_align_tensors(subframes_dict)

    inds = np.array(list(retults_dict.keys()))

    data = list(retults_dict.values())
    eigvals_list = np.array([tup[0] for tup in data])
    eigvecs_list = np.array([tup[1] for tup in data])

    eigvals = np.full((n_frames, n_resid, 3), np.nan)
    eigvals[inds[:, 0], inds[:, 1]] = eigvals_list

    eigvecs = np.full((n_frames, n_resid, 3, 3), np.nan)
    eigvecs[inds[:, 0], inds[:, 1]] = eigvecs_list

    nh_resids, _, _ = compute_nh_indices(traj[0].top)
    mask = (half_win <= nh_resids) & (nh_resids < n_resid - half_win)

    eigvals = eigvals[:, nh_resids[mask]]
    eigvecs = eigvecs[:, nh_resids[mask]]

    assert not np.isnan(eigvals).any()
    assert not np.isnan(eigvecs).any()

    n_coords_list = []
    h_coords_list = []

    for frame in traj:
        nh_resids_, n_inds, h_inds = compute_nh_indices(frame.top)
        assert np.all(nh_resids == nh_resids_)
        n_coords_list.append(frame.xyz[:, n_inds[mask]])
        h_coords_list.append(frame.xyz[:, h_inds[mask]])

    n_coords = np.concatenate(n_coords_list, axis=0)
    h_coords = np.concatenate(h_coords_list, axis=0)

    nh_vecs = h_coords - n_coords

    # if we multiply a vector by the transpose of the matrix whose columns are the alignment tensor eigenvectors
    # we take a vector expressed in the laboratory frame to the basis formed by the alignment tensor eigenvectors
    # eigvecs.T @ nh_vec
    # rot_nh_vecs = np.einsum("ijk,jkl->ijl", nh_vecs, eigvecs)
    rot_nh_vecs = np.einsum("ijk,ijkl->ijl", nh_vecs, eigvecs)

    _, phis, thetas = cart2sph(rot_nh_vecs)

    Axx, Ayy, Azz = np.moveaxis(eigvals, 2, 0)

    Aa = Azz
    Ar = 2 / 3 * (Axx - Ayy)

    g15N = gammas["N"]
    g1H = gammas["H"]
    Dmax = (g15N * g1H * h * mu_0) / (8 * np.pi**3 * bond[("N", "H")] ** 3)

    rdc = Dmax * (
        0.5 * Aa * (3 * np.cos(thetas) ** 2 - 1)
        + 0.75 * Ar * np.sin(thetas) ** 2 * np.cos(2 * phis)
    )

    df = pd.DataFrame(data=rdc, index=range(len(traj)), columns=nh_resids[mask] + 1)
    df.columns.name = "resSeq"
    df.index.name = "frame"

    df = df.stack().to_frame("value")

    return df
