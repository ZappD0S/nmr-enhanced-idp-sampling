from itertools import product
from math import ceil
import numpy as np
import mdtraj as mdt
from KDEpy import FFTKDE
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import simpson
from scipy import constants
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def old_make_grid(resolution):
    step = 2 * np.pi / resolution
    grid_width = step * (resolution + 2)
    angles = np.linspace(
        -0.5 * grid_width, 0.5 * grid_width, resolution + 3, endpoint=True
    )
    return np.stack(np.meshgrid(angles, angles, indexing="ij"), axis=-1), angles


def make_grid(resolution, factor, frac):
    step_size = 2 * np.pi / (resolution * factor)

    grid_width = 2 * np.pi + 2 * frac * 2 * np.pi

    steps = ceil(grid_width / step_size)
    left_steps = (steps - resolution * factor) // 2

    angles = np.linspace(0, 1, steps + 1)
    angles -= angles[left_steps]
    angles /= angles[left_steps + resolution * factor]
    angles *= 2 * np.pi
    angles -= np.pi

    assert np.isclose(angles[left_steps], -np.pi)
    assert np.isclose(angles[left_steps + resolution * factor], np.pi)
    assert angles[0] <= -np.pi - frac * 2 * np.pi
    assert angles[-1] >= np.pi + frac * 2 * np.pi

    # inds = np.arange(left_points, left_points + resolution * factor + 1, factor)
    # inds = np.arange(left_steps, left_steps + resolution * factor, factor)
    slc = slice(left_steps, left_steps + resolution * factor + 1, factor)

    grid = np.stack(np.meshgrid(angles, angles, indexing="ij"), axis=-1)
    # return grid, angles, inds
    return grid, angles, slc


def check_inds(top, phi_inds, psi_inds):
    resnames = [res.name for res in top.residues][1:-1]

    for resname, phi_ind, psi_ind in zip(
        resnames, phi_inds[:-1], psi_inds[1:], strict=True
    ):
        assert all(phi_ind[1:] == psi_ind[:-1])

        atom1 = top.atom(phi_ind[2])
        atom2 = top.atom(psi_ind[1])

        assert atom1 == atom2
        assert atom1.name == "CA"

        assert atom1.residue.name == atom2.residue.name == resname


def build_phipsi(traj):
    phi_inds, phi = mdt.compute_phi(traj)
    psi_inds, psi = mdt.compute_psi(traj)

    check_inds(traj.topology, phi_inds, psi_inds)

    zero_pad = np.zeros([phi.shape[0], 1])
    phi = np.concatenate([zero_pad, phi], axis=1)
    phi = np.mod(phi + np.pi, 2 * np.pi) - np.pi

    psi = np.concatenate([psi, zero_pad], axis=1)
    psi = np.mod(psi + np.pi, 2 * np.pi) - np.pi

    phipsi = np.stack([phi, psi], axis=-1).transpose([1, 0, 2])

    return phipsi


def extend_phipsi(phipsi, frac):
    phi0, psi0 = np.moveaxis(phipsi, 1, 0)

    shifted_list = []

    for i, j in product([-1, 0, 1], repeat=2):
        mask = np.ones_like(phi0, dtype=bool)

        if i != 0:
            mask &= i * phi0 <= -np.pi + frac * 2 * np.pi

        if j != 0:
            mask &= j * psi0 <= -np.pi + frac * 2 * np.pi

        phi = phi0[mask] + 2 * i * np.pi
        psi = psi0[mask] + 2 * j * np.pi
        shifted_list.append(np.stack([phi, psi], axis=-1))

    # print(shifted_list[0].shape)
    phipsi = np.concatenate(shifted_list, axis=0)

    return phipsi

def find_bw(phipsi, id):
    fft_kde = FFTKDE(bw="silverman")
    n_vecs = 32

    angs = np.linspace(0, np.pi, n_vecs, endpoint=False)
    vecs = np.stack([np.cos(angs), np.sin(angs)], axis=0)
    proj = phipsi.dot(vecs)

    bws = []
    for i in range(n_vecs):
        bws.append(fft_kde.fit(proj[:, i]).bw)

    bws = np.array(bws)
    f = InterpolatedUnivariateSpline(angs, bws, k=4)
    cr_pts = f.derivative().roots()
    cr_vals = f(cr_pts)

    i_max = cr_vals.argmax()

    plt.plot(angs, bws, "bo")
    plt.plot(cr_pts[i_max], cr_vals[i_max], "ro")

    plt.savefig(f"plots/{id}_bw_curve.png")
    plt.clf()

    return cr_vals.max()


def compute_pdf(phipsi, grid, angles, slc, bw):
    fft_kde = FFTKDE(bw=bw)

    fft_kde.fit(phipsi)
    # pdf = fft_kde.evaluate(grid.reshape(-1, 2)).reshape(grid.shape[:2]).T
    pdf = fft_kde.evaluate(grid.reshape(-1, 2)).reshape(grid.shape[:2])

    pdf = pdf[slc.start:slc.stop, slc.start:slc.stop]

    angles = angles[slc.start:slc.stop]

    vol = simpson(simpson(pdf, x=angles), x=angles)
    print(vol)
    pdf /= vol

    pdf = pdf[:-1:slc.step, :-1:slc.step]
    print(pdf.shape)

    return pdf

# ref_bws = {}
# ens_bws = {}

# for ind, resids in ind_to_resids:
#     sel_ref_phipsi = ref_phipsi[np.asarray(resids) - 1].reshape(-1, 2)
#     ref_bw = find_bw(sel_ref_phipsi, "_".join(map(str, resids)) + "_ref")
#     ref_bws[ind] = ref_bw

#     sel_ens_phipsi = ens_phipsi[np.asarray(resids) - 1].reshape(-1, 2)
#     ens_bw = find_bw(sel_ens_phipsi, "_".join(map(str, resids)) + "_ens")
#     ens_bws[ind] = ens_bw

# ref_bws_list = list(ref_bws.values())
# plt.hist(ref_bws_list, density=True, label="ref")
# plt.vlines(np.mean(ref_bws_list), 0, 10, colors="r")
# plt.savefig("plots/bw/bw_ref.png")
# plt.clf()

# ens_bws_list = list(ens_bws.values())
# plt.hist(ens_bws_list, density=True, label="ens")
# plt.vlines(np.mean(ens_bws_list), 0, 10, colors="r")
# plt.savefig("plots/bw/bw_ens.png")
# plt.clf()
# print("fatto!")

def build_cmap(
    ens_phipsi,
    ref_phipsi,
    ens_bw,
    ref_bw,
    resolution: int = 24,
    eps: float = 1.1615857613434818e-09,
    temp: float = 300.0,
):
    R = constants.R / (constants.calorie * 1e3)

    factor = 100
    frac = 0.2

    ens_phipsi = extend_phipsi(ens_phipsi, frac)
    ref_phipsi = extend_phipsi(ref_phipsi, frac)

    # grid, angles, inds = make_grid(resolution, factor, frac)
    # ixgrid = np.ix_(inds, inds)
    grid, angles, slc = make_grid(resolution, factor, frac)

    # bw = find_bw(sel_ref_phipsi, "_".join(map(str, resids)) + "_ref")
    # print(bw)
    ref_pdf = compute_pdf(ref_phipsi, grid, angles, slc, ref_bw)
    # ref_pdf = ref_pdf[:-1:factor, :-1:factor]
    # ref_pdf = ref_pdf[ixgrid]
    # print(ref_pdf.shape)

    # plt.imshow(
    #     ref_dens.T, origin="lower", cmap="seismic", norm=colors.CenteredNorm()
    # )
    # plt.colorbar()
    # plt.savefig(f"plots/" + "_".join(map(str, resids)) + "_dens_ref.png")
    # plt.clf()

    # sel_ens_phipsi = ens_phipsi[np.asarray(resids) - 1].reshape(-1, 2)

    # plt.plot(sel_ens_phipsi[:, 0], sel_ens_phipsi[:, 1], ",")
    # plt.savefig(f"plots/" + "_".join(map(str, resids)) + "_scatter.png")
    # plt.clf()

    # bw = find_bw(sel_ens_phipsi, "_".join(map(str, resids)) + "_ens")
    # print(bw)
    ens_pdf = compute_pdf(ens_phipsi, grid, angles, slc, ens_bw)
    # ens_pdf = ens_pdf[:-1:factor, :-1:factor]
    # ens_pdf = ens_pdf[ixgrid]

    # plt.imshow(
    #     ens_dens.T, origin="lower", cmap="seismic", norm=colors.CenteredNorm()
    # )
    # plt.colorbar()
    # plt.savefig(f"plots/" + "_".join(map(str, resids)) + "_dens_ens.png")
    # plt.clf()

    print(np.log(eps + ens_pdf).min())
    print(np.log(eps + ref_pdf).min())
    # print(np.log(eps + ref_pdf).max())

    RT = R * temp
    cmap = -RT * (np.log(eps + ens_pdf) - np.log(eps + ref_pdf))

    print(cmap.min())

    # plt.imshow(cmap.T, origin="lower", cmap="seismic", norm=colors.CenteredNorm())
    # plt.colorbar()
    # plt.savefig(f"plots/" + "_".join(map(str, resids)) + "_cmap.png")
    # plt.clf()

    # cmaps_dict[ind] = cmap

    # plt.close()
    # return cmaps_dict, np.rint(angles[:-1:factor] * 180 / np.pi)
    return cmap, ref_pdf, ens_pdf, angles
