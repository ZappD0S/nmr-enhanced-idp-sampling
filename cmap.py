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

from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter


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

    pad = np.full([phi.shape[0], 1], np.nan)
    phi = np.concatenate([pad, phi], axis=1)
    phi = np.mod(phi + np.pi, 2 * np.pi) - np.pi

    psi = np.concatenate([psi, pad], axis=1)
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


def find_bw(phipsi):
    fft_kde = FFTKDE(bw="silverman")
    n_vecs = 64

    angs = np.linspace(-np.pi, np.pi, n_vecs, endpoint=False)
    vecs = np.stack([np.cos(angs), np.sin(angs)], axis=0)
    proj = phipsi.dot(vecs)

    bws = []
    for i in range(n_vecs):
        bws.append(fft_kde.fit(proj[:, i]).bw)

    bws = np.array(bws)
    f = InterpolatedUnivariateSpline(angs, bws, k=4)
    cr_pts = f.derivative().roots()
    cr_vals = f(cr_pts)

    # i_max = cr_vals.argmax()

    # plt.plot(angs, bws, "bo")
    # plt.plot(cr_pts[i_max], cr_vals[i_max], "ro")

    # plt.savefig(f"plots/{id}_bw_curve.png")
    # plt.clf()

    return cr_vals.max()
    # return cr_vals.min()


def compute_pdf(phipsi, grid, angles, slc, bw):
    fft_kde = FFTKDE(bw=bw)

    fft_kde.fit(phipsi)
    # pdf = fft_kde.evaluate(grid.reshape(-1, 2)).reshape(grid.shape[:2]).T
    pdf = fft_kde.evaluate(grid.reshape(-1, 2)).reshape(grid.shape[:2])

    pdf = pdf[slc.start : slc.stop, slc.start : slc.stop]

    angles = angles[slc.start : slc.stop]

    vol = simpson(simpson(pdf, x=angles), x=angles)
    # print(vol)
    pdf /= vol

    pdf = pdf[: -1 : slc.step, : -1 : slc.step]

    return pdf


def compute_pdf_r(phipsi, resolution, Hpi=None, factor=10):
    ks = importr("ks")

    numpy2ri.activate()

    if Hpi is None:
        Hpi = np.array(ks.Hpi(x=phipsi))

    xmin = np.array([-np.pi, -np.pi])
    xmax = np.array([np.pi, np.pi])

    # stride = ceil(151 / resolution)

    gridsize = np.array([resolution * factor + 1, resolution * factor + 1])

    k = ks.kde(x=phipsi, H=Hpi, gridsize=gridsize, xmin=xmin, xmax=xmax, density=True)

    pdf = np.array(k.rx2("estimate"))
    xangles = np.array(k.rx2("eval.points").rx2(1))
    yangles = np.array(k.rx2("eval.points").rx2(2))

    assert np.all(xangles == yangles)

    # vol = simpson(simpson(pdf, x=xangles), x=xangles)
    # print("vol:", vol)
    # pdf /= vol

    return pdf[:-1:factor, :-1:factor], xangles[:-1:factor], Hpi


def build_cmap(
    ens_phipsi,
    ref_phipsi,
    basepath,
    resolution: int = 24,
    Emin=-7.0,
    temp: float = 298.0,
):
    R = constants.R / (constants.calorie * 1e3)

    frac = 0.2

    ens_phipsi = extend_phipsi(ens_phipsi, frac)
    ref_phipsi = extend_phipsi(ref_phipsi, frac)

    # grid, angles, slc = make_grid(resolution, factor, frac)

    # if ref_bw is None:
    #     ref_bw = find_bw(ref_phipsi)

    # ref_pdf = compute_pdf(ref_phipsi, grid, angles, slc, ref_bw)
    ref_pdf, _, _ = compute_pdf_r(ref_phipsi, resolution)

    plt.imshow(ref_pdf.T, origin="lower", cmap="seismic", norm=colors.CenteredNorm())
    plt.colorbar()
    plt.savefig(basepath + "_dens_ref.png")
    plt.clf()

    plt.plot(ens_phipsi[:, 0], ens_phipsi[:, 1], ".")
    plt.savefig(basepath + "_scatter.png")
    plt.clf()

    # if ens_bw is None:
    #     ens_bw = find_bw(ens_phipsi)

    ens_pdf, angles, _ = compute_pdf_r(ens_phipsi, resolution)

    plt.imshow(ens_pdf.T, origin="lower", cmap="seismic", norm=colors.CenteredNorm())
    plt.colorbar()
    plt.savefig(basepath + "_dens_ens.png")
    plt.clf()

    RT = R * temp
    eps = np.exp(Emin / RT)
    cmap = -RT * (np.log(eps + ens_pdf) - np.log(eps + ref_pdf))
    # cmap = -RT * np.log((ens_pdf + eps) / (ref_pdf + eps))

    plt.imshow(cmap.T, origin="lower", cmap="seismic", norm=colors.CenteredNorm())
    plt.colorbar()
    plt.savefig(basepath + "_cmap.png")
    plt.clf()

    plt.close()
    # return cmaps_dict, np.rint(angles[:-1:factor] * 180 / np.pi)
    return cmap, ens_pdf, ref_pdf, angles
