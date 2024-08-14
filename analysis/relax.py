import sys
import numpy as np
import mdtraj as mdt

import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, check_grad, approx_fprime
from scipy.interpolate import UnivariateSpline

from rotacf import rotacf
from nmr_relax_rates import NMR_relaxation_rates


def build_corrs_dict(traj):

    corrs = {}
    top = traj.topology

    for res in list(top.residues):
        if res.index == 0 or res.name == "PRO":
            continue

        n_atom = next(atom for atom in res.atoms if atom.name == "N")
        h_atom = next(atom for atom in res.atoms if atom.name == "H")

        n_coords = traj.xyz[:, n_atom.index]
        h_coords = traj.xyz[:, h_atom.index]

        nh_vec = h_coords - n_coords

        # corr = pt.timecorr(nh_vec, nh_vec, tcorr=1e4, norm=True)
        corr, err = rotacf(nh_vec)
        if not np.isfinite(corr).all():
            print(res.resSeq)
            print(np.nonzero(~np.isfinite(corr)))
            raise Exception("Error!")

        corrs[res.resSeq] = (corr, err)

    return corrs, traj.time[:-1]


def params_to_arr(params):
    n = params.size // 2

    As = np.asarray(params[:n])
    lambdas = np.asarray(params[n:])

    return As, lambdas


def sum_of_exps(t, As, lambdas):
    exp_terms = As[None, :] * np.exp(-lambdas[None, :] * t[:, None])
    return exp_terms.sum(axis=1)


def f(t, *params):
    As, lambdas = params_to_arr(params)
    return sum_of_exps(t, As, lambdas)


def loss(params, t, corr, err):
    As, lambdas = params_to_arr(params)

    return np.sum((sum_of_exps(t, As, lambdas) - corr) ** 2 / err**2)


def jac(params, t, corr, err):
    As, lambdas = params_to_arr(params)

    exps = np.exp(-lambdas[None, :] * t[:, None])

    A_grads = exps.copy()
    lambda_grads = -As[None, :] * t[:, None] * exps

    grads = np.concatenate([A_grads, lambda_grads], axis=1)

    f_t = sum_of_exps(t, As, lambdas)

    grad = 2 * grads * (f_t[:, None] - corr[:, None]) / err[:, None] ** 2

    return grad.sum(axis=0)


def select_data(t, corr, err):
    assert t.size == corr.size
    ind = np.argmax(corr < 0)
    print(ind)
    mask = (t > 1) & (t < t[ind]) & np.isfinite(err)

    return mask


def fit(t, corr, err, A0s, tau0s):
    lambda0s = 1e-3 / tau0s
    x0 = np.concatenate([A0s, lambda0s])

    # eps = np.sqrt(np.finfo(float).eps)

    # g1 = approx_fprime(x0, loss, eps, t, corr, err)
    # g2 = jac(x0, scaled_t, corr, err)
    # print(g1)
    # print(g2)
    # assert np.allclose(g1, g2)
    # grad_error = check_grad(loss, jac, x0, t, corr, err)
    # print("grad_error:", grad_error)
    # assert grad_error < eps

    bounds = (
        [(0, 1)] * A0s.size
        + [(0, None)] * lambda0s.size
    )

    # first unconstrained fit
    res = minimize(
        loss,
        x0=x0,
        args=(t, corr, err),
        bounds=bounds,
        # jac=jac,
    )

    As, lambdas = params_to_arr(res.x)

    # rescale coeffs so that they add up to 1
    tot = As.sum()
    As /= tot

    x0 = np.concatenate([As, lambdas])

    bounds = (
        [(0, None)] * A0s.size
        + [(0, None)] * lambda0s.size
    )
    cons = (
        LinearConstraint(
            [[1] * A0s.size + [0] * lambda0s.size],
            lb=1,
            ub=1,
        ),
    )

    # constrained fit
    res = minimize(
        loss,
        x0=x0,
        args=(t, corr, err),
        bounds=bounds,
        constraints=cons,
        # jac=jac,
    )
    print(res.success)

    As, lambdas = params_to_arr(res.x)
    taus = 1e-3 / lambdas

    return As, taus


def generate_init(n_exps):
    A0s = 2.0 ** -(np.arange(n_exps - 1) + 1)
    A0s = np.append(A0s, 1 - A0s.sum())

    tau0s = np.ones(n_exps)

    return A0s, tau0s


def fit_corrs(t, corrs):

    n0_exps = 4
    fit_params = {}

    for key, (corr, err) in corrs.items():
        mask = select_data(t, corr, err)

        sel_t = t[mask]
        sel_corr = corr[mask]
        sel_err = err[mask]
        print(key)

        n_exps = n0_exps
        while n_exps > 0:
            A0s, tau0s = generate_init(n_exps)
            As, taus = fit(sel_t, sel_corr, sel_err, A0s, tau0s)
            print(taus)
            print(f"n_exps: {n_exps}")

            # if any pair of exps is close to each other repeat with less exponents
            if taus.max() < 100 * sel_t[-1]:
                break

            print("bad fit, repeating")
            n_exps -= 1

        if n0_exps == 0:
            raise Exception

        fit_params[key] = (As, taus)
        print()

    return fit_params


# TODO: compute relax parameters from amps and taus and plot them
def compute_relax_rates(fit_params):
    pass


if __name__ == "__main__":
    xtc = sys.argv[1]
    pdb_or_gro = sys.argv[2]

    traj = mdt.load(xtc, top=pdb_or_gro)

    corrs_dict, t = build_corrs_dict(traj)

    fit_params = fit_corrs(t, corrs_dict)
