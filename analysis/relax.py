import sys
import numpy as np
import mdtraj as mdt
import pytraj as pt

from scipy.optimize import minimize, LinearConstraint, check_grad, approx_fprime

from rotacf import rotacf
from nmr_relax_rates import NMR_relaxation_rates
from scipy.signal import find_peaks
from scipy.special import logsumexp, softmax

from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter


def build_corrs_dict(traj):
    corrs = {}
    top = traj.topology
    bonds_set = {tuple(bond) for bond in top.bonds}

    for res in list(top.residues):
        if res.index == 0 or res.name == "PRO":
            continue

        print(res.resSeq)

        n_atom = next(atom for atom in res.atoms if atom.name == "N")
        h_atom = next(atom for atom in res.atoms if atom.name == "H")
        assert (n_atom, h_atom) in bonds_set or (h_atom, n_atom) in bonds_set

        n_coords = traj.xyz[:, n_atom.index]
        h_coords = traj.xyz[:, h_atom.index]

        nh_vec = h_coords - n_coords

        corr = pt.timecorr(nh_vec, nh_vec, tcorr=1e6, norm=True)
        err = np.ones_like(corr)

        # corr, err = rotacf(nh_vec)
        if not np.isfinite(corr).all():
            print(np.nonzero(~np.isfinite(corr)))
            raise Exception("Error!")

        corrs[res.resSeq] = (corr, err)

    # return corrs, traj.time[:-1]
    return corrs, traj.time - traj.time[0]


def split_params(params):
    # n = (params.size + 1) // 2
    n = params.size // 2

    As = np.asarray(params[:n])
    lambdas = np.asarray(params[n:])
    assert lambdas.size == n

    return As, lambdas


def sum_of_exps(logt, logAs, loglambdas):
    # lambdas = np.append(lambdas, 0.0)
    exp_terms = np.exp(logAs[None, :] - np.exp(loglambdas[None, :] + logt[:, None]))
    return exp_terms.sum(axis=1)


def soe_loss(params, logt, corr, err):
    logAs, loglambdas = split_params(params)

    w = err**-2
    return np.mean(w * (sum_of_exps(logt, logAs, loglambdas) - corr) ** 2)


def soe_jac(params, logt, corr, err):
    logAs, loglambdas = split_params(params)

    # exps = np.exp(logAs[None, :] - lambdas[None, :] * t[:, None])

    A_grads = np.exp(logAs[None, :] - np.exp(loglambdas[None, :] + logt[:, None]))
    lambda_grads = -np.exp(
        logAs[None, :]
        - np.exp(loglambdas[None, :] + logt[:, None])
        + loglambdas[None, :]
        + logt[:, None]
    )
    grads = np.concatenate([A_grads, lambda_grads], axis=1)

    f_t = sum_of_exps(logt, logAs, loglambdas)
    w = err**-2
    grad = 2 * grads * w[:, None] * (f_t[:, None] - corr[:, None])

    return grad.mean(axis=0)


def log_sum_of_exps(t, logAs, lambdas):
    return logsumexp(logAs[None, :] - lambdas[None, :] * t[:, None], axis=1)


def log_soe_loss(params, t, corr, err):
    logAs, lambdas = split_params(params)

    # the corr^2 comes from the error propagation on the log
    w = corr**2 * err**-2
    return np.mean(w * (log_sum_of_exps(t, logAs, lambdas) - np.log(corr)) ** 2)


def log_soe_jac(params, t, corr, err):
    logAs, lambdas = split_params(params)

    sm = softmax(logAs[None, :] - lambdas[None, :] * t[:, None], axis=1)

    logA_grads = sm
    lambda_grads = -t[:, None] * sm
    grads = np.concatenate([logA_grads, lambda_grads], axis=1)

    f_t = log_sum_of_exps(t, logAs, lambdas)
    w = corr**2 * err**-2
    grad = 2 * grads * w[:, None] * (f_t[:, None] - np.log(corr[:, None]))

    return grad.mean(axis=0)


def select_data(t, corr, err):
    assert t.size == corr.size == err.size
    mask = np.ones_like(t, dtype=bool)

    if np.any(corr < 0):
        ind = np.argmax(corr < 0)
        mask[ind:] = False

    # inds = find_peaks(-corr[mask], prominence=1e-3, width=1e2)[0]

    # if inds.size != 0:
    #     ind = inds[0]
    #     mask[:ind] = True

    # mask &= t > 1
    assert np.all(np.isfinite(err[mask]))

    return mask


def fit(t, corr, err, A0s, lambda0s):
    eps = np.finfo(float).eps
    # if scale is None:
    #     scale = t.max()
    logt = np.log(t + eps)
    scale = logt.max()
    # scaled_t = t / scale
    logt -= scale

    logA0s = np.log(A0s)
    loglambda0s = np.log(lambda0s) + scale

    x0 = np.concatenate([logA0s, loglambda0s])
    # x0 = np.concatenate([A0s, lambda0s * scale])

    # bounds = [(0, 1)] * A0s.size + [(0, None)] * lambda0s.size
    bounds = [(None, 0)] * A0s.size + [(None, None)] * lambda0s.size

    # first unconstrained fit
    res = minimize(
        # log_soe_loss,
        soe_loss,
        x0=x0,
        args=(logt, corr, err),
        bounds=bounds,
        # jac=log_soe_jac,
        jac=soe_jac,
    )

    logAs, loglambdas = split_params(res.x)
    # As, scaled_lambdas = split_params(res.x)
    print(res.success)
    # print(scale * 1e-3 / scaled_lambdas)

    # rescale coeffs so that they add up to 1
    # tot = As.sum()
    # As /= tot
    # assert np.isclose(As.sum(), 1.0)
    logAs -= logsumexp(logAs)
    assert np.isclose(np.exp(logAs).sum(), 1.0)

    x0 = np.concatenate([logAs, loglambdas])
    # x0 = np.concatenate([As, scaled_lambdas])

    # bounds = [(0, None)] * A0s.size + [(0, None)] * lambda0s.size
    # cons = (
    #     LinearConstraint(
    #         [[1] * A0s.size + [0] * lambda0s.size],
    #         lb=1,
    #         ub=1,
    #     ),
    # )

    bounds = [(None, 0)] * A0s.size + [(None, None)] * lambda0s.size
    # bounds = [(0, 1)] * A0s.size + [(0, None)] * lambda0s.size

    cons = {
        "type": "eq",
        "fun": lambda x: logsumexp(x[: A0s.size]),
        # "fun": lambda x: x[: A0s.size].sum() - 1.0,
        "jac": lambda x: np.concatenate(
            [softmax(x[: A0s.size]), np.zeros(lambda0s.size)]
            # [np.ones(A0s.size), np.zeros(lambda0s.size)]
        ),
    }

    # constrained fit
    res = minimize(
        # log_soe_loss,
        soe_loss,
        x0=x0,
        args=(logt, corr, err),
        bounds=bounds,
        constraints=[cons],
        # jac=log_soe_jac,
        jac=soe_jac,
    )
    print(res.success)

    logAs, loglambdas = split_params(res.x)
    As = np.exp(logAs)
    # As, scaled_lambdas = split_params(res.x)
    lambdas = np.exp(loglambdas - scale)

    return As, lambdas


def fit_r(t, corr, lambda0s):
    pracma = importr("pracma")

    lambda0s = np.asarray(lambda0s)

    # t_max = t.max()
    # minexp = -np.finfo(t.dtype).minexp * np.log(2)

    # if t_max > minexp:
    #     scale =  t_max / minexp
    #     print(scale)
    # else:
    #     scale = 1.0

    t_r = robjects.FloatVector(t)
    corr_r = robjects.FloatVector(corr)
    lambda0s_r = robjects.FloatVector(lambda0s)

    res = pracma.mexpfit(t_r, corr_r, lambda0s_r, const=False)

    As = np.array(res.rx2("a"))
    lambdas = np.array(res.rx2("b"))

    assert As.shape == lambdas.shape == lambda0s.shape

    return As, lambdas


def generate_init(n_exps):
    # A0s = 2.0 ** -(np.arange(n_exps - 1) + 1)
    # A0s = np.append(A0s, 1 - A0s.sum())
    A0s = 2.0 ** np.arange(n_exps)
    A0s /= A0s.sum()

    # lambda0s = np.ones(n_exps) * 1e-5
    lambda0s = 10.0 ** np.arange(4, 0, -1)

    assert A0s.size == lambda0s.size == n_exps
    return A0s, lambda0s


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
            A0s, lambda0s = generate_init(n_exps)
            As, lambdas = fit(sel_t, sel_corr, sel_err, A0s, lambda0s)
            taus = 1e-3 / lambdas
            print(taus)
            print(f"n_exps: {n_exps}")

            # TODO: if any pair of exps is close to each other repeat with less exponents
            # TODO: if any of the As is really small repeat with one less exp
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
