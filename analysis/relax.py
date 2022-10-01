import numpy as np
import pytraj as pt

from rotacf import rotacf
from nmr_relax_rates import NMR_relaxation_rates


from lmfit.models import ExponentialModel


def build_corrs_dict(traj):
    corrs = {}
    top = traj.topology
    bonds_set = {tuple(bond) for bond in top.bonds}

    for res in list(top.residues):
        if res.index == 0 or res.name == "PRO":
            continue

        n_atom = next(atom for atom in res.atoms if atom.name == "N")
        h_atom = next(atom for atom in res.atoms if atom.name == "H")
        assert (n_atom, h_atom) in bonds_set or (h_atom, n_atom) in bonds_set

        n_coords = traj.xyz[:, n_atom.index]
        h_coords = traj.xyz[:, h_atom.index]

        nh_vec = h_coords - n_coords

        corr = pt.timecorr(nh_vec, nh_vec, tcorr=1e5)
        # corr, _ = rotacf(nh_vec)
        # corr = pt.timecorr(nh_vec, nh_vec)

        if not np.isfinite(corr).all():
            print(np.nonzero(~np.isfinite(corr)))
            raise Exception("Error!")

        corrs[res.resSeq] = corr

    # return corrs, traj.time[:-1]
    return corrs, traj.time - traj.time[0]


def fit_lmfit(t, corr, err, A0s, tau0s):
    n = len(A0s)
    assert len(tau0s) == n
    assert n > 0

    model = ExponentialModel(prefix=f"e0_")
    model = sum([ExponentialModel(prefix=f"e{i}_") for i in range(1, n)], start=model)

    for i in range(n):
        model.set_param_hint(f"e{i}_amplitude", min=0)
        model.set_param_hint(f"e{i}_decay", min=0)

    params_dict = {f"e{i}_amplitude": A for i, A in enumerate(A0s)} | {
        f"e{i}_decay": tau for i, tau in enumerate(tau0s)
    }
    params = model.make_params(**params_dict)

    weights = err ** -2
    res = model.fit(corr, params, weights=weights, x=t)

    As = np.array([res.values[f"e{i}_amplitude"] for i in range(n)])
    taus = np.array([res.values[f"e{i}_decay"] for i in range(n)])

    return As, taus, res

