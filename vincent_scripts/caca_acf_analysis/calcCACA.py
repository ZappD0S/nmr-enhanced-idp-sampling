top = "CG.pdb"
traj = "nopbc.dcd"

from pickle import dump

import numpy as np
import pytraj as pt
from pytraj import vector as va

traj = pt.iterload(traj, top)

sequence = {res.index+1:res.name for res in traj.top.residues}

#read CACA vectors
ca_indices = pt.select_atoms('@CA', traj.top)
ca_pairs = np.array(list(zip(ca_indices, ca_indices[1:])))
data_vec = va.vector_mask(traj, ca_pairs, dtype='ndarray')

caca_res = sorted(list(sequence.keys()))[1::]

corrs = {el:pt.timecorr(data_vec[n], data_vec[n], tcorr=np.floor(len(data_vec[n])/4), norm=False) for n, el in enumerate(caca_res) }

with open('CACA_ACF.pkl', 'wb') as tf:
    dump(corrs, tf)