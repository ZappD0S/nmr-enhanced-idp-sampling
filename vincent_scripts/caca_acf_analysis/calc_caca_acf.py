#!/usr/bin/env python
# use python 3.6
from numpy import mean
from pytraj import load, select_atoms
from pytraj.analysis.vector import vector_mask
from pickle import dump
from argparse import ArgumentParser
import MD as mdv


def mypickle(filename, obj):
    with open(filename, 'wb') as out:
        dump(obj, out)


parser = ArgumentParser(description='calculate CACA autocorrelation functions of a protein for relaxation computation. Output is a dict()')
parser.add_argument('-f', type=str, help='trajectories directories .list or trajectory itself .xtc (default=traj.list)', default="traj.list")
parser.add_argument('-t', type=str, help='topology file (default=protein.pdb)', default="protein.pdb")
parser.add_argument('-n', type=int, help='number of residues. default=132', default=132)
parser.add_argument('-o', type=str, help='pickle output file (default=ave_caca_acf.pkl)', default="ave_caca_acf.pkl")

args = parser.parse_args()

output_file = str(args.o)
topol = str(args.t)
trajd = str(args.f)

if trajd[-1] == 't':
    traj_dirs = dh.Data(trajd).data
    traj_dirs = [i[0].rstrip("\n") for i in traj_dirs]
else:
    traj_dirs = [trajd]

traj = load(traj_dirs[0], topol)
sequence = dict([(res.index+1, res.name) for res in traj.top.residues])
res_list = sorted(sequence.keys())

print(len(sequence))

if trajd[-1] == 't':
    ave_acf = dict()
ml = 0
Ntraj = 0
for (t, f) in enumerate(traj_dirs):
    print(f)
    traj = load(f, topol)

    ca1_indices = select_atoms("@CA & !(:"+str(args.n)+")", traj.top)
    ca2_indices = select_atoms("@CA & !(:1)", traj.top)
    ca_pairs = list(zip(ca1_indices, ca2_indices))
    ca_vects = vector_mask(traj, ca_pairs, dtype="ndarray")
    #ca_vects -> [res][time][xyz]

    acf = dict()
    # acf -> [res][time]
    ca_res = [el for el in res_list][1:]
    for n in range(len(ca_vects)):
        mat = ca_vects[n] 
        #mat [time][xyz]
        acf[ca_res[n]] = mdv.rotacf(mat)
    
    trajlen = len(acf[list(acf.keys())[0]])
    if ml == 0 or trajlen < ml:
        ml = trajlen

    Ntraj += 1

    if trajd[-1] == 't':
        mypickle("traj_"+str(t)+"_caca_acf.pkl", acf)
    else:
        mypickle(output_file, acf)
    
    if trajd[-1] == 't':
        for res in acf.keys():
            if res not in ave_acf.keys():
                ave_acf[res] = acf[res]
            else:
                if len(ave_acf[res]) > len(acf[res]):
                    ave_acf[res] = [el + acf[res][i] for i, el in enumerate(ave_acf[res][0:len(acf[res])])]
                else:
                    ave_acf[res] = [el + acf[res][i] for i, el in enumerate(ave_acf[res])]

if trajd[-1] == 't':
    for res in ave_acf.keys():
        ave_acf[res] = [el/Ntraj for el in ave_acf[res]]
        ave_acf[res] = ave_acf[res][0:ml]
    # ave_acf -> [res][time]
    mypickle(output_file, ave_acf)

