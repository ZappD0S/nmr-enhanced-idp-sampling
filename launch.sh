#!/bin/sh
# shellcheck disable=SC1143

n_runs=11

# find ensemble/ -regextype posix-extended -regex "^.*/[0-9]+\.pdb$" \
find ensemble/ -regextype posix-extended -regex "^.*/[0-9]+a_132\.pdb$" \
    | shuf -n $n_runs \
    | xargs -t python make_input_conf.py  \
        --output-dir runs/ \
        --n-tasks 1 \
        --lmp-path /home/gzappavigna/lammps/build/lmp_custom \
        --topo-pdb CG.pdb \
        --dry-run
        # --use-cmap \
        # --ens-traj /data/vschnapka/202310-CMAP-HPS/MeV-NT/MeV_NT_ens/ensemble_200_1/ensemble_200_1.xtc \
        # --target-dist /data/vschnapka/202310-CMAP-HPS/reference_dih/all-rama-ref.out \
