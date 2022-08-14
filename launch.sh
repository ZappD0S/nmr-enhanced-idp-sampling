#!/bin/bash

n_runs=10

aster_ens=$(find /data/vschnapka/202310-CMAP-HPS/MeV-NT/MeV_NT_ens/ensemble_200_*/ -regex "^.*/[0-9]+a_132\.pdb$")
# init_conform=$(echo "$aster_ens" | shuf -n $n_runs | tr "\n" " ")

readarray -t init_conform <<< "$(echo "$aster_ens" | shuf -n $n_runs)"

# printf '%s\n' "${init_conform[@]}"

readarray -t aster_ens <<< "$aster_ens"

# printf '%s\n' "${aster_ens[@]}"

# aster_ens=$(echo "$aster_ens" | tr "\n" " ")

# find ensemble/ -regextype posix-extended -regex "^.*/[0-9]+\.pdb$" \
# find ensemble/ -regextype posix-extended -regex "^.*/[0-9]+a_132\.pdb$" \
# find ensemble/ -regex "^.*/[0-9]+a_132\.pdb$" \
#     | shuf -n $n_runs \
#     | xargs -t

python make_input_conf.py  \
    --output-dir runs/ \
    --n-tasks 1 \
    --lmp-path /home/gzappavigna/lammps/build/lmp_custom \
    --ref-data chm2lmp_test/333_clean2.data \
    --topo-pdb chm2lmp_test/333_clean2.pdb \
    --dry-run \
    ensemble/6921a_132.pdb
    # --skip-ref \
    # --aster-ens-confs "${aster_ens[@]}" -- \
    # "${init_conform[@]}"
    # --skip-cmap \



# mpirun -n 1 /home/gzappavigna/lammps/build/lmp_custom -k on g 1 t 8 -pk kokkos newton on neigh half binsize 2.8 -sf kk -in