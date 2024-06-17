#!/bin/bash

n_runs=10

aster_ens=$(find ensemble/ -regex "^.*/[0-9]+a_132\.pdb$")
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
    --skip-ref \
    ensemble/6921a_132.pdb
    # --aster-ens-confs "${aster_ens[@]}" -- \
    # "${init_conform[@]}"

