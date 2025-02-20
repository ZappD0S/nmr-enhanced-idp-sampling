#!/bin/bash

set -eu


eval "$(conda shell.bash hook)"
conda activate lab


readarray -t folders < folders.txt

name=md

pdbs=("${folders[@]/%//${name}_nojump.pdb}")
xtcs=("${folders[@]/%//${name}_nojump.xtc}")

mapfile -t ref <<< "$(paste -d '\n' <(printf "%s\n" "${pdbs[@]}") <(printf "%s\n" "${xtcs[@]}"))"
mapfile -t ens <<< "$(find MeV_NT_ens -path "*/ensemble_200_*/*a_132.pdb")"

python ../python_scripts/add_cmap_to_gmx.py build -ens "${ens[@]}" -ref "${ref[@]}" -o cmaps.npy
