#!/bin/bash

set -eu

GMX=gmx_mpi

readarray -t folders < folders.txt

name=md

nojump_xtc=()

for folder in "${folders[@]}"; do

    cd "$folder" || exit

    printf "Protein\n" | $GMX trjconv -f "$name.xtc" -s "$name.tpr" -o "${name}_whole.xtc" -pbc whole
    printf "Protein\n" | $GMX trjconv -f "${name}_whole.xtc" -s "$name.tpr" -o "${name}_nojump.xtc" -pbc nojump

    nojump_xtc+=("$folder/${name}_nojump.xtc")

    cd ..
done

yes c | $GMX trjcat -f "${folders[@]/%//${name}_nojump.xtc}" -o final_nojump.xtc -settime
printf "Protein\n" | $GMX trjconv -f "${folders[0]}/${name}_nojump.xtc" -s "${folders[0]}/$name.tpr" -o final_nojump.pdb -dump 0

