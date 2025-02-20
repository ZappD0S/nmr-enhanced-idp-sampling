#!/bin/bash

set -eu

# TODO: we need to source the correct GMXRC!

eval "$(conda shell.bash hook)"
conda activate lab

cur_dir=$(pwd)

cd "$1"

readarray -t folders < "folders.txt"

files=$(
    find "${folders[@]}" -type f \( \
        -name "*.gro" -o \
        -name "*.pdb" -o \
        -name "*.mdp" -o \
        -name "*.top" -o \
        -name "*.itp" \
    \) ! -name "mdout.mdp"
)


echo "$files" | xargs -d '\n' cp --parents -t "$cur_dir"
cp -t "$cur_dir" cmaps.npy

cd "$cur_dir"

# TODO: put this loop in a fuction and use xargs to make it run in parallel

for folder in "${folders[@]}"; do
    cd "$folder" || exit

    if [[ ! -f "topol.top" && -f "topol.top.bak" ]]; then
        rm topol.top
        mv topol.top.bak topol.top
    fi

    python ~/traj/python_scripts/add_cmap_to_gmx.py inject -top topol.top -xyz "${folder}_solv_ions.gro" -cmap ../cmaps.npy -o topol_cmap.top

    mv topol.top topol.top.bak
    ln -s topol_cmap.top topol.top

    cd ..
done
