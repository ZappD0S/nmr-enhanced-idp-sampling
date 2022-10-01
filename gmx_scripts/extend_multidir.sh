#!/bin/bash

set -eu

export GMX=gmx_mpi


readarray -t folders < folders.txt

name=md


extend_gmx() {
    local span="$1" # in ps

    for folder in "${folders[@]}"; do
        cd "$folder" || exit

        $GMX convert-tpr -s "$name.tpr" -until "$span" -o "$name.tpr"

        cd ..
    done

    # mpirun -np "$(nproc)" $GMX mdrun -v -deffnm "$cur" -multidir "${folders[@]}"
    mpirun -np 96 --hostfile ~/my-hostfile $GMX mdrun -v -deffnm "$name" -multidir "${folders[@]}" -cpi
}

extend_gmx 10000