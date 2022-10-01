#!/bin/bash

set -eu

export GMX=gmx_mpi


readarray -t folders < folders.txt

run_gmx() {

    local cur="$1"
    local prev=${2:-""}

    local default_params=(-f "$cur.mdp" -o "$cur.tpr" -p topol.top)

    for folder in "${folders[@]}"; do
        cd "$folder" || exit

        local params=("${default_params[@]}" -c "${prev:-${folder}_solv_ions}.gro")

        if [ "$POSRE" == "true" ]; then
            params+=(-r "${prev:-${1}_solv_ions}.gro")
        fi

        $GMX grompp "${params[@]}"

        cd ..
    done

    # mpirun -np "$(nproc)" $GMX mdrun -v -deffnm "$cur" -multidir "${folders[@]}"
    mpirun -np 96 --hostfile ~/my-hostfile $GMX mdrun -v -deffnm "$cur" -multidir "${folders[@]}" -cpi

}

POSRE=false run_gmx minim
POSRE=true run_gmx nvt minim
POSRE=true run_gmx npt nvt
POSRE=false run_gmx md npt