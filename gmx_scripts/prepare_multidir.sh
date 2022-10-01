#!/bin/bash

set -eu

Rg=3.275
L=$(echo "4 * $Rg" | bc -l)

export GMX=gmx_mpi
export FF=amber99sb-ildn


build_system() {
    $GMX pdb2gmx -f "$1.pdb" -o "$1.gro" -water tip4p -ignh -ter -i -ff $FF

    sed -i.bak "/^#include/ s/\(\".*tip4p\.itp\"\)/\"opc.itp\"/g" topol.top

    $GMX editconf -f "$1.gro" -o "${1}_newbox.gro" -c -box "$L" "$L" "$L" -bt triclinic
    $GMX solvate -cp "${1}_newbox.gro" -cs tip4p.gro -o "${1}_solv.gro" -p topol.top

    touch ions.mdp

    $GMX grompp -f ions.mdp -c "${1}_solv.gro" -p topol.top -o ions.tpr -maxwarn 1
    printf "SOL\n" | $GMX genion -s ions.tpr -o "${1}_solv_ions.gro" -conc 0.15 -p topol.top -pname NA -nname CL -neutral

}

: > folders.txt

mapfile -t selected <<< "$(printf "%s\n" "$@" | shuf -n 96)"

# Loop over each file provided as an argument
for file in "${selected[@]}"; do

    # Check if the file does NOT exist
    if [ ! -e "$file" ]; then
        echo "File $file does not exist. Exiting."
        break
    fi

    # Get the stem of the filename (name without extension)
    stem="$(basename "$file" .pdb)"

    # Create a folder with the stem name
    mkdir -p "$stem"

    echo "$stem" >> folders.txt

    # Copy start conformations
    cp "$file" "$stem/${stem}_cg.pdb"

    # Copy mdps
    cp {minim,nvt,npt,md}.mdp "$stem"

    # copy OPC model
    cp opc.itp "$stem"

    # Change into the folder
    cd "$stem" || exit

    scwrl -i "${stem}_cg.pdb" -o "$stem.pdb"

    build_system "$stem"

    # Go back to the previous directory
    cd ..
done
