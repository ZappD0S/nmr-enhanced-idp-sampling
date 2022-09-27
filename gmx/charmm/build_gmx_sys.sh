#!/bin/sh

source /usr/local/gromacs/bin/GMXRC

gmx pdb2gmx -f 6921a_all.pdb -o 6921a_processed.gro -water tip4p -ignh -ter -i
gmx editconf -f 6921a_processed.gro -o 6921a_newbox.gro -c -d 1.0 -bt triclinic
# gmx solvate -cp 6921a_newbox.gro -cs spc216.gro -o 6921a_solv.gro -p topol.top
gmx solvate -cp 6921a_newbox.gro -cs tip4p.gro -o 6921a_solv.gro -p topol.top
touch ions.mdp
gmx grompp -f ions.mdp -c 6921a_solv.gro -p topol.top -o ions.tpr
# printf "SOL\n" | gmx genion -s ions.tpr -o 6921a_solv_ions.gro -conc 0.15 -p topol.top -pname SOD -nname CLA -neutral
printf "SOL\n" | gmx genion -s ions.tpr -o 6921a_solv_ions.gro -conc 0.15 -p topol.top -pname NA -nname CL -neutral
# gmx editconf -f 6921a_solv_ions.gro -o 6921a_solv_ions_centered.gro -center 0 0 0
