#! /bin/bash

printf "SOLU\n" | gmx trjconv -f "$1.xtc" -s "$1.tpr" -n  -o "$1_whole.xtc" -pbc whole
printf "SOLU\n" | gmx trjconv -f "$1_whole.xtc" -s "$1.tpr" -n index.ndx -o "$1_nojump.xtc" -pbc nojump
printf "SOLU\n" | gmx trjconv -f "$1_nojump.xtc" -s "$1.gro" -n index.ndx -o "$1_nojump.gro" -dump 0
rm "$1_whole.xtc"
