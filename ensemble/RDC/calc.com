#!/bin/bash

residues=132

source /usr/local/gromacs/bin/GMXRC

count=0

   for (( res=1; res<=$residues; res++ )); do

       ls ../?.pdb ../??.pdb ../???.pdb | parallel -j128 ./run1.com {} $count $res

   done

   sem --wait



