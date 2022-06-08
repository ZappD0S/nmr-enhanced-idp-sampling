#!/bin/bash

nruns=20
ENS_DIR=/data/vschnapka/202310-CMAP-HPS/MeV-NT/MeV_NT_ens/ensemble_200_1
nstructs=200
date=20231024
script=make_input_4BR_$date\.py

sim=5

for (( c=1; c<=$nruns; c++ ))
do
    mkdir $c
    cd $c

    num=$(( $RANDOM % $nstructs + 1 ))

    echo $num > pdbid
    grep ' CA\| C \| N \| CB \| 2HA' $ENS_DIR/$num.pdb > temp.pdb
    grep ATOM temp.pdb > temp2.pdb
    sed -n '/C   GLY/{h;n;G;p;d;};p' temp2.pdb > CG.pdb
    rm temp.pdb temp2.pdb

    cp ../$script .

    python $script $sim # frict

    ../lmp_mpi -sf gpu -in in.CG

    cd ..
done


