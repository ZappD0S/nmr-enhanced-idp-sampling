#!/bin/bash

nruns=5
ENS_DIR=/data/nsalvi/NT_HPS/ensemble2
nstructs=200
date=20210820B
script=make_input_4BR_$date\.py

for sim in $date
do



mkdir $sim
cd $sim


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

    cp ../../$script .

    python $script 10 #$sim #$eps 0.75 0.5

    #~/programs/lammps-mod/build/lmp_mpi -sf gpu -in in.CG

    cd ..
done



cd ..

done

