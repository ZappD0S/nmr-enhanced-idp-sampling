#!/usr/bin/env bash

i=1

while [  $i -lt 401 ];
do
        ~/programs/SPARTA+/sparta+ -in $i\.pdb
        mv pred.tab pred_$i\.tab
        mv struct.tab struct_$i\.tab
        let "i=i+1"
done

