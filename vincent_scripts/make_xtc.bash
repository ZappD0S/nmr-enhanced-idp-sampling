#!/usr/bin/env bash

cd ensemble_200_1/

for i in `ls ?.pdb ??.pdb ???.pdb`
do
	nb=`echo ${i} | sed 's/.pdb//g'`
	grep ' CA\| C \| N \| CB \| 2HA' ${nb}.pdb > ${nb}_CG.pdb
	gmx trjconv -f ${nb}_CG.pdb -o ${nb}.xtc
done

gmx trjcat -f `ls | grep .xtc` -o ensemble_200_1.xtc -cat
gmx trjconv -f ensemble_200_1.xtc -o ensemble_200_1.pdb

rm `ls | grep .xtc | grep -v ensemble`

cd ..
