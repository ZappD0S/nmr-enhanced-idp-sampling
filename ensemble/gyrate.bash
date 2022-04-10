#!/usr/bin/env bash

rm rg_ensemble.txt
touch rg_ensemble.txt

for i in `seq 1 200`
do
	echo 1 | gmx gyrate -f ${i}.pdb -s ${i}.pdb -o ${i}_rg.xvg
	echo `tail -n 1 ${i}_rg.xvg | awk '{print $2}'`
	echo `tail -n 1 ${i}_rg.xvg | awk '{print $2}'` >> rg_ensemble.txt
done
