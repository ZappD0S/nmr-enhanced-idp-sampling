#!/usr/bin/env bash


in_dir=./
out_dir=../short_ensemble_200_1/
mkdir $out_dir

for i in $(seq 1 200);
do

in=${in_dir}${i}.pdb
out=${out_dir}${i}_short.pdb
rm -f $out

for j in $(seq 72 132);
do
    cat $in | grep "   $j   " >> $out
done

done
