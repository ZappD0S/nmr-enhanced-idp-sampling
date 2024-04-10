#!/usr/bin/bash

for f in `ls *.pdb`

do 
     ~/programs/pulchra304/bin/linux/pulchra $f

done

count=1

for f in `ls *.rebuilt.pdb`

do

	/home/nsalvi/programs/reduce.2.23.050314/reduce_src/reduce -build $f > $count.pdb
	let 'count=count+1' 
done
