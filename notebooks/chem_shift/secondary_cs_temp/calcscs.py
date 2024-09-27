#!/usr/bin/python
import numpy as np
import math
import re
import sys

file1=open(sys.argv[1],"r")
file2=open(sys.argv[2],"r")
file3=open(sys.argv[3],"w")
for line1 in file1:
    ls1=line1.split()
    rnb1=int(re.findall("[0-9]+",ls1[3])[0])
    rnm1=re.findall("[A-Z][a-z][a-z]",ls1[3])[0]
    cs=float(ls1[0])
    
# Trying to find corresponding residue (number) in the second file
    found_2=1
    file2.seek(0)
    for line2 in file2:
	ls2=line2.split()
        rnb2=int(re.findall("[0-9]+",ls2[0])[0])
	rcs=float(ls2[1])
	if rnb2 == rnb1:
	    break
    else:
        found_2=0

# Calculate absolute difference, error, relative difference, error

    if found_2 == 1:
	print str(rnb1)+rnm1, cs-rcs, cs, rcs
	print >>file3, str(rnb1)+rnm1, cs-rcs, cs, rcs
file2.close()
file1.close()
file3.close()
