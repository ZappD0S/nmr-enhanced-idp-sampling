#!/usr/bin/python
import numpy as np
import math
import re
import sys

temp=5.0

a31 = []
a31.append(["A","Ala"])
a31.append(["C","Cys"])
a31.append(["D","Asp"])
a31.append(["E","Glu"])
a31.append(["F","Phe"])
a31.append(["G","Gly"])
a31.append(["H","His"])
a31.append(["I","Ile"])
a31.append(["K","Lys"])
a31.append(["L","Leu"])
a31.append(["M","Met"])
a31.append(["N","Asn"])
a31.append(["P","Pro"])
a31.append(["Q","Gln"])
a31.append(["R","Arg"])
a31.append(["S","Ser"])
a31.append(["T","Thr"])
a31.append(["V","Val"])
a31.append(["W","Trp"])
a31.append(["Y","Tyr"])
def cnv1(a3):
    for i in range(len(a31)):
	if (a3 == a31[i][1]):
	    return a31[i][0]
    return "?"
def cnv3(a1):
    for i in range(len(a31)):
	if (a1 == a31[i][0]):
	    return a31[i][1]
    return "???"

fileIn=open('ntail.seq','r')
buf=fileIn.readline()
first=int(buf)
buf=fileIn.readline()
data=[]
for i in range(len(buf)):
    found=0
    for j in range(len(a31)):
	if (buf[i] == a31[j][0]):
	    found=1
	    break
    if (found==0):
	continue
    data.append([i+first,j,0.0,0.0,0.0,0.0,0.0,0.0])
print "Sequence"
for i in range(len(data)):
    print data[i][0],a31[data[i][1]][0],a31[data[i][1]][1]
fileIn.close()
data_ca=[]
fileIn=open('ca.info','r')
for i in range (20):
    fileIn.seek(0)
    found=0
    for line in fileIn:
	ls=line.split()
	if ls[0] == a31[i][0]:
	    found=1
	    break
    if (found==0):
	print "Error in ca.info"
	exit(1)
    data_ca.append([float(ls[1]),float(ls[2]),float(ls[3]),float(ls[4]),float(ls[5]),float(ls[6])])
fileIn.close()

print "CA table"
for i in range (20):
    print a31[i][0], data_ca[i]

data_co=[]
fileIn=open('co.info','r')
for i in range (20):
    fileIn.seek(0)
    found=0
    for line in fileIn:
	ls=line.split()
	if ls[0] == a31[i][0]:
	    found=1
	    break
    if (found==0):
	print "Error in co.info"
	exit(1)
    data_co.append([float(ls[1]),float(ls[2]),float(ls[3]),float(ls[4]),float(ls[5]),float(ls[6])])
fileIn.close()

print "CO table"
for i in range (20):
    print a31[i][0], data_co[i]

print "Temperature",temp

fileOut=open('ntail.ca','w')
for i in range(len(data)):
    cs_base = data_ca[data[i][1]][0]
    temp_corr = data_ca[data[i][1]][1]*0.001*(temp-25.)
    a=0.0;
    b=0.0;
    c=0.0;
    d=0.0;
    if (i<len(data)-2):
	a=data_ca[data[i+2][1]][2]
    if (i<len(data)-1):
	b=data_ca[data[i+1][1]][3]
    if (i>0):
	c=data_ca[data[i-1][1]][4]
    if (i>1):
	d=data_ca[data[i-2][1]][5]
    print str(data[i][0])+a31[data[i][1]][1],a31[data[i][1]][0],"base RCS %.3f" %cs_base,"seq cor %.3f" %(a+b+c+d),"temp cor %.3f" %temp_corr
    print >>fileOut,str(data[i][0])+a31[data[i][1]][1],cs_base+a+b+c+d+temp_corr
fileOut.close()
fileOut=open('ntail.co','w')
for i in range(len(data)):
    cs_base = data_co[data[i][1]][0]
    temp_corr = data_co[data[i][1]][1]*0.001*(temp-25.)
    a=0.0;
    b=0.0;
    c=0.0;
    d=0.0;
    if (i<len(data)-2):
	a=data_co[data[i+2][1]][2]
    if (i<len(data)-1):
	b=data_co[data[i+1][1]][3]
    if (i>0):
	c=data_co[data[i-1][1]][4]
    if (i>1):
	d=data_co[data[i-2][1]][5]
    print str(data[i][0])+a31[data[i][1]][1],a31[data[i][1]][0],"base RCS %.3f" %cs_base,"seq cor %.3f" %(a+b+c+d),"temp cor %.3f" %temp_corr
    print >>fileOut,str(data[i][0])+a31[data[i][1]][1],cs_base+a+b+c+d+temp_corr
fileOut.close()
