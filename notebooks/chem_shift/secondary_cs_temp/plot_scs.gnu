set terminal postscript enhanced color solid 
#set nokey

set size 1,1
set xtics 10
set grid

set output 'scs.ps'
plot [] 'ntail_25c_ca.scs' using 1:2 wi imp lw 4 lt 1, 'ntail_5c_ca.scs' using 1:2 wi imp lw 1 lt 3
plot [] 'ntail_25c_co.scs' using 1:2 wi imp lw 4 lt 1, 'ntail_5c_co.scs' using 1:2 wi imp lw 1 lt 3


#wi lines, wi linespoints
#lt 1 2 3 4 (colors) 
