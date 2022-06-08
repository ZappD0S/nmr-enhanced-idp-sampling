#!/bin/sh

/home/gzappavigna/lammps/tools/ch2lmp/charmm2lammps.pl

vmd -dispdev none -e mev_nt.pgn

# perl /home/gzappavigna/lammps/tools/ch2lmp/charmm2lammps.pl all36m_prot mev_nt_chm -border=2.0 -cmap=36 -water -ions
/home/gzappavigna/lammps/tools/ch2lmp/charmm2lammps.pl all36m_prot 123_chm -L=400