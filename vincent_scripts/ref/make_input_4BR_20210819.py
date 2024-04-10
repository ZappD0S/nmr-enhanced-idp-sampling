import sys

langdamp=float(sys.argv[1])

pdb = "CG.pdb"
conc = 0.5 #salt, M
CO_charges = 30.0 #cut-off Coulomb/Debye/GPU
diel = 78.5 # for water, dielectric constant 
NP = 7501 #number of points in non-bonded tables
CO_NB = 30.0 #cut-off non-bonded interactions, A
epsilon = 0.2 #Kcal/mol, energy scale of non-bonded interactions 
rminNB = 0.5 # A
rmaxNB = 30.0 # A
#langdamp = 200.0 # in fs
T = 298.0 # 
trajT = 500 # ns
tstep = 4.0 # fs
XTC = '/data/nsalvi/NT_HPS/ensemble2/ensemble.dcd'
PDB = '/data/nsalvi/NT_HPS/ensemble2/18.rebuilt.pdb'
REFfile = "/data/nsalvi/NT_HPS/test_4BR/all-rama-ref.out"
NCMAP = 24 #number of points to build correction maps (NCMAPxNCMAP)
scaling = 0.66 #scaling factor of ML vdW radii (this is 0.66 in FM)

#### NO CHANGES BELOW HERE #####

#VERSION HISTORY:
#20210819: 1st running version

import numpy as np
import Bio.SeqUtils
from datetime import date
from math import floor
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral, Ramachandran
from MDAnalysis.analysis import dihedrals
from astropy.convolution import convolve
from astropy.convolution.kernels import Gaussian1DKernel, Gaussian2DKernel
from random import randint
from scipy import ndimage
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")
sns.set_style("ticks", {"xtick.direction": "in","ytick.direction": "in", "xtick.major.size": 8, "ytick.major.size": 8, 'font.family': ['Liberation Sans']})

today = date.today()
d1 = today.strftime("%Y-%m-%d")

RT = 1.98720425864083*1e-3*T

#sigma values in nm (note: internal units are A, multiply by 10)
sigma_dict = {'A':0.504, 'R':0.656, 'N':0.568, 'D':0.558, 'C':0.548,
                'Q':0.602, 'E':0.592, 'G':0.450, 'H':0.608, 'I':0.618,
                        'L':0.618, 'K':0.636, 'M':0.618, 'F':0.636, 'P':0.556,
                                'S':0.518, 'T':0.562, 'W':0.678, 'Y':0.646, 'V':0.586}

#lambda values in nm (note: internal units are A, multiply by 10)
# original values
# lambda_dict = {'A':0.730, 'R':0.000, 'N':0.432, 'D':0.378, 'C':0.595,
#                 'Q':0.514, 'E':0.459, 'G':0.649, 'H':0.514, 'I':0.973,
#                         'L':0.973, 'K':0.514, 'M':0.838, 'F':1.000, 'P':1.000,
#                                 'S':0.595, 'T':0.676, 'W':0.946, 'Y':0.865, 'V':0.892}
# values from bob best
lambda_dict = {'A':0.51507, 'R':0.24025, 'N': 0.78447, 'D':0.30525, 'C': 0.46169,
                'Q': 0.29516, 'E':0.42621, 'G':1.24153, 'H':0.55537, 'I': 0.83907,
                        'L':0.51207, 'K':0.47106, 'M': 0.64648, 'F':1.17854, 'P':0.34128,
                                'S': 0.11195, 'T':0.27538, 'W':0.97588, 'Y': 1.04266, 'V': 0.55645}

#charges
charges_dict = {'A':0, 'R':1, 'N':0, 'D':-1, 'C':0,
                'Q':0, 'E':-1, 'G':0, 'H':0.5, 'I':0,
                        'L':0, 'K':1, 'M':0, 'F':0, 'P':0, 'S':0, 'T':0, 'W':0, 'Y':0, 'V':0}

#masses
#these are the masses of side chains
masses_dict = {'A':15.0347, 'R':100.1431, 'N':58.0597, 'D':59.0445, 'C':47.0947,
                'Q':72.0865, 'E':73.0713, 'G':1.0079, 'H':81.0969, 'I':57.1151,
                        'L':57.1151, 'K':72.1297, 'M':75.1483, 'F':91.1323, 'P':41.0725, 'S':31.0341, 'T':45.0609, 
                'W':130.1689, 'Y':107.1317, 'V':43.0883}

#side-chain r0 from M. Levitt J Mol Biol 1976 (derived from Clothia 1975)
#values in A
r0_dict = {'A':4.6, 'R':6.8, 'N':5.7, 'D':5.6, 'C':5.0,
                'Q':6.1, 'E':6.1, 'G':3.8, 'H':6.2, 'I':6.2,
                        'L':6.3, 'K':6.3, 'M':6.2, 'F':6.8, 'P':5.6,
                                'S':4.8, 'T':5.6, 'W':7.2, 'Y':6.9, 'V':5.8}

#side-chain epsilon from M. Levitt J Mol Biol 1976 (derived from number of heavy atoms)
#values in kcal/mol
epsilon_dict = {'A':0.05, 'R':0.39, 'N':0.21, 'D':0.21, 'C':0.10,
                'Q':0.27, 'E':0.27, 'G':0.025, 'H':0.33, 'I':0.21,
                        'L':0.21, 'K':0.27, 'M':0.21, 'F':0.39, 'P':0.16,
                                'S':0.10, 'T':0.16, 'W':0.56, 'Y':0.45, 'V':0.16}

kspring = 200 # in E/distance^2 units

#define non-bonded forces
def lj(r, epsilon, sigma):
    V = 4 * epsilon * ( (sigma / r)**12 - (sigma / r)**6);
    F = 4 * epsilon / r * ( 12 * (sigma / r)**12 - 6 * (sigma / r)**6);
    return (V, F)

def lj86(r, epsilon, sigma):
    V = epsilon * ( 3*(sigma / r)**8 - 4*(sigma / r)**6);
    F = 24* epsilon / r * ( (sigma / r)**8 - (sigma / r)**6);
    return (V, F)

def phinb(r, eps, sigma, lmbd):
    VLJ, FLJ = lj(r, eps, sigma)
    if r<2**(1/6)*sigma:
        V = VLJ+(1-lmbd)*eps
        F = FLJ
    else:
        V = lmbd*VLJ
        F = lmbd*FLJ
    return (V, F)

def rep8(r, epsilon, sigma):
    V = epsilon * ( 3*(sigma / r)**8);
    F = 24* epsilon / r * ( (sigma / r)**8 );
    return (V, F)

#equilibrium bond lengths in Ang
#values are taken from N.V. Bhagavan, Chung-Eun Ha, in Essentials of Medical Biochemistry (Second Edition), 2015
#this assumes that PDB atoms are sorted
#for each residue one has N, CA, CO, CB (or HA2 for gly)
r0 = dict()
#first, all N-CA bonds and CA-CB
for k in charges_dict.keys():
        r0["NN-"+k] = 1.455
        if k=='G':
            r0[k+"-CB"] = 1.09
        else:
            r0[k+"-CB"] = 1.53
        #now all CA-CO bonds
        for k in charges_dict.keys():
                r0[k+"-CO"] = 1.524
                #finally the CO-N bonds, which are assumed to be the same
                r0["CO-NN"] = 1.334
                #and general N-CA and CA-CO bond
                r0["NN-CA"] = 1.455
                r0["CA-CO"] = 1.524

#read CG coordinates
CG_coords = np.genfromtxt(pdb, usecols=[5, 6, 7])

#read sequence
sequence = np.recfromtxt(pdb, usecols=[3])

res_numbers = np.recfromtxt(pdb, usecols=[4])

#read atomtypes
atomtype = np.recfromtxt(pdb, usecols=[2])
atomtype = [el.decode() for el in atomtype]

#move protein to center
CG_coords = CG_coords-CG_coords.mean(axis=0)
natoms = len(CG_coords)

with open("data.CG", "w") as tf:
    tf.write("LAMMPS CG input data\n")
    tf.write("\n")
    tf.write(str(natoms)+" atoms\n")
    nbonds = natoms-1
    tf.write(str(nbonds)+" bonds\n")
    #this counts backbone dihs 
    ndih = 0
    nres = 0
    for n, el in enumerate(atomtype):
        if el=="CA" and n>2:
            ndih += 3
            nres += 1
    #ndih -=1
    nres += 1
    nangles = 5*nres-3
    tf.write(str(nangles)+" angles\n")
    tf.write(str(ndih)+" dihedrals\n")
    tf.write("0 impropers\n")
    ncmaps=nres-2
    tf.write(str(ncmaps)+" crossterms\n")
    tf.write("\n")
    nattypes = len(set(sequence))*2+2
    if "PRO" in [el.decode() for el in sequence]:
        nattypes += 1
    tf.write(str(nattypes)+" atom types\n")
    #for each AA type we have specific N-CA, CA-CB and CA-C bonds 
    #plus a generic C-N bond
    nbondtypes = 3*len(set(sequence))+1
    tf.write(str(nbondtypes)+" bond types\n")
    nangletypes = 5 #N-CA-CO, CA-CO-N, CO-N-CA, N-CA-CB, C-CA-CB 
    tf.write(str(nangletypes)+" angle types\n")
    
    ndihs=1+(nres-1)*2 #a generic omega + residue-specific phi and psi
    tf.write(str(ndihs)+" dihedral types\n")
    tf.write("\n")
    #create a LARGE box around the protein
    #command to create box actually in in.CG file
    #side = np.abs(CG_coords).max()*4
    #hs = side/2
    hs = 200
    tf.write(str(-hs)+" "+str(hs)+" xlo xhi\n")
    tf.write(str(-hs)+" "+str(hs)+" ylo yhi\n")
    tf.write(str(-hs)+" "+str(hs)+" zlo zhi\n")
    tf.write("\n")
    tf.write("Masses\n")
    tf.write("\n")
    for n, el in enumerate(set(sequence)):
        # first we write all CAs, which actually have the mass of CAHA
        tf.write(str(n+1)+" "+str(12.011 + 1.008)+"\n")
    # then all CBs, whose mass = mass of sidechain
    for n, el in enumerate(set(sequence)):
        type_letter = Bio.SeqUtils.IUPACData.protein_letters_3to1[el.decode().title()]
        tf.write(str(len(set(sequence))+n+1)+" "+str(masses_dict[type_letter])+"\n")
    tf.write(str(2*len(set(sequence))+1)+" 15.015\n") # this corresponds to the mass of N + H
    tf.write(str(2*len(set(sequence))+2)+" 28.010\n") # this corresponds to the mass of C + O
    if "PRO" in [el.decode() for el in sequence]:
        tf.write(str(2*len(set(sequence))+3)+" 14.007\n") # PRO does not have HN...
    tf.write("\n")
    
    
    tf.write("Bond Coeffs # harmonic\n")
    tf.write("\n")
    temp=1
    for n, el in enumerate(set(sequence)):
       type1 = Bio.SeqUtils.IUPACData.protein_letters_3to1[el.decode().title()]
       tf.write(str(temp)+" "+str(kspring)+" "+str(r0["NN-"+type1])+"\n")
       temp += 1
       tf.write(str(temp)+" "+str(kspring)+" "+str(r0[type1+"-CB"])+"\n")
       temp += 1
       tf.write(str(temp)+" "+str(kspring)+" "+str(r0[type1+"-CO"])+"\n")
       temp += 1
    tf.write(str(temp)+" "+str(kspring)+" "+str(r0["CO-NN"])+"\n")
    tf.write("\n")
    tf.write("Angle Coeffs # harmonic\n")
    tf.write("\n")
    tf.write("1 10.0 121.4\n")
    tf.write("2 10.0 109.0\n")
    tf.write("3 10.0 116.2\n")
    tf.write("4 10.0 110.5\n")
    tf.write("5 10.0 111.2\n")
    tf.write("\n")
    tf.write("Dihedral Coeffs # fourier\n")
    tf.write("\n")
    tf.write("1 2 6.10883 1 0.0 10.46 2 180.0\n")
#     #add dummy values for phi and psi (0 kcal/mol)
#     for n in range(2, ndihs+1):
#         tf.write(str(n)+" 2 0.0 1 0.0 0.0 2 180.0\n")
    #add values for phi and psi
    temp2=2
    for n in range(natoms):
        #adds psi
        if atomtype[n]=="N":
            if n+4 < natoms:
                tf.write(str(temp2)+" 1 0.6 1 0.0\n")
                temp2+=1
        #adds phi
        if atomtype[n]=="C":
            if n+4 < natoms:
                tf.write(str(temp2)+" 1 0.2 1 180.0\n")
                temp2+=1
    tf.write("\n")
    tf.write("Atoms # full\n")
    tf.write("\n")
    for n in range(natoms):
        #syntax: atom-ID molecule-ID atom-type x y z
        at = atomtype[n]
        row=str(n+1)+" 1 "
        #this adds N atoms
        if at=="N":
            if sequence[n].decode() == "PRO":
                row=row+str(2*len(set(sequence))+3)
            else:
                row=row+str(2*len(set(sequence))+1)
            if res_numbers[n]==1:
                row = row + " 1 " #charged N-terminus
            else:
                row = row + " 0 "
        #this adds C atoms
        if at=="C":
            row=row+str(2*len(set(sequence))+2)
            if res_numbers[n]==max(res_numbers):
                row = row+" -1 " #charged C-terminus
            else:
                row = row + " 0 "
        #this adds CA atoms
        if at=="CA":
            restype=sequence[n]
            m = list(set(sequence)).index(restype) + 1
            type1=Bio.SeqUtils.IUPACData.protein_letters_3to1[restype.decode().title()]
            row = row+str(m)+" 0 "
        #this adds CB/HA2 atoms
        if at=="CB" or at=="2HA":
            restype=sequence[n]
            m = list(set(sequence)).index(restype) + 1 + len(set(sequence))
            type1=Bio.SeqUtils.IUPACData.protein_letters_3to1[restype.decode().title()]
            row = row+str(m)+" "+str(charges_dict[type1])+" "
        x, y, z = CG_coords[n]
        row = row+str(x)+" "+str(y)+" "+str(z)+" 0 0 0\n"
        tf.write(row)
    tf.write("\n")
    tf.write("Bonds\n")
    tf.write("\n")
    for n in range(natoms-1):
        row=str(n+1)+" "
        at = atomtype[n]
        restype = sequence[n]
        pos = [n for n, el in enumerate(set(sequence)) if el==restype][0]
        if at=="N":
            bondtype=str(3*pos+1)
            row=row+bondtype+" "+str(n+1)+" "+str(n+2)+"\n"
        if at=="C":
            bondtype=str(nbondtypes)
            row=row+bondtype+" "+str(n+1)+" "+str(n+2)+"\n"
        if at=="CA":
            bondtype=str(3*pos+3)
            row=row+bondtype+" "+str(n+1)+" "+str(n+3)+"\n"
        if at=="CB" or at=="2HA":
            bondtype=str(3*pos+2)
            row=row+bondtype+" "+str(n+1)+" "+str(n)+"\n"
        tf.write(row)
    tf.write("\n")
    tf.write("Angles\n")
    tf.write("\n")
    index=1
    for n in range(natoms-3):
        row=str(index)+" "
        at = atomtype[n]
        if at=="N":
            angletype=str(2)
            row=row+angletype+" "+str(n+1)+" "+str(n+2)+" "+str(n+4)+"\n"
            tf.write(row)
            index=index+1
        if at=="C":
            angletype=str(1)
            row=row+angletype+" "+str(n+1)+" "+str(n+2)+" "+str(n+3)+"\n"
            tf.write(row)
            index=index+1
        if at=="CA":
            angletype=str(3)
            row=row+angletype+" "+str(n+1)+" "+str(n+3)+" "+str(n+4)+"\n"
            tf.write(row)
            index=index+1
        
    #now add angles involving CBs/2HA
    for n in range(natoms-2):
        row=str(index)+" "
        at = atomtype[n]
        if at=="N":
            angletype=str(4)
            row=row+angletype+" "+str(n+1)+" "+str(n+2)+" "+str(n+3)+"\n"
            tf.write(row)
            index=index+1
        if at=="CB" or at=="2HA":
            angletype=str(5)
            row=row+angletype+" "+str(n+1)+" "+str(n)+" "+str(n+2)+"\n"
            tf.write(row)
            index=index+1
    tf.write("\n")
    tf.write("Dihedrals\n")
    tf.write("\n")
    temp=1
    #this writes all omega angles
    for n in range(natoms-4):
        if atomtype[n]=="CA":
            row=str(temp)+" 1"
            row=row+" "+str(n+1)+" "+str(n+3)+" "+str(n+4)+" "+str(n+5)+"\n"
            temp +=1
            tf.write(row)
    #now add all phi, psi
    temp2=2
    for n in range(natoms):
        #adds psi
        if atomtype[n]=="N":
            row=str(temp)+" "+str(temp2)+" "
            row=row+str(n+1)+" "+str(n+2)+" "+str(n+4)+" "+str(n+5)+"\n"
            if n+4 < natoms:
                tf.write(row)
                temp+=1
                temp2+=1
        #adds phi
        if atomtype[n]=="C":
            row=str(temp)+" "+str(temp2)+" "
            row=row+str(n+1)+" "+str(n+2)+" "+str(n+3)+" "+str(n+5)+"\n"
            if n+4 < natoms:
                tf.write(row)
                temp+=1
                temp2+=1
    tf.write("\n")
    tf.write("CMAP\n")
    tf.write("\n")
    temp=1
    #read data necessary to calculate CMAPs
    ref_rama = np.genfromtxt(REFfile)
    #ref_rama = ndimage.rotate(ref_rama, -90)
    #FMangles = np.genfromtxt(FMfile)
    #NpointsFM = int(len(FMangles)/nres)
    u = mda.Universe(PDB, XTC)
    
    delta = int(360/NCMAP)
    bin_i = [n for n in range(-180, 179, delta)]
    #write CMAP file
    with open("CG.cmap", "w") as tf2:
        tf2.write("# DATE: "+d1+" CONTRIBUTOR: nsalvi CITATION: TBA\n")
        tf2.write("# Title: residue-specific dihedral correction map created by make_input_4BR.py\n")
        tf2.write("\n")
        for n in range(natoms):
            if atomtype[n]=="C":
                row=str(temp)+" "+str(temp)+" "
                row=row+str(n+1)+" "+str(n+2)+" "+str(n+3)+" "+str(n+5)+" "+str(n+6)+"\n"
                if n+6 <= natoms:
                    tf.write(row)
                    tf2.write("# residue "+str(sequence[n+1].decode())+str(res_numbers[n+1])+", type "+str(temp)+"\n")
                    tf2.write("\n")
                    r = u.select_atoms("resid "+str(res_numbers[n+1]))
                    R = Ramachandran(r).run(step=1)
                    FMphi = R.angles[:, :, 0].flatten()
                    FMpsi = R.angles[:, :, 1].flatten()
                    NpointsFM = len(FMpsi)
#                     #FMphi=FMangles[NpointsFM*(res_numbers[n+1]-1):NpointsFM*res_numbers[n+1], 0]
#                     #FMpsi=FMangles[NpointsFM*(res_numbers[n+1]-1):NpointsFM*res_numbers[n+1], 1]
                    indsFMphi = np.digitize(FMphi, bin_i)

                    FM_rama = []
                    
                    for bcounter, bi in enumerate(bin_i): #loop over psi, calculate values
                        selectedFMpsi = [el for acounter, el in enumerate(FMpsi) if indsFMphi[acounter]==bcounter+1]
                        indsFMpsi = np.digitize(selectedFMpsi, bin_i)
                        unique, counts = np.unique(indsFMpsi, return_counts=True)
                        counts = dict(zip(unique, counts))
                        popFM = [counts[acounter+1]/NpointsFM if (acounter+1) in counts.keys() else 1e-5 #0 #1e-4#1e-12 
                                     for acounter, el in enumerate(bin_i)]
                        FM_rama.append(popFM)
                    
                    #convolve with Gaussian kernel
                    FM_rama = convolve(FM_rama, Gaussian2DKernel(x_stddev=0.3, y_stddev=0.3), boundary='extend')
                    FM_rama = np.abs(FM_rama)
                    FM_rama = np.array(FM_rama)/np.sum(np.sum(FM_rama))
                    
                    all_values = [[-RT*np.log(a/b) for a, b in zip(FM_rama[bc], ref_rama[bc])] for bc in range(NCMAP)]
                    
#                     all_values = convolve(all_values, Gaussian2DKernel(x_stddev=0.15, y_stddev=0.15), boundary='extend')
#                     all_values = np.abs(all_values)
#                     all_values = np.array(all_values)/np.sum(np.sum(all_values))
                   
                    np.savetxt(str(res_numbers[n+1])+".cmap", all_values)
                    np.savetxt(str(res_numbers[n+1])+".rama", FM_rama)


                    for bcounter, bi in enumerate(bin_i): #loop over psi, write smoothed values to file
                        tf2.write("# "+str(bi)+"\n")
                        tf2.write("\n")
                        
                        values = all_values[bcounter]
                        row =""
                        for acounter in range(len(bin_i)):
                            row=row+"{:.6f}".format(values[acounter])+" "
                            if (acounter + 1) % 5 == 0:
                                row = row + "\n"
                                tf2.write(row)
                                row = ""
                            if (acounter + 1) == len(bin_i):
                                row = row + "\n"
                                tf2.write(row)
                                row = ""
                        tf2.write("\n")
                    temp += 1
                        

with open("in.CG", "w") as tf:
    tf.write("# test 4BR \n")
    tf.write("\n")
    tf.write("units    real\n")
    tf.write("atom_style    full\n")
    tf.write("region mybox block "+str(-hs)+" "+str(hs)+" "+str(-hs)+" "+str(hs)+" "+str(-hs)+" "+str(hs)+"\n")
    tf.write("create_box    "+str(nattypes)+" mybox bond/types "+str(nbondtypes)+" angle/types "+str(nangletypes)+" dihedral/types "+str(ndihs)+" extra/bond/per/atom 3 extra/angle/per/atom 3 extra/dihedral/per/atom 2"+"\n")
    tf.write("\n")
    tf.write("special_bonds charmm \n")
    kappa = 3.04/np.sqrt(2*conc)
    
    nrestype = len(list(set(sequence)))
    tf.write("pair_style hybrid/overlay table linear "+str(NP)+" coul/debye "+str(1./kappa)+" "+str(CO_charges)+"\n")
    with open("Ashbaugh-Hatch.table", "w") as tf2:
        tf2.write("# DATE: "+d1+" UNITS: real  CONTRIBUTOR: nsalvi\n")
        tf2.write("# Ashbaugh-Hatch potentials\n")
        tf2.write("\n")
        for n in range(nattypes):
            if n < nrestype: #condition to be a CA
                r1 = 3.32 #CT1+HB1 in C36
                e1 = epsilon
                l1 = 0.0
                el1 = list(set(sequence))[n].decode()+"_CA"
            if n >= nrestype and n < 2*nrestype: #condition to be a CB
                el1 = list(set(sequence))[n-nrestype]
                type1 = Bio.SeqUtils.IUPACData.protein_letters_3to1[el1.decode().title()]
                r1 = r0_dict[type1]
                e1 = epsilon_dict[type1]
                l1 = lambda_dict[type1]
                el1 = list(set(sequence))[n-nrestype].decode()+"_CB"
            if n >= 2*nrestype: #condition to be an N or C'
                if n-2*nrestype==1:
                    r1 = 3.7 #C+O in C36
                else:
                    r1 = 2.0745 #NH1+H in C36
                e1 = epsilon
                l1 = 0.0
                if n-2*nrestype==0:
                    el1 = "gen_N"
                if n-2*nrestype==1:
                    el1 = "gen_C"
                if n-2*nrestype==2:
                    el1 = "gen_NPRO"
                
            for m in range(nattypes):
                if m>=n:
                    if m < nrestype: #condition to be a CA
                        r2 = 3.32 #CT1+HB1 in C36
                        e2 = epsilon
                        l2 = 0.0
                        el2 = list(set(sequence))[m].decode()+"_CA"
                    if m >= nrestype and m < 2*nrestype: #condition to be a CB
                        el2 = list(set(sequence))[m-nrestype]
                        type2 = Bio.SeqUtils.IUPACData.protein_letters_3to1[el2.decode().title()]
                        r2 = r0_dict[type2]
                        e2 = epsilon_dict[type2]
                        l2 = lambda_dict[type2]
                        el2 = list(set(sequence))[m-nrestype].decode()+"_CB"
                    if m >= 2*nrestype: #condition to be an N or C'
                        if n-2*nrestype==1:
                            r2 = 3.7 #C+O in C36
                        else:
                            r2 = 2.0745 #NH1+H in C36
                        e2 = epsilon
                        l2 = 0.0
                        if m-2*nrestype==0:
                            el2 = "gen_N"
                        if m-2*nrestype==1:
                            el2 = "gen_C"
                        if m-2*nrestype==2:
                            el2 = "gen_NPRO"
                            
                    entry = el1+"_"+el2
                    
                    tf.write("pair_coeff "+str(n+1)+" "+str(m+1)+" table Ashbaugh-Hatch.table "+entry+" "+str(CO_NB)+"\n")
                        
                    rij = 0.5*(r1+r2)*scaling #np.sqrt(r1*r2)
                    eij = 0.5*(e1+e2) #np.sqrt(e1*e2)
                    lij = 0.5*(l1+l2)
                        
                    tf2.write(entry+"\n")
                    tf2.write("N "+str(NP)+" R "+str(rminNB)+" "+str(rmaxNB)+"\n")
                    tf2.write("\n")
                    for point in range(NP):
                        #(index, r, energy, force)
                        index = point+1
                        r = rminNB+point/(NP-1)*(rmaxNB-rminNB)
                        if n < nrestype or m < nrestype or n >= 2*nrestype or m >= 2*nrestype:
                            E, F = rep8(r, eij, rij)
                        else:
#                             E, F = phinb(r, eij, rij, lij)
                            E, F = lj86(r, eij, rij)
                        tf2.write(str(index)+" "+str(r)+" "+str(E)+" "+str(F)+"\n")
                    tf2.write("\n")
                    tf2.write("\n")
            
    tf.write("pair_coeff * * coul/debye\n")

    tf.write("dielectric "+str(diel)+"\n")
    # add CMAP corrections
    tf.write("fix cmap all cmap CG.cmap\n")
    tf.write("fix_modify cmap energy yes\n")
    # add bond interactions
    tf.write("bond_style harmonic\n")
    # add angle terms 
    tf.write("angle_style harmonic\n")
    # add omega angle
    tf.write("dihedral_style fourier\n")
    #tf.write("kspace_style pppm/cg 0.0001\n")
    tf.write("read_data data.CG add append fix cmap crossterm CMAP\n")
#     tf.write("read_data data.CG add append\n")
    tf.write("\n")
    tf.write("neighbor    2.0 bin\n")
    tf.write("neigh_modify    delay 5\n")
    tf.write("\n")
    tf.write("timestep    "+str(tstep)+"\n")
    tf.write("thermo_style    multi\n")
    tf.write("thermo    50\n")
    tf.write("\n")
    tf.write("minimize 1.0e-4 1.0e-6 10000 100000\n")
    tf.write("fix 1 all nve\n")
    tf.write("fix 2 all langevin "+str(T)+" "+str(T)+" "+str(langdamp)+" "+str(randint(1, 100000))+"\n")
    tf.write("\n")
    tf.write("dump 1 all dcd 250 traj.dcd\n")
    Nsteps = int(floor(trajT*1e6/tstep))
    tf.write("run    "+str(Nsteps)+"\n")

