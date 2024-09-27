import parmed as pmd
from parmed.amber import readparm


# amber = pmd.load_file('0.15_80_10_pH7_6921a_all.leap.top', '0.15_80_10_pH7_6921a_all.leap.crd')
# amber = pmd.load_file('RAMP1_ion.prmtop', xyz='RAMP1_ion.inpcrd')
# parm = readparm.AmberParm('RAMP1_ion.prmtop', 'RAMP1_ion.inpcrd')

parm = readparm.AmberParm('0.15_80_10_pH7_6921a_all.leap.prmtop', '0.15_80_10_pH7_6921a_all.leap.inpcrd')
parm.save('charmm.psf', overwrite=True)
parm.save('charmm.crd', overwrite=True)

pmd.charmm.CharmmParameterSet.from_structure(parm).write(top='top_charmm.rtf', par='par_charmm.prm')