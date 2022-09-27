import sys
import os
import parmed as pmd

name = sys.argv[1]

in_gro_file = name + ".gro"

assert os.path.exists(in_gro_file)

gmx_top = pmd.load_file("topol.top", xyz=in_gro_file)

# gmx_top.save('6921a_solv_ions.psf', overwrite=True)
s = pmd.charmm.CharmmPsfFile.from_structure(gmx_top)
s.flags = {"EXT", "XPLOR"}
s.write_psf(name + ".psf")


# crd = pmd.charmm.CharmmCrdFile.write(gmx_top, fname, **kwargs)
# gmx_top.save('6921a_solv_ions.psf', overwrite=True, vmd=True)
gmx_top.save(name + ".crd", overwrite=True)

# pmd.charmm.CharmmParameterSet.from_structure(gmx_top).write(
#     top="top_charmm.rtf", par="par_charmm.prm"
# )
