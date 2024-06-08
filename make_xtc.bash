#!/usr/bin/env bash

cd ensemble || exit

for i in *a_132.pdb
do
	stem=$(basename "${i}" .pdb)
	grep ' CA \| C \| N \| CB \| 2HA' "${stem}.pdb" > "${stem}_CG.pdb"
	gmx trjconv -f "${stem}_CG.pdb" -o "${stem}.xtc" &> /dev/null
done


find . -maxdepth 1 -type f -name "*a_132.xtc" -exec gmx trjcat -o ensemble.xtc -cat -f {} +

# gmx trjcat -f ./*.xtc -o ensemble.xtc -cat
# gmx trjconv -f ensemble.xtc -o ensemble.pdb

find . -maxdepth 1 -type f -name "*a_132.xtc" -delete

# cd ..
