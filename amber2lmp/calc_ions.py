from scipy.constants import Avogadro

vol_ang = 1460338.886 # angstrom ^ 3
salt_molarity = 0.15 # M = mol/L

vol_l = vol_ang / (1e10) ** 3 * 1000

ions_per_l = salt_molarity * Avogadro

print(round(vol_l * ions_per_l))