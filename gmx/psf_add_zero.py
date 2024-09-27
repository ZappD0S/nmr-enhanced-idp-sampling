import sys
from parse import parse, compile
import os.path


# psf_fmt = "{:10d} {:<8w} {:<8d} {:<8w} {:<8w} {:<8w}{:8.6f} {:13.4f}" + 11 * " "
psf_fmt = "{:10d} {:<8w} {:<8d} {:<8w} {:<8w} {:<8w}{:9.6f} {:13.4f}" + 11 * " "
psf_parser = compile(psf_fmt)
psf_fmt = psf_fmt.replace("w", "")

# dst_fmt = "{:10d} {:8w} {:8d} {:8w} {:8w} {:6w} {:10.6f} {:13.4f} {:10}" + 11 * " "
# "         6 SYS      1        MET      HA       HB1      0.100000        1.0080           "
# "       996 SYS      64       SER      N        NH1     -0.470000       14.0070           "
# '      9994 SYS      2798     SOL      OW       OT      -0.834000       15.9994           '
# '         1 SYS      1        MET      N        NH3     -0.300000       14.0070           0
# '         2 SYS      1        MET      H1       HC      0.330000        1.0080           0'

crd_fmt = "  {:8d}  {:8d}  {:<8w}  {:<8w}  {:18.10f}  {:18.10f}  {:18.10f}  {:<8w}  {:<8d}  {:18.10f}"
crd_paser = compile(crd_fmt)
crd_fmt = crd_fmt.replace("w", "")

name = sys.argv[1]

in_psf_file = name + ".psf"
in_crd_file = name + ".crd"

assert os.path.exists(in_psf_file)
assert os.path.exists(in_crd_file)

out_psf_file = name + "_fixed.psf"
out_crd_file = name + "_fixed.crd"

with open(in_psf_file) as in_f, open(out_psf_file, "w") as out_f:
    for line in in_f:
        # line = line.rstrip("\n")
        line = line[:-1]
        if (match := psf_parser.parse(line)) is not None:
            new_vals = list(match)
            if new_vals[3].strip() == "SOL":
                new_vals[3] = "TIP3"
                new_vals[1] = "WAT"

            if new_vals[3].strip() in {"SOD", "CLA"}:
                new_vals[1] = "ION"

            # if new_vals[3].strip() == "NA":
            #     new_vals[3] = "SOD"

            # if new_vals[3].strip() == "CL":
            #     new_vals[3] = "CLA"

            print(psf_fmt.format(*new_vals) + "0", file=out_f)
        else:
            print(line, file=out_f)

with open(in_crd_file) as in_f, open(out_crd_file, "w") as out_f:
    for line in in_f:
        # line = line.rstrip("\n")
        line = line[:-1]
        if (match := crd_paser.parse(line)) is not None:
            new_vals = list(match)

            if new_vals[2].strip() == "SOL":
                new_vals[2] = "TIP3"
                new_vals[-3] = "WAT"

            if new_vals[2].strip() in {"SOD", "CLA"}:
                new_vals[-3] = "ION"

            # if new_vals[3].strip() == "NA":
            #     new_vals[3] = "SOD"

            # if new_vals[3].strip() == "CL":
            #     new_vals[3] = "CLA"

            new_vals[-2] += 1
            print(crd_fmt.format(*new_vals), file=out_f)
        else:
            print(line, file=out_f)
