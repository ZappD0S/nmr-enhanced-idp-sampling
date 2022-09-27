import io
import re
import sys
from pathlib import Path

def clean_pdb(input: io.TextIOBase, output: io.TextIOBase):
    pattern = re.compile(
        r"""^ATOM
            \s+
            \d+                         # atom index
            \s+
            \d?[A-Z]{1,2}\d?            # atom name
            \s+
            [A-Z]{3}                    # residue
            \s+
            \d+                         # residue id
            \s+
            (?:-?\d{1,3}\.\d{3}\s*){2}  # position
            -?\d{1,3}\.\d{3}            # position
            \s+
            \d\.\d{2}                   # other..
            \s+
            \d\.\d{2}                   # other..
            """,
        re.X,
    )

    for line in input:
        if (match := pattern.search(line)) is None:
            print(line)
            raise Exception

        output.write(match[0] + "\n")


raw_pdb = Path(sys.argv[1])
cleaned_pdb = raw_pdb.parent / (raw_pdb.stem + "_clean" + raw_pdb.suffix)

with raw_pdb.open("r") as input, cleaned_pdb.open("w") as output:
    clean_pdb(input, output)



