#!/bin/sh

cut -c -79 "$1" | pdb_tidy | pdb_element | pdb_delelem -H | pdb_chain -A