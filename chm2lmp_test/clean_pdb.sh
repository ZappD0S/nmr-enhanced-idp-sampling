#!/bin/sh

cut -c -79 "$1" | pdb_tidy | pdb_element | pdb_chain -A