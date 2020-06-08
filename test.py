#!/usr/bin/env python
import sys

if (len(sys.argv) != 2):
    exit()

try:
    filename = sys.argv[1]
    with open(filename, 'r') as fh:
        from qlearn import q_learning
        q_learning("test", fh)
except (OSError, IOError) as e:
    print(e)
