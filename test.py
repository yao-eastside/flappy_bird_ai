#!/usr/bin/env python
import sys

if (len(sys.argv) != 2):
    print("usage: python test.py <model-file-name.h5>")
    print("for example: python test.py model-00100000.h5")
    exit()

try:
    filename = sys.argv[1]
    with open(filename, 'r') as fh:
        from qlearn import q_learning
        q_learning("test", fh)
except (OSError, IOError) as e:
    print(e)
