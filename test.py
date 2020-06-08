#!/usr/bin/env python
import sys
from os import path

if (len(sys.argv) != 2):
    print("usage: python test.py <model-file-name.h5>")
    print("for example: python test.py model-00100000.h5")
    exit()

filename = sys.argv[1]
if path.exists(filename):
    from qlearn import q_learning
    q_learning("test", filename)
else:
    print('file does not exist')
