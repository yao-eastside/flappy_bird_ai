#!/usr/bin/env python
import sys

if (len(sys.argv) != 1):
    exit()

from qlearn import q_learning
q_learning("train")
