#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description='Deep Reinforcement Learning for Flappy Bird on macOS')
parser.add_argument('-m','--mode', help='train | test', required=True)
args = vars(parser.parse_args())
mode = None
if args['mode'] == 'test':
    mode = 'test'
elif args['mode'] == 'train':
    mode = 'train'
else:
    parser.print_help()
    exit()

from qlearn import trainNetwork
trainNetwork(mode)
