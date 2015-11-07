#!/usr/bin/env python

"""A simple python script template.

"""

import os
import sys
import argparse


def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('infile', help="Input file", type=argparse.FileType('r'))
    parser.add_argument('-o', '--outfile', help="Output file",
                        default=sys.stdout, type=argparse.FileType('w'), dest = "outfile")

    args = parser.parse_args(arguments)

    for line in args.infile:
        fields = line.strip().split(',')
        index = map(str, range(1,len(fields)))
        
        out = [":".join(i) for i in zip(index, fields[1:])]

        args.outfile.write(' '.join([fields[0]] + out) + '\n')
        
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
