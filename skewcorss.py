import torch
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="")

parser.add_argument("-J",type = float,default=1,help="")
parser.add_argument("-T",type = float,default=1,help="")
parser.add_argument("-iters",type = int,default=20,help="")
parser.add_argument("-cut", type = int, default=20,help="")

args = parser.parse_args()

K = torch.tensor([1/args.T*args.J])

T = torch.empty(2,2,2,2)
ran = [1,-1]
for i0,s0 in enumerate(ran):
    for i1,s1 in enumerate(ran):
        for i2,s2 in enumerate(ran):
            for i3,s3 in enumerate(ran):
                T[i0,i1,i2,i3]=torch.exp(K*(s0*s1+s1*s2+s2*s3+s3*s0))