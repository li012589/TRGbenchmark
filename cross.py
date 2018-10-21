import torch
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="")

parser.add_argument("-J",type = float,defautl=1,help="")
parser.add_argument("-T",type = float,defautl=1,help="")
parser.add_argument("-iters",type = int,defautl=20,help="")
parser.add_argument("-cut", type = int, default=20,help="")

args = parser.parse_args()

K = torch.tensor([1/args.T*args.J])

M = torch.tensor([[torch.exp(K),torch.exp(-K)],[torch.exp(-K),torch.exp(K)]])

U,S,V = torch.svd(M)

P = torch.matmul(U,torch.diag(torch.sqrt(S)))

PT = torch.matmul(torch.diag(torch.sqrt(S)),V.t())

T  = torch.einsum('ai,aj,ka,la->ijkl',(P,P,PT,PT))

lnZ = torch.zeros(1)

cut = args.cut
epsilon=0

for n in range(args.iters):
    maxT = T.max()
    T = T/maxT
    lnZ += 2**(args.iters-n)*torch.log(maxT)

    TypeA = T.view(2**2,2**2)

    TypeB = T.permute(0,3,1,2).contiguous().view(2**2,2**2)

    Ua,Sa,Va = torch.svd(TypeA)
    Ub,Sb,Vb = torch.svd(TypeB)

    D = min(min(2**2,cut),min((Sa>epsilon).sum().item(),(Sb>epsilon).sum().item()))

    S2 = torch.matmul(Ub,torch.diag(torch.sqrt(Sb))).view(2,2,4)
    S1 = torch.matmul(torch.diag(torch.sqrt(Sb)),Vb.t()).view(4,2,2)
    S3 = torch.matmul(Ua,torch.diag(torch.sqrt(Sa))).view(2,2,4)
    S4 = torch.matmul(torch.diag(torch.sqrt(Sa)),Va.t()).view(4,2,2)

    T = torch.einsum('abc,bde,dfg,ncf->aegn',(S1,S3,S2,S4))