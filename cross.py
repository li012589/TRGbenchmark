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

M = torch.tensor([[torch.exp(K),torch.exp(-K)],[torch.exp(-K),torch.exp(K)]])

U,S,V = torch.svd(M)

P = torch.matmul(U,torch.diag(torch.sqrt(S)))

PT = torch.matmul(torch.diag(torch.sqrt(S)),V.t())

T = torch.einsum('ai,aj,ka,la->ijkl',(P,P,PT,PT))

lnZ = torch.zeros(1)

maxCut = args.cut
D = 2
epsilon=0

for n in range(args.iters):
    maxT = T.max()
    T = T/maxT
    lnZ += 2**(args.iters-n)*torch.log(maxT)

    TypeA = T.view(D**2,D**2)

    TypeB = T.permute(0,3,1,2).contiguous().view(D**2,D**2)

    Ua,Sa,Va = torch.svd(TypeA)
    Ub,Sb,Vb = torch.svd(TypeB)

    cut = min(maxCut,min((Sa>epsilon).sum().item(),(Sb>epsilon).sum().item()))
    #print(cut)
    #print(maxT)

    S2 = torch.matmul(Ub[:,:cut],torch.diag(torch.sqrt(Sb[:cut]))).view(D,D,cut)
    S1 = torch.matmul(torch.diag(torch.sqrt(Sb[:cut])),Vb.t()[:cut,:]).view(cut,D,D)
    S3 = torch.matmul(Ua[:,:cut],torch.diag(torch.sqrt(Sa[:cut]))).view(D,D,cut)
    S4 = torch.matmul(torch.diag(torch.sqrt(Sa[:cut])),Va.t()[:cut,:]).view(cut,D,D)

    T = torch.einsum('abc,cde,fdg,nfb->aegn',(S1,S3,S2,S4))
    D = cut
    #import pdb
    #pdb.set_trace()

trace = 0
for i in range(D):
    trace += T[i,i,i,i]
    #print(trace)
#print(lnZ)
lnZ += torch.log(trace)

print("K:",K.item())
print("lnZ:",lnZ.item()/2**args.iters)