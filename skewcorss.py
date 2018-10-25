import torch
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="")

parser.add_argument("-J",type = float,default=1,help="")
parser.add_argument("-T",type = float,default=1,help="")
parser.add_argument("-iters",type = int,default=20,help="")
parser.add_argument("-cut", type = int, default=20,help="")
parser.add_argument("-test",action = 'store_true',help="")

args = parser.parse_args()
torch.set_grad_enabled(False)

K = torch.tensor([1/args.T*args.J],dtype= torch.float64)

T = torch.empty(2,2,2,2)
ran = [1,-1]
for i0,s0 in enumerate(ran):
    for i1,s1 in enumerate(ran):
        for i2,s2 in enumerate(ran):
            for i3,s3 in enumerate(ran):
                T[i0,i1,i2,i3]=torch.exp(K*(s0*s1+s1*s2+s2*s3+s3*s0))

if args.test:
    print("test")
    M = torch.tensor([[torch.exp(K),torch.exp(-K)],[torch.exp(-K),torch.exp(K)]])
    D3 = torch.zeros(2,2,2)
    for i in range(2):
        D3[i,i,i]=1

    Ttest = torch.einsum("abc,bd,efd,fg,ghq,jq,ijk,ci->aehk",(D3,M,D3,M,D3,M,D3,M))
    Ttestp = torch.einsum("bd,dg,gq,qb->bdgq",(M,M,M,M))
    from numpy.testing import assert_array_almost_equal,assert_array_equal
    assert_array_almost_equal(T.numpy(),Ttest.numpy(),5)
    assert_array_almost_equal(Ttest.numpy(),Ttestp.numpy(),5)

maxCut = args.cut
D = 2
epsilon=0
lnZ = torch.zeros(1)

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

trace = torch.einsum("aaaa->",T)

lnZ += torch.log(trace)

print("K:",K.item())
print("lnZ:",lnZ.item()/2**(args.iters+1))

