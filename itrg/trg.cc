#include "itensor/all_basic.h"
#include <string>
#include <stdio.h>

using namespace itensor;

int main(int argc, char* argv[])
{
    double T;
    int maxm;
    int topscale;

    if(argc == 4)
    {
        std::cout<<"using parse"<<std::endl;
        sscanf(argv[1], "%lf", &T);
        sscanf(argv[2], "%d", &maxm);
        sscanf(argv[3], "%d", &topscale);
    }
    else
    {
        std::cout<<"using default"<<std::endl;
        T = 1.;
        maxm = 20;
        topscale = 20;
    }

auto m0 = 2;
auto x = Index("x0",m0,Xtype);
auto y = Index("y0",m0,Ytype);
auto x2 = prime(x,2);
auto y2 = prime(y,2);

auto A = ITensor(x,y2,x2,y);

auto Sig = [](int s) { return 1.-2.*(s-1); };

auto E0 = -0.;

for(auto s1 : range1(m0))
for(auto s2 : range1(m0))
for(auto s3 : range1(m0))
for(auto s4 : range1(m0))
    {
    auto E = Sig(s1)*Sig(s2)+Sig(s2)*Sig(s3)
            +Sig(s3)*Sig(s4)+Sig(s4)*Sig(s1);
    auto val = exp(-(E-E0)*T);
    A.set(x(s1),y2(s2),x2(s3),y(s4),val);
    }

auto maxVal = 0.0;

auto lnZ = 0.0;
auto sumA=0.0;

for(auto scale : range(topscale))
    {
    printfln("\n---------- Scale %d -> %d  ----------",scale,1+scale);

    auto tmp_inds = A.inds();
    maxVal = 0.0;
    for (auto i0 : range1(tmp_inds[0].m()))
    for (auto i1 : range1(tmp_inds[1].m()))
    for (auto i2 : range1(tmp_inds[2].m()))
    for (auto i3 : range1(tmp_inds[3].m()))
    {
        if(A.real(tmp_inds[0](i0),tmp_inds[1](i1),tmp_inds[2](i2),tmp_inds[3](i3))>maxVal)
        maxVal = A.real(tmp_inds[0](i0),tmp_inds[1](i1),tmp_inds[2](i2),tmp_inds[3](i3));
    }

    if(maxVal != 0)
    {
        lnZ += pow(2,topscale-scale)*log(maxVal);
        A = A/maxVal;
    }

    auto y = noprime(findtype(A,Ytype));
    auto y2 = prime(y,2);
    auto x = noprime(findtype(A,Xtype));
    auto x2 = prime(x,2);

    auto F1 = ITensor(x2,y);
    auto F3 = ITensor(x,y2);
    auto xname = format("x%d",scale+1);
    factor(A,F1,F3,{"Maxm=",maxm,"ShowEigs=",true,
                    "IndexType=",Xtype,"IndexName=",xname});

    auto F2 = ITensor(x,y);
    auto F4 = ITensor(y2,x2);
    auto yname = format("y%d",scale+1);
    factor(A,F2,F4,{"Maxm=",maxm,"ShowEigs=",true,
                    "IndexType=",Ytype,"IndexName=",yname});

    auto l13 = commonIndex(F1,F3);
    A = F1 * noprime(F4) * prime(F2,2) * prime(F3,l13,2);

    }

println("\n---------- Calculating at Scale ",topscale," ----------");

auto xt = noprime(findtype(A,Xtype));
auto yt = noprime(findtype(A,Ytype));
auto xt2 = prime(xt,2);
auto yt2 = prime(yt,2);

auto Trx = delta(xt,xt2);
auto Try = delta(yt,yt2);
auto Z = (Trx*A*Try).real();

Real Ns = pow(2,1+topscale);

printfln("log(Z)/Ns = %.12f",(log(Z)+lnZ)/Ns);

return 0;
}