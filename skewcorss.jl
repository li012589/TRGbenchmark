using LinearAlgebra
using TensorOperations

function main()
    K = 1.0

    iteration = 20

    maxD = 20

    T = zeros(Float64,2,2,2,2)

    no2s(no::Int64) = -2*no+3 ::Int64
    sumS(s0::Int64,s1::Int64,s2::Int64,s3::Int64) = Float64(no2s(s0)*no2s(s1)+no2s(s1)*no2s(s2)+no2s(s2)*no2s(s3)+no2s(s3)*no2s(s0))

    for s0 = 1:2
        for s1 = 1:2
            for s2 = 1:2
                for s3 = 1:2
                    T[s3,s2,s1,s0] = exp(K*sumS(s3,s2,s1,s0))
                end
            end
        end
    end

    D = 2
    epsilon = 0.0

    checkBigger(M::Float64) = M>=epsilon

    lnZ = 0.0

    for n = 1:iteration
        maxVal = maximum(T)
        T /= maxVal
        lnZ += 2^(iteration-n+1)*log(maxVal)
        TypeA = reshape(T,(D^2,D^2))
        TypeB = reshape(permutedims(T,[1 4 2 3]),(D^2,D^2))
        ua,sa,va = svd(TypeA)
        ub,sb,vb = svd(TypeB)
        cut = min(size(filter(checkBigger,sb))[1],size(filter(checkBigger,sa))[1],maxD)
        S2 = reshape(ub[:,1:cut]*Diagonal(sqrt.(sb[1:cut])),(D,D,cut))
        S1 = reshape(Diagonal(sqrt.(sb[1:cut]))*transpose(vb)[1:cut,:],(cut,D,D))
        S3 = reshape(ua[:,1:cut]*Diagonal(sqrt.(sa[1:cut])),(D,D,cut))
        S4 = reshape(Diagonal(sqrt.(sa[1:cut]))*transpose(va)[1:cut,:],(cut,D,D))

        @tensor T[a,e,g,n] := S1[a,b,c]*S3[c,d,e]*S2[f,d,g]*S4[n,f,b]
        #T = Tp
        D = cut
    end

    trace = 0.0
    for i = 1:D
        trace += T[i,i,i,i]
    end
    lnZ += log(trace)

    println("K:",K)
    println("lnZ/N:",lnZ/(2^(iteration+1)))
end

main()
