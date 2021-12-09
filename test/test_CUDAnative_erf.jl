using CUDA
using SpecialFunctions: erf

# CUDAnative is no longer needed here

function kernel_erf( vin, vout )
    idx = ( blockIdx().x - 1 )*blockDim().x + threadIdx().x
    N = length(vin)
    if idx <= N
        @inbounds vout[idx] = erf(vin[idx])
    end
    return
end

function test_erf()
    Ndata = 100_000

    Nthreads = 256
    Nblocks = ceil(Int64, Ndata/Nthreads)

    vin = CUDA.rand(Float64, Ndata)
    vout = CUDA.zeros(Float64, Ndata)

    @cuda threads=Nthreads blocks=Nblocks kernel_erf( vin, vout )

    println("Pass here")
end

test_erf()
