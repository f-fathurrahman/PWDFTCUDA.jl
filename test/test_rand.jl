using Random
using CUDA

Random.seed!(1234)
CUDA.CURAND.seed!(1234)

function gen_rand()
    v = rand(ComplexF64,2,2)
    return v
end

function cu_gen_rand()
    v = CUDA.CURAND.rand(ComplexF64,2,2)
    return v
end

function main()
    v_cpu = gen_rand()
    v_gpu = cu_gen_rand()

    println(v_cpu)
    println(v_gpu)
end

main()