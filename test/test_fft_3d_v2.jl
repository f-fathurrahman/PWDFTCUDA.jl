using Printf
using CUDA
using PWDFT
using PWDFTCUDA

function main_CPU(; ecutwfc=15.0)
    pw = PWGrid(ecutwfc, gen_lattice_sc(20.0))
    Npoints = prod(pw.Ns)
    Rhoe = rand(ComplexF64,Npoints)
    @printf("%8d %8d %8d CPU: ", pw.gvec.Ng, pw.Ns[1], Npoints)
    R_to_G!(pw, Rhoe) # warm up
    @time R_to_G!(pw, Rhoe)
end

function main_GPU(; ecutwfc=15.0)
    pw = CuPWGrid(ecutwfc, gen_lattice_sc(20.0))
    Npoints = prod(pw.Ns)
    Rhoe = CuArray(rand(ComplexF64,Npoints))
    @printf("%8d %8d %8d GPU: ", pw.gvec.Ng, pw.Ns[1], Npoints)
    R_to_G!(pw, Rhoe) # warm up
    @time CUDA.@sync R_to_G!(pw, Rhoe)
end

for ecut in 30.0:5.0:60.0
    println("ecut = ", ecut)
    main_CPU(ecutwfc=ecut)
    main_GPU(ecutwfc=ecut)
end
