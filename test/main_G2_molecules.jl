using Printf
using Random

using CUDA
using PWDFT
using PWDFT_cuda

const DIR_PWDFT = joinpath(dirname(pathof(PWDFT)),"..")
const DIR_PSP = joinpath(DIR_PWDFT, "pseudopotentials", "pade_gth")
const DIR_STRUCTURES = joinpath(DIR_PWDFT, "structures")

include("../../get_default_psp.jl")

function main_CPU(molname)

    Random.seed!(1234)

    filename = joinpath(DIR_STRUCTURES, "DATA_G2_mols", molname*".xyz")
    atoms = Atoms(ext_xyz_file=filename)
    pspfiles = get_default_psp(atoms)
    ecutwfc = 15.0

    Nspin = 1
    
    Ham_cpu = Hamiltonian( atoms, pspfiles, ecutwfc, use_symmetry=false, Nspin=Nspin )
    println( Ham_cpu )

    KS_solve_Emin_PCG!( Ham_cpu, skip_initial_diag=true, startingrhoe=:random )
    
    @time KS_solve_Emin_PCG!( Ham_cpu, skip_initial_diag=true, startingrhoe=:random )

end

function main_GPU(molname)

    Random.seed!(1234)

    filename = joinpath(DIR_STRUCTURES, "DATA_G2_mols", molname*".xyz")
    atoms = Atoms(ext_xyz_file=filename)
    pspfiles = get_default_psp(atoms)

    ecutwfc = 15.0

    Nspin = 1

    Ham = CuHamiltonian( atoms, pspfiles, ecutwfc, use_symmetry=false, Nspin=Nspin )
    KS_solve_Emin_PCG!( Ham, skip_initial_diag=true, startingrhoe=:random )
    
    @time KS_solve_Emin_PCG!( Ham, skip_initial_diag=true, startingrhoe=:random )

end


function main()
    Nargs = length(ARGS)
    if Nargs >= 1
        molname = ARGS[1]
    else
        molname = "H2O"
    end
    main_CPU(molname)
    main_GPU(molname)
end

main()

