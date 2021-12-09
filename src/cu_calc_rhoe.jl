import PWDFT: calc_rhoe, calc_rhoe!

function calc_rhoe!( Ham::CuHamiltonian, psis::CuBlochWavefunc, Rhoe::CuArray{Float64,2}; renormalize=true )

    pw = Ham.pw
    Focc = Ham.electrons.Focc
    Nspin = Ham.electrons.Nspin
    Nelectrons_true = Ham.electrons.Nelectrons

    CellVolume  = pw.CellVolume
    Ns = pw.Ns
    Nkpt = pw.gvecw.kpoints.Nkpt
    Ngw = pw.gvecw.Ngw
    wk = pw.gvecw.kpoints.wk
    Npoints = prod(Ns)
    Nstates = size(psis[1])[2]

    ctmp = CUDA.zeros(ComplexF64, Npoints)

    # dont forget to zero out the Rhoe first
    fill!(Rhoe, 0.0)
    NptsPerSqrtVol = Npoints/sqrt(CellVolume)
    
    Nthreads = 256
    for ispin in 1:Nspin, ik in 1:Nkpt
        i = ik + (ispin - 1)*Nkpt
        idx = pw.gvecw.idx_gw2r[ik]
        Nblocks = ceil( Int64, Ngw[ik]/Nthreads )
        for ist in 1:Nstates
            fill!(ctmp, 0.0 + im*0.0)
            @views psii = psis[i][:,ist]
            @cuda threads=Nthreads blocks=Nblocks kernel_copy_to_fft_grid_gw2r_1state!( idx, psii, ctmp )
            # Transform to real space
            G_to_R!(pw, ctmp)
            ctmp[:] .= NptsPerSqrtVol*ctmp[:]
            w = wk[ik]*Focc[ist,i]
            @views Rhoe[:,ispin] .+= w*real( conj(ctmp) .* ctmp )
        end
    end

    # renormalize
    if renormalize
        integ_rho = sum(Rhoe)*CellVolume/Npoints
        Rhoe[:] = Nelectrons_true/integ_rho * Rhoe[:]
    end

    #
    # XXX This is rather difficult to parallelize
    #
    # Symmetrize Rhoe if needed
    #if Ham.sym_info.Nsyms > 1
    #    symmetrize_rhoe!( Ham.pw, Ham.sym_info, Ham.rhoe_symmetrizer, Rhoe )
    #end

    return
end

function calc_rhoe( Ham::CuHamiltonian, psis::CuBlochWavefunc )
    Npoints = prod(Ham.pw.Ns)
    Nspin = Ham.electrons.Nspin
    Rhoe = CUDA.zeros(Float64, Npoints, Nspin)
    calc_rhoe!( Ham, psis, Rhoe )
    return Rhoe
end


