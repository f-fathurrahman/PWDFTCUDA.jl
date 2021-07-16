function get_default_psp( atoms::Atoms; xcfunc="VWN" )

    DIR_PWDFT = joinpath( dirname(pathof(PWDFT)), "..")
    if xcfunc == "VWN"
        DIR_PSP = joinpath(DIR_PWDFT, "pseudopotentials", "pade_gth")
        ALL_PSP = PWDFT.ALL_PADE_PSP
    elseif xcfunc == "PBE"
        DIR_PSP = joinpath(DIR_PWDFT, "pseudopotentials", "pbe_gth")
        ALL_PSP = PWDFT.ALL_PBE_PSP
    else
        errmsg = @sprintf("xcfunc in get_default_psp is not known %s\n", xcfunc)
        error(errmsg)
    end

    Nspecies = atoms.Nspecies
    pspfiles = Array{String}(undef,Nspecies)

    SpeciesSymbols = atoms.SpeciesSymbols
    for isp = 1:Nspecies
        atsymb = SpeciesSymbols[isp]
        pspfiles[isp] = joinpath(DIR_PSP, ALL_PSP[atsymb][1])
    end
    return pspfiles
end