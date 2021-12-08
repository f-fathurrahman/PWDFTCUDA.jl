import PWDFT: G_to_R, G_to_R!, R_to_G, R_to_G!
import GPUArrays: AbstractGPUArray

#
# In-place version, input 3d data as 3d array
#
function G_to_R!( pw, fG::CuArray{ComplexF64,3} )
    pw.planbw*fG
    return
end

function R_to_G!( pw, fR::CuArray{ComplexF64,3} )
    pw.planfw*fR
    return
end

#
# Return new array, input 3d data as 3d array
#
function G_to_R( pw, fG::CuArray{ComplexF64,3} )
    ff = copy(fG)
    pw.planbw*ff
    return ff
end

function R_to_G( pw, fR::CuArray{ComplexF64,3} )
    ff = copy(fR)
    pw.planfw*ff
    return ff
end


#
# In-place version, input 3d data as column vector
#
function G_to_R!( pw, fG::AbstractGPUArray{ComplexF64,1} )
    ff = reshape(fG, pw.Ns)
    pw.planbw*ff
    return
end

function R_to_G!( pw, fR::AbstractGPUArray{ComplexF64,1} )
    ff = reshape(fR, pw.Ns)
    pw.planfw*ff
    return
end


#
# Return a new array
#
function G_to_R( pw, fG::AbstractGPUArray{ComplexF64,1} )
    ff = copy(fG)
    ff = reshape(ff, pw.Ns)
    pw.planbw*ff
    return reshape(ff, prod(pw.Ns))
end

function R_to_G( pw, fR::AbstractGPUArray{ComplexF64,1} )
    ff = copy(fR)
    ff = reshape(fR, pw.Ns)
    pw.planfw*ff
    return reshape(ff, prod(pw.Ns))
end


#
# used in Poisson solver
#
function R_to_G( pw, fR_::AbstractGPUArray{Float64,1} )
    fR = convert(CuArray{ComplexF64,1}, fR_) # This will make a copy
    ff = reshape(fR, pw.Ns)
    pw.planfw*ff
    return reshape(ff, prod(pw.Ns))
end


#
# Used in calc_rhoe
#
function G_to_R!( pw, fG::CuMatrix{ComplexF64} )
    for i in 1:size(fG,2)
        @views ff = reshape(fG[:,i], pw.Ns)
        pw.planbw*ff
    end
    return
end

# Used op_V_loc, multicolumn
function R_to_G!( pw, fR::CuArray{ComplexF64,2} )
    for i in 1:size(fR,2)
        @views ff = reshape(fR[:,i], pw.Ns)
        pw.planfw*ff
    end
    return
end