function cu_Ylm_real( l::Int64, m::Int64, R1, R2, R3 )

    SMALL = 1.0e-9

    Rmod = CUDA.sqrt( R1^2 + R2^2 + R3^2 )
    if Rmod < SMALL
        cost = 0.0
    else
        cost = R3/Rmod
    end

    #
    # beware the arc tan, it is defined modulo pi
    #
    if R1 > SMALL
        phi = CUDA.atan( R2/R1 )
    elseif R1 < -SMALL
        phi = CUDA.atan( R2/R1 ) + pi
    else
        phi = if R2 >= 0 pi/2 else -pi/2 end
    end
    sint = CUDA.sqrt( CUDA.max(0.0, 1.0 - cost^2) )

    ylm = 0.0

    if l == 0
        ylm = 0.5*CUDA.sqrt(1.0/Float64(pi))
        return ylm

    elseif l == 1
        # py
        if m == -1
            ylm = 0.5*CUDA.sqrt(3.0/pi)*sint*CUDA.sin(phi)
            return ylm
        # pz
        elseif m == 0
            ylm = 0.5*CUDA.sqrt(3.0/pi)*cost
            return ylm
        # px
        elseif m == 1
            ylm = 0.5*CUDA.sqrt(3.0/pi)*sint*CUDA.cos(phi)
            return ylm
        end

    elseif l == 2
        # dxy
        if m == -2
            ylm = CUDA.sqrt(15.0/16.0/pi) * sint^2 * CUDA.sin(2.0*phi)
            return ylm
        # dyz
        elseif m == -1
            ylm = CUDA.sqrt(15.0/4.0/pi)*cost*sint*CUDA.sin(phi)
            return ylm
        # dz2
        elseif m == 0
            ylm = 0.25*CUDA.sqrt(5.0/pi)*( 3.0*cost^2 - 1.0 )
            return ylm
        # dxz
        elseif m == 1
            ylm = CUDA.sqrt(15.0/4.0/pi)*cost*sint*CUDA.cos(phi)
            return ylm
        # dx2-y2
        elseif m == 2
            ylm = CUDA.sqrt(15.0/16.0/pi) * sint^2 * CUDA.cos(2.0*phi)
            return ylm
        end

    elseif l == 3

        if m == -3
            ylm = 0.25*CUDA.sqrt(35.0/2.0/pi) * sint^3 * CUDA.sin(3.0*phi)
            return ylm

        elseif m == -2
            ylm = 0.25*CUDA.sqrt(105.0/pi)* sint^2 *cost * CUDA.sin(2.0*phi)
            return ylm

        elseif m == -1
            ylm = 0.25*CUDA.sqrt(21.0/2.0/pi)*sint*( 5.0*cost^2 - 1.0 )*CUDA.sin(phi)
            return ylm

        elseif m == 0
            ylm = 0.25*CUDA.sqrt(7.0/pi)*( 5.0*cost^3 - 3.0*cost )
            return ylm

        elseif m == 1
            ylm = 0.25*CUDA.sqrt(21.0/2.0/pi)*sint*( 5.0*cost^2 - 1.0 )*CUDA.cos(phi)
            return ylm

        elseif m == 2
            ylm = 0.25*CUDA.sqrt(105.0/pi) * sint^2 * cost * CUDA.cos(2.0*phi)
            return ylm

        elseif m == 3
            ylm = 0.25*CUDA.sqrt(35.0/2.0/pi) * sint^3 * CUDA.cos(3.0*phi)
            return ylm
        end

    end

    return ylm

end
